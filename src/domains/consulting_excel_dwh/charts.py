"""Genera los gráficos del dominio de consultoría/data warehouse, todos vía la
librería reusable `src.toolkit.viz`.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.domains.consulting_excel_dwh.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.toolkit import viz

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "consulting"
WAREHOUSE_PATH = PROCESSED_DIR / "wdi_warehouse.duckdb"
REPORTS_DIR = ROOT / "outputs" / "consulting"
FIG_DIR = REPORTS_DIR / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((REPORTS_DIR / "clean_report.json").read_text(encoding="utf-8"))
    features_df = pd.read_csv(PROCESSED_DIR / "consulting_features.csv")
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))

    # 1. Funnel de filas a través del ETL: ancho crudo -> largo -> sin agregados -> tras interpolar.
    viz.plot_funnel(
        {
            "ancho (crudo, país×indicador)": report["filas_crudas_ancho"],
            "largo (país×indicador×año)": report["filas_largas_pre_filtro"],
            "sin agregados regionales": report["filas_tras_excluir_agregados"],
            "con valor real (pre-interpolar)": report["filas_con_valor_real_antes_de_interpolar"],
            "tras interpolar por serie": report["filas_con_valor_tras_interpolar"],
        },
        FIG_DIR / "etl_funnel.png",
        title="Filas a través del pipeline ETL (data lake -> warehouse)",
    )

    # 2. Missingness antes/después de interpolar, por indicador.
    before = pd.DataFrame(report["missingness_antes"]).set_index("columna")["pct_nulos"]
    after = pd.DataFrame(report["missingness_despues"]).set_index("columna")["pct_nulos"]
    viz.plot_missingness_before_after(
        before, after, FIG_DIR / "missingness_before_after.png",
        title="% de nulos en fact_indicator_value: antes vs. después de interpolar por serie país×indicador",
    )

    # 3. Esperanza de vida en el tiempo, 3 países reales representativos.
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    sample_df = con.execute("""
        SELECT dc.nombre_pais, f.anio, f.valor AS esperanza_vida
        FROM fact_indicator_value f
        JOIN dim_country dc ON dc.country_code = f.country_code
        WHERE f.indicator_code = 'SP.DYN.LE00.IN' AND f.country_code IN ('CHL', 'HTI', 'JPN')
        ORDER BY dc.nombre_pais, f.anio
    """).fetchdf()
    con.close()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pais, group in sample_df.groupby("nombre_pais"):
        ax.plot(group["anio"], group["esperanza_vida"], label=pais, linewidth=1.6)
    ax.set_xlabel("año")
    ax.set_ylabel("esperanza de vida al nacer (años)")
    ax.set_title("Esperanza de vida real, 1960-2024: Chile vs. Haití vs. Japón (WDI)")
    ax.legend()
    fig.tight_layout()
    (FIG_DIR / "esperanza_vida_paises.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "esperanza_vida_paises.png", dpi=150)
    plt.close(fig)

    # 4. Correlación entre features y target.
    viz.plot_correlation_heatmap(
        features_df, output_path=FIG_DIR / "feature_correlation.png",
        columns=FEATURE_COLUMNS + [TARGET_COLUMN],
        title="Correlación: indicadores socioeconómicos vs. esperanza de vida del año siguiente",
    )

    # 5. Curva de entrenamiento del MLP.
    viz.plot_training_curve(
        metrics["train_losses"], metrics["val_losses"], FIG_DIR / "mlp_training_curve.png",
        title="MLP -- pérdida de entrenamiento vs. validación por época", best_epoch=metrics["best_epoch"],
    )

    # 6. Real vs. predicho (test set, XGBoost -- el mejor modelo real).
    viz.plot_regression_diagnostics(
        metrics["y_test"], metrics["xgb_pred"], FIG_DIR / "xgb_regression_diagnostics.png",
        title="XGBoost -- esperanza de vida real vs. predicha (holdout 2019-2024)", unit="(años)",
    )

    # 7. Comparación baseline vs. MLP vs. XGBoost.
    viz.plot_model_comparison_bars(
        metrics["results"], FIG_DIR / "model_comparison.png", metrics=("r2", "rmse", "mae"),
        title="Predicción de esperanza de vida del año siguiente: baseline vs. MLP vs. XGBoost",
    )

    print(f"7 gráficos -> {FIG_DIR}")


if __name__ == "__main__":
    main()
