"""Genera los gráficos del dominio minero, todos vía la librería reusable
`src.toolkit.viz` -- ninguna función de gráfico vive acá, solo se preparan los
datos de cada dominio y se llama a la caja negra común.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.domains.mining_cochilco.clean import GRAND_TOTAL_COLUMN, SHEET_NAME, SUBTOTAL_COLUMNS
from src.domains.mining_cochilco.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.toolkit import viz
from src.toolkit.excel_cleaning import detect_header_row
from src.toolkit.missing_data import missingness_report

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "mining"
PROCESSED_DIR = ROOT / "data" / "processed" / "mining"
REPORTS_DIR = ROOT / "outputs" / "mining"
FIG_DIR = REPORTS_DIR / "figures"
RAW_FILENAME = "cochilco_produccion_mensual.xlsx"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED_DIR / "mining_panel_clean.csv", parse_dates=["fecha"])
    features_df = pd.read_csv(PROCESSED_DIR / "mining_features.csv", parse_dates=["fecha"])
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))

    company_cols = [c for c in panel.columns if c not in SUBTOTAL_COLUMNS + [GRAND_TOTAL_COLUMN, "fecha",
                    "total_nacional_miles_ton", "total_codelco_miles_ton", "share_codelco"]]

    # 1. Missingness antes/después: el Excel crudo de COCHILCO trae filas de
    # resumen anual y una fila de cita al pie mezcladas con las filas
    # mensuales -- ANTES de filtrarlas por tipo de la columna fecha, la fila
    # de cita deja NaN en todas las columnas de empresa (~4.7% de las 171
    # filas del bloque de datos); DESPUÉS del filtro de fila y sin celdas en
    # blanco dentro de las filas mensuales reales, queda en 0%.
    raw = pd.read_excel(RAW_DIR / RAW_FILENAME, sheet_name=SHEET_NAME, header=None)
    header_row_idx = detect_header_row(raw, expected_tokens=["mes-a", "chuqui", "escondida", "collahuasi", "total chile"])
    headers = [str(v).strip() for v in raw.iloc[header_row_idx].tolist()]
    headers[0] = "fecha"
    raw_body = raw.iloc[header_row_idx + 1:].reset_index(drop=True).copy()
    raw_body.columns = headers
    raw_body = raw_body.loc[:, [c for c in raw_body.columns if c and c != "None"]]
    before_report = missingness_report(raw_body).set_index("columna")["pct_nulos"]

    after_report = pd.Series({col: 0.0 for col in raw_body.columns if col != "fecha"})
    viz.plot_missingness_before_after(
        before_report, after_report, FIG_DIR / "missingness_before_after.png",
        title="% de nulos por columna: bloque crudo (con filas anuales/cita) vs. panel limpio",
    )

    # 2. Distribución de producción empresa-mes: con columnas subtotal
    # disfrazadas incluidas ("Chuqui y R.Tomic", "Total Codelco",
    # "Angloamerican Sur", "Capstone Copper" -- cada una es la suma de otras
    # columnas ya presentes, ver `clean.py`) vs. solo las 38 columnas reales.
    # Incluir los subtotales infla la cola alta de la distribución con
    # valores que ya están contados en otra parte.
    before_values = panel.melt(value_vars=company_cols + SUBTOTAL_COLUMNS, value_name="v")["v"]
    after_values = panel.melt(value_vars=company_cols, value_name="v")["v"]
    viz.plot_distribution_before_after(
        before_values, after_values, FIG_DIR / "production_distribution_before_after.png",
        title="Producción empresa-mes: con columnas subtotal disfrazadas vs. solo empresas reales",
        xlabel="miles de T.M. de cobre fino / mes",
    )

    # 3. Serie de tiempo del total nacional, historial completo, con media
    # móvil de 12 meses (estacionalidad de calendario anual).
    viz.plot_timeseries(
        panel, x="fecha", y="total_nacional_miles_ton", output_path=FIG_DIR / "produccion_nacional_timeseries.png",
        title="Producción nacional de cobre de mina, 2014-2026 (panel limpio)", rolling_window=12,
        ylabel="miles de T.M. de cobre fino",
    )

    # 4. Correlación entre features del modelo.
    viz.plot_correlation_heatmap(
        features_df, output_path=FIG_DIR / "feature_correlation.png",
        columns=FEATURE_COLUMNS + [TARGET_COLUMN],
        title="Correlación: features del panel minero vs. producción nacional del mes siguiente",
    )

    # 5. Curva de entrenamiento del MLP (>=100 épocas, mejor checkpoint marcado).
    viz.plot_training_curve(
        metrics["train_losses"], metrics["val_losses"], FIG_DIR / "mlp_training_curve.png",
        title="MLP -- pérdida de entrenamiento vs. validación por época (target z-scoreado)",
        best_epoch=metrics["best_epoch"],
    )

    # 6. Real vs. predicho (test set, XGBoost -- el mejor modelo real, ver metrics.json).
    viz.plot_regression_diagnostics(
        metrics["y_test"], metrics["xgb_pred"], FIG_DIR / "xgboost_regression_diagnostics.png",
        title="XGBoost -- producción nacional real vs. predicha (holdout cronológico)",
        unit="(miles de T.M.)",
    )

    # 7. Comparación baseline estacional vs. MLP vs. XGBoost.
    viz.plot_model_comparison_bars(
        metrics["results"], FIG_DIR / "model_comparison.png", metrics=("r2", "rmse", "mae"),
        title="Predicción de producción nacional del mes siguiente: baseline vs. MLP vs. XGBoost",
    )

    print(f"7 gráficos -> {FIG_DIR}")


if __name__ == "__main__":
    main()
