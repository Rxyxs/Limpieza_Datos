"""Genera los gráficos del dominio financiero, todos vía la librería reusable
`src.toolkit.viz` -- ninguna función de gráfico vive acá, solo se preparan los
datos de cada dominio y se llama a la caja negra común.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.toolkit import viz

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "financial"
PROCESSED_DIR = ROOT / "data" / "processed" / "financial"
REPORTS_DIR = ROOT / "outputs" / "financial"
FIG_DIR = REPORTS_DIR / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED_DIR / "financial_panel_clean.csv", parse_dates=["fecha"])
    features_df = pd.read_csv(PROCESSED_DIR / "financial_features.csv", parse_dates=["fecha"])
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))

    # 1. Missingness antes/después: % de días de calendario sin publicación
    # antes de reindexar+forward-fill, para dólar/UF/TPM.
    pct_before, pct_after = {}, {}
    for codigo in ["dolar", "uf", "tpm"]:
        raw = pd.read_csv(RAW_DIR / f"{codigo}.csv", parse_dates=["fecha"])
        raw["fecha"] = raw["fecha"].dt.tz_localize(None).dt.normalize()
        full_range = pd.date_range(raw["fecha"].min(), raw["fecha"].max(), freq="D")
        pct_before[codigo] = 100 * (1 - len(raw["fecha"].unique()) / len(full_range))
        pct_after[codigo] = 0.0
    viz.plot_missingness_before_after(
        pd.Series(pct_before), pd.Series(pct_after), FIG_DIR / "missingness_before_after.png",
        title="% de días de calendario sin publicación: antes vs. después de reindexar",
    )

    # 2. Distribución del retorno del dólar: crudo vs. winsorizado (recorta colas).
    dolar = pd.read_csv(RAW_DIR / "dolar.csv", parse_dates=["fecha"])
    dolar["fecha"] = dolar["fecha"].dt.tz_localize(None).dt.normalize()
    dolar_full = dolar.set_index("fecha")["valor"].reindex(
        pd.date_range(dolar["fecha"].min(), dolar["fecha"].max(), freq="D")
    ).ffill()
    raw_return = np.log(dolar_full / dolar_full.shift(1)).dropna()
    viz.plot_distribution_before_after(
        raw_return, panel["dolar_log_return"], FIG_DIR / "return_distribution_before_after.png",
        title="Retorno log. diario del dólar: crudo vs. winsorizado (IQR k=4)", xlabel="retorno log. diario",
    )

    # 3. Serie de tiempo del dólar observado, historial completo, con media móvil de 20 días.
    viz.plot_timeseries(
        panel, x="fecha", y="dolar", output_path=FIG_DIR / "dolar_timeseries.png",
        title="USD/CLP observado, 2013-2026 (panel limpio)", rolling_window=20, ylabel="CLP por USD",
    )

    # 4. Correlación entre features del modelo.
    from src.domains.financial_bcch.features import FEATURE_COLUMNS, TARGET_COLUMN
    viz.plot_correlation_heatmap(
        features_df, output_path=FIG_DIR / "feature_correlation.png",
        columns=FEATURE_COLUMNS + [TARGET_COLUMN],
        title="Correlación: features del panel financiero vs. retorno del día siguiente",
    )

    # 5. Curva de entrenamiento del MLP (>=100 épocas, mejor checkpoint marcado).
    viz.plot_training_curve(
        metrics["train_losses"], metrics["val_losses"], FIG_DIR / "mlp_training_curve.png",
        title="MLP -- pérdida de entrenamiento vs. validación por época", best_epoch=metrics["best_epoch"],
    )

    # 6. Real vs. predicho (test set, MLP) -- retorno del día siguiente.
    viz.plot_regression_diagnostics(
        metrics["y_test"], metrics["mlp_pred"], FIG_DIR / "mlp_regression_diagnostics.png",
        title="MLP -- retorno real vs. predicho (holdout cronológico)",
    )

    # 7. Comparación baseline vs. MLP vs. XGBoost.
    viz.plot_model_comparison_bars(
        metrics["results"], FIG_DIR / "model_comparison.png", metrics=("r2", "rmse", "mae"),
        title="Predicción del retorno del dólar al día siguiente: baseline vs. MLP vs. XGBoost",
    )

    print(f"7 gráficos -> {FIG_DIR}")


if __name__ == "__main__":
    main()
