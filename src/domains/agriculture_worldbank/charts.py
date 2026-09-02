"""Genera los gráficos del dominio agrícola, todos vía la librería reusable
`src.toolkit.viz` -- ninguna función de gráfico vive acá, solo se preparan los
datos de cada dominio y se llama a la caja negra común.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.domains.agriculture_worldbank.clean import _load_wide_panel
from src.domains.agriculture_worldbank.features import COUNTRY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.toolkit import viz
from src.toolkit.missing_data import missingness_report

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "agriculture"
REPORTS_DIR = ROOT / "outputs" / "agriculture"
FIG_DIR = REPORTS_DIR / "figures"

REPRESENTATIVE_COUNTRIES = ["CHL", "ARG"]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED_DIR / "agriculture_panel_clean.csv")
    features_df = pd.read_csv(PROCESSED_DIR / "agriculture_features.csv")
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))

    # 1. Missingness antes/después: panel crudo (8 indicadores del Banco
    # Mundial recién unidos, sin tratar) vs. panel final ya interpolado por
    # país + fallback de media entre países.
    raw_panel = _load_wide_panel()
    pct_before = missingness_report(raw_panel).set_index("columna")["pct_nulos"]
    pct_after = missingness_report(panel).set_index("columna")["pct_nulos"]
    viz.plot_missingness_before_after(
        pct_before, pct_after, FIG_DIR / "missingness_before_after.png",
        title="% de país-años sin dato real: antes vs. después de interpolar + fallback por país",
    )

    # 2. Correlación entre features numéricas (sin dummies de país) y el
    # rendimiento de cereales objetivo.
    numeric_feature_columns = [c for c in FEATURE_COLUMNS if c not in COUNTRY_COLUMNS]
    viz.plot_correlation_heatmap(
        features_df, output_path=FIG_DIR / "feature_correlation.png",
        columns=numeric_feature_columns + [TARGET_COLUMN],
        title="Correlación: indicadores agrícolas + lags vs. rendimiento de cereales",
    )

    # 3. Serie de tiempo del rendimiento de cereales, 1-2 países representativos
    # (Chile: ancla temática del portafolio; Argentina: mayor productor
    # regional de cereales del panel).
    for country in REPRESENTATIVE_COUNTRIES:
        country_df = panel[panel["country_iso3"] == country].sort_values("year").reset_index(drop=True)
        country_name = country_df["country_name"].iloc[0]
        viz.plot_timeseries(
            country_df, x="year", y="cereal_yield_kg_ha",
            output_path=FIG_DIR / f"cereal_yield_timeseries_{country}.png",
            title=f"Rendimiento de cereales observado, {country_name} 1990-2025 (panel limpio)",
            rolling_window=3, ylabel="kg por hectárea",
        )

    # 4. Curva de entrenamiento del MLP (>=100 épocas, mejor checkpoint marcado).
    viz.plot_training_curve(
        metrics["train_losses"], metrics["val_losses"], FIG_DIR / "mlp_training_curve.png",
        title="MLP -- pérdida de entrenamiento vs. validación por época", best_epoch=metrics["best_epoch"],
    )

    # 5. Real vs. predicho (test set, mejor de los 2 modelos entrenados por R²).
    r2_by_model = {"mlp_pytorch": metrics["results"]["mlp_pytorch"]["r2"], "xgboost": metrics["results"]["xgboost"]["r2"]}
    best_model = max(r2_by_model, key=r2_by_model.get)
    best_pred = metrics["mlp_pred"] if best_model == "mlp_pytorch" else metrics["xgb_pred"]
    viz.plot_regression_diagnostics(
        metrics["y_test"], best_pred, FIG_DIR / "regression_diagnostics.png",
        title=f"{best_model} -- rendimiento real vs. predicho (holdout cronológico 2020-2025)",
        unit="(kg/ha)",
    )

    # 6. Comparación baseline vs. MLP vs. XGBoost.
    viz.plot_model_comparison_bars(
        metrics["results"], FIG_DIR / "model_comparison.png", metrics=("r2", "rmse", "mae"),
        title="Predicción del rendimiento de cereales: baseline por país vs. MLP vs. XGBoost",
    )

    n_figures = 5 + len(REPRESENTATIVE_COUNTRIES)  # missingness, correlación, entrenamiento, diagnóstico, comparación + 1 por país
    print(f"{n_figures} gráficos -> {FIG_DIR}")


if __name__ == "__main__":
    main()
