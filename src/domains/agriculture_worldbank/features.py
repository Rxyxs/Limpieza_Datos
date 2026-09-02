"""Feature engineering sobre el panel agrícola limpio.

Target: `cereal_yield_kg_ha` (rendimiento de cereales, kg/hectárea) del propio
año-país -- se explica con los OTROS indicadores contemporáneos del mismo año
(fertilizante, tierra arable, irrigación, etc., que son insumos reales de
producción de ese mismo ciclo agrícola, no información del futuro) más
lags/ventanas móviles del PROPIO rendimiento y fertilizante pasados. La
disciplina "no look-ahead" acá es sobre los lags: `shift(1)`/`shift(2)` nunca
`shift(-1)`, y todo lag/rolling se calcula agrupado por país
(`groupby("country_iso3")`) para no mezclar el año 2015 de Bolivia con el
2016 de Argentina -- el mismo cuidado que `interpolate_within_group` aplica
en `clean.py`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domains.agriculture_worldbank.fetch import COUNTRIES
from src.toolkit.encoding import encode_categorical_onehot

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "agriculture"

TARGET_COLUMN = "cereal_yield_kg_ha"

# Indicadores contemporáneos (mismo año) usados tal cual, sin lag: insumos
# reales de producción del propio ciclo agrícola.
CONTEMPORANEOUS_COLUMNS = [
    "arable_land_pct",
    "fertilizer_kg_ha",
    "agri_land_pct",
    "crop_production_index",
    "rural_pop_pct",
    "irrigated_land_pct",
    "agri_value_added_gdp_pct",
]

YIELD_LAGS = [1, 2]
FERTILIZER_LAGS = [1]
ROLLING_WINDOWS = [3]

COUNTRY_COLUMNS = [f"country_iso3_{c}" for c in COUNTRIES]

FEATURE_COLUMNS = (
    CONTEMPORANEOUS_COLUMNS
    + [f"cereal_yield_lag{lag}" for lag in YIELD_LAGS]
    + [f"fertilizer_lag{lag}" for lag in FERTILIZER_LAGS]
    + [f"cereal_yield_rolling{w}" for w in ROLLING_WINDOWS]
    + ["year"]
    + COUNTRY_COLUMNS
)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["country_iso3", "year"]).reset_index(drop=True).copy()

    for lag in YIELD_LAGS:
        df[f"cereal_yield_lag{lag}"] = df.groupby("country_iso3")[TARGET_COLUMN].shift(lag)

    for lag in FERTILIZER_LAGS:
        df[f"fertilizer_lag{lag}"] = df.groupby("country_iso3")["fertilizer_kg_ha"].shift(lag)

    for window in ROLLING_WINDOWS:
        # Media móvil del rendimiento pasado (shift(1) antes de rolling): usa
        # solo años ANTERIORES al actual, nunca el año que se está prediciendo.
        df[f"cereal_yield_rolling{window}"] = df.groupby("country_iso3")[TARGET_COLUMN].transform(
            lambda s: s.shift(1).rolling(window).mean()
        )

    df = encode_categorical_onehot(df, columns=["country_iso3"])
    for col in COUNTRY_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # país sin filas en este panel (no debería pasar con los 9 países reales, pero por robustez)

    model_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return model_df


def main() -> None:
    panel = pd.read_csv(PROCESSED_DIR / "agriculture_panel_clean.csv")
    features_df = build_features(panel)
    out_path = PROCESSED_DIR / "agriculture_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Features: {len(features_df):,} filas x {len(FEATURE_COLUMNS)} features -> {out_path}")
    print(f"  (de {len(panel):,} filas del panel limpio; se pierden las primeras filas de cada país por lags/rolling)")


if __name__ == "__main__":
    main()
