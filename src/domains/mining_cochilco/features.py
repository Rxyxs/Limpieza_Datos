"""Feature engineering sobre el panel minero limpio.

Target: `target_next_total` = producción nacional total (miles de T.M. de
cobre fino) del MES SIGUIENTE (`total_nacional_miles_ton.shift(-1)`). Todas
las features usan solo información disponible hasta el mes `t` (lags y
ventanas móviles *shifted*), sin look-ahead.

La producción minera de cobre tiene autocorrelación operacional fuerte (el
plan minero de un mes se parece al del mes anterior) y estacionalidad de
calendario real: se confirmó empíricamente sobre el panel limpio que febrero
es sistemáticamente el mes más bajo (menos días + mantenciones programadas de
verano) y diciembre el más alto (cierre de año) -- por eso se incluye tanto el
lag12 (mismo mes, año anterior) como el mes calendario como features, y por lo
mismo el baseline de `model.py` es estacional-naive (mes t-11 relativo a la
fila actual = mismo mes, año anterior, relativo al mes objetivo) y no la media
plana: en una serie con estacionalidad real, "el mismo mes del año pasado" es
un piso mucho más difícil de superar que "el promedio histórico".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "mining"

TOTAL_COLUMN = "total_nacional_miles_ton"
LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]

FEATURE_COLUMNS = (
    [f"total_lag{lag}" for lag in LAGS]
    + [f"total_rolling_mean_{w}m" for w in ROLLING_WINDOWS]
    + ["mes", "anio", "yoy_growth", "share_codelco"]
)
TARGET_COLUMN = "target_next_total"
SEASONAL_NAIVE_COLUMN = "seasonal_naive_pred"  # no es una feature de entrada -- baseline para model.py


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values("fecha").reset_index(drop=True).copy()

    for lag in LAGS:
        df[f"total_lag{lag}"] = df[TOTAL_COLUMN].shift(lag)

    for window in ROLLING_WINDOWS:
        # Media móvil de meses YA observados (shift(1) antes de la ventana):
        # la media móvil "centrada" en t usaría producción de meses futuros.
        df[f"total_rolling_mean_{window}m"] = df[TOTAL_COLUMN].shift(1).rolling(window).mean()

    df["mes"] = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year
    df["yoy_growth"] = df[TOTAL_COLUMN].pct_change(12)

    # Baseline estacional-naive: producción del mismo mes, año anterior,
    # relativa al mes OBJETIVO (t+1) -- 12 meses antes de t+1 es 11 meses
    # antes de t.
    df[SEASONAL_NAIVE_COLUMN] = df[TOTAL_COLUMN].shift(11)

    df[TARGET_COLUMN] = df[TOTAL_COLUMN].shift(-1)

    required = FEATURE_COLUMNS + [TARGET_COLUMN, SEASONAL_NAIVE_COLUMN]
    model_df = df.dropna(subset=required).reset_index(drop=True)
    return model_df


def main() -> None:
    panel = pd.read_csv(PROCESSED_DIR / "mining_panel_clean.csv", parse_dates=["fecha"])
    features_df = build_features(panel)
    out_path = PROCESSED_DIR / "mining_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Features: {len(features_df):,} filas x {len(FEATURE_COLUMNS)} features -> {out_path}")


if __name__ == "__main__":
    main()
