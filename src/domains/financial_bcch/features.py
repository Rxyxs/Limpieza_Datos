"""Feature engineering sobre el panel financiero limpio.

Target: `target_next_return` = retorno logarítmico del dólar observado al DÍA
SIGUIENTE (`dolar_log_return.shift(-1)`). Todas las features usan solo
información disponible hasta el día `t` (lags y ventanas móviles *shifted*),
sin look-ahead -- el punto central de la Hipótesis de Mercados Eficientes es
que esto es genuinamente difícil de predecir, así que un R² modesto acá es un
resultado honesto, no una falla del pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "financial"

LAGS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20]

FEATURE_COLUMNS = (
    [f"dolar_log_return_lag{lag}" for lag in LAGS]
    + [f"dolar_volatility_{w}d" for w in ROLLING_WINDOWS]
    + ["tpm", "ipc", "imacec", "dow", "month", "uf_log_return"]
)
TARGET_COLUMN = "target_next_return"


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values("fecha").reset_index(drop=True).copy()

    for lag in LAGS:
        df[f"dolar_log_return_lag{lag}"] = df["dolar_log_return"].shift(lag)

    for window in ROLLING_WINDOWS:
        # Volatilidad realizada: desviación estándar móvil de los retornos, la
        # feature más informativa en finanzas para predecir la MAGNITUD de un
        # movimiento futuro (aunque no necesariamente su signo).
        df[f"dolar_volatility_{window}d"] = df["dolar_log_return"].shift(1).rolling(window).std()

    df["uf_log_return"] = df["uf"].pct_change()
    df["dow"] = df["fecha"].dt.dayofweek
    df["month"] = df["fecha"].dt.month

    df[TARGET_COLUMN] = df["dolar_log_return"].shift(-1)

    model_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return model_df


def main() -> None:
    panel = pd.read_csv(PROCESSED_DIR / "financial_panel_clean.csv", parse_dates=["fecha"])
    features_df = build_features(panel)
    out_path = PROCESSED_DIR / "financial_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Features: {len(features_df):,} filas x {len(FEATURE_COLUMNS)} features -> {out_path}")


if __name__ == "__main__":
    main()
