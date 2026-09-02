"""Descarga indicadores económicos reales de Chile vía la API pública de
mindicador.cl (que replica series del Banco Central de Chile) -- sin
autenticación, sin dataset descargado a mano.

El endpoint sin año (`/api/{codigo}`) solo devuelve los últimos ~30 registros,
así que se itera año por año para reconstruir el histórico completo. Esto
también es lo que genera la mezcla de frecuencias real que hace interesante
este dominio para limpieza: dólar/UF publican todos los días hábiles, IPC/
IMACEC/TPM publican una vez al mes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "financial"
BASE_URL = "https://mindicador.cl/api"

DAILY_INDICATORS = ["dolar", "uf"]
MONTHLY_INDICATORS = ["ipc", "imacec", "tpm"]
YEARS = range(2013, 2027)


def _fetch_year(codigo: str, year: int) -> list[dict]:
    resp = requests.get(f"{BASE_URL}/{codigo}/{year}", timeout=30)
    resp.raise_for_status()
    return resp.json().get("serie", [])


def fetch_indicator_history(codigo: str) -> pd.DataFrame:
    """Reconstruye el histórico completo de `codigo` iterando año por año."""
    rows: list[dict] = []
    for year in YEARS:
        try:
            serie = _fetch_year(codigo, year)
        except requests.RequestException as exc:
            print(f"[warn] {codigo}/{year}: {exc}", file=sys.stderr)
            continue
        rows.extend(serie)
        time.sleep(0.15)  # cortesía con la API pública, no tiene rate limit documentado

    if not rows:
        raise ValueError(f"'{codigo}': no se obtuvo ningún dato")

    df = pd.DataFrame(rows).drop_duplicates(subset="fecha")
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df["indicador"] = codigo
    return df.sort_values("fecha").reset_index(drop=True)[["fecha", "indicador", "valor"]]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for codigo in DAILY_INDICATORS + MONTHLY_INDICATORS:
        df = fetch_indicator_history(codigo)
        out_path = RAW_DIR / f"{codigo}.csv"
        df.to_csv(out_path, index=False)
        print(f"{codigo}: {len(df):,} filas ({df['fecha'].min().date()} a {df['fecha'].max().date()}) -> {out_path.name}")


if __name__ == "__main__":
    main()
