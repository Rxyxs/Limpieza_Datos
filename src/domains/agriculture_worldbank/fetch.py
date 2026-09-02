"""Descarga indicadores agrícolas reales de un panel de 9 países sudamericanos
vía la API pública del Banco Mundial (`api.worldbank.org`) -- sin autenticación,
sin dataset descargado a mano.

Un único llamado por indicador trae los 9 países x 36 años en una sola
respuesta JSON (el endpoint acepta múltiples códigos ISO3 separados por `;`),
así que son 8 llamados en total, uno por indicador. La API devuelve un array
de 2 elementos: `[0]` es metadata de paginación, `[1]` es la lista de
registros. Un valor faltante llega como JSON `null` (no como registro
ausente) -- eso es missingness real, más denso en décadas tempranas y en
países más chicos/pobres, reflejo genuino de la cobertura estadística
internacional real, no algo que se fabrique acá.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "agriculture"
BASE_URL = "https://api.worldbank.org/v2/country"

COUNTRIES = ["CHL", "PER", "ARG", "BRA", "COL", "BOL", "URY", "PRY", "ECU"]
DATE_RANGE = "1990:2025"

INDICATORS = {
    "AG.YLD.CREL.KG": "cereal_yield_kg_ha",
    "AG.LND.ARBL.ZS": "arable_land_pct",
    "AG.CON.FERT.ZS": "fertilizer_kg_ha",
    "AG.LND.AGRI.ZS": "agri_land_pct",
    "AG.PRD.CROP.XD": "crop_production_index",
    "SP.RUR.TOTL.ZS": "rural_pop_pct",
    "AG.LND.IRIG.AG.ZS": "irrigated_land_pct",
    "NV.AGR.TOTL.ZS": "agri_value_added_gdp_pct",
}


def fetch_indicator(indicator_code: str) -> pd.DataFrame:
    """Descarga el panel país x año completo (1990-2025) para `indicator_code`."""
    countries = ";".join(COUNTRIES)
    url = f"{BASE_URL}/{countries}/indicator/{indicator_code}"
    resp = requests.get(url, params={"format": "json", "per_page": 1000, "date": DATE_RANGE}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError(f"'{indicator_code}': respuesta vacía o inesperada de la API")

    records = payload[1]
    df = pd.DataFrame([
        {
            "country_iso3": r["countryiso3code"],
            "country_name": r["country"]["value"],
            "year": int(r["date"]),
            "indicator_code": indicator_code,
            "value": r["value"],
        }
        for r in records
    ])
    return df.sort_values(["country_iso3", "year"]).reset_index(drop=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for code, slug in INDICATORS.items():
        try:
            df = fetch_indicator(code)
        except (requests.RequestException, ValueError) as exc:
            print(f"[warn] {code}: {exc}", file=sys.stderr)
            continue

        out_path = RAW_DIR / f"{slug}.csv"
        df.to_csv(out_path, index=False)

        n_countries = df["country_iso3"].nunique()
        n_non_null = df["value"].notna().sum()
        years_with_data = df.loc[df["value"].notna(), "year"]
        year_span = (
            f"{years_with_data.min()}-{years_with_data.max()}" if not years_with_data.empty else "sin datos"
        )
        print(
            f"{code} ({slug}): {len(df):,} filas ({n_countries} países x hasta 36 años), "
            f"{n_non_null:,} valores reales no-nulos, cobertura real {year_span} -> {out_path.name}"
        )
        time.sleep(0.3)  # cortesía con la API pública del Banco Mundial


if __name__ == "__main__":
    main()
