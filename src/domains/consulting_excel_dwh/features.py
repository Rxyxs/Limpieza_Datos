"""Consulta el warehouse DuckDB (esquema estrella) y lo aplana a la tabla ancha
que un modelo necesita -- el patrón real de cómo un pipeline de ML consume un
data warehouse: el fact table queda normalizado/largo (una fila por
observación), y cada consumidor específico (acá, un modelo de esperanza de
vida) pivotea solo lo que necesita vía SQL, no una tabla ancha pre-armada
para todos los casos de uso.

Target: `esperanza_vida` (SP.DYN.LE00.IN) del AÑO SIGUIENTE, a partir de los
indicadores socioeconómicos del año actual -- sin look-ahead.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.domains.consulting_excel_dwh.fetch import CURATED_INDICATORS

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "consulting"
WAREHOUSE_PATH = PROCESSED_DIR / "wdi_warehouse.duckdb"

INDICATOR_COLUMNS = list(CURATED_INDICATORS.values())  # nombres amigables en español
TARGET_COLUMN = "target_esperanza_vida_siguiente"
FEATURE_COLUMNS = INDICATOR_COLUMNS + [f"{c}_lag1" for c in INDICATOR_COLUMNS]


def query_wide_panel() -> pd.DataFrame:
    """Pivotea `fact_indicator_value` (largo) a un panel ancho país×año, vía SQL
    directamente sobre el warehouse -- exactamente la consulta que un analista
    de una consultora escribiría contra el modelo dimensional ya construido.
    """
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)

    pivot_cases = ",\n".join(
        f"MAX(CASE WHEN indicator_code = '{code}' THEN valor END) AS {name}"
        for code, name in CURATED_INDICATORS.items()
    )
    query = f"""
        SELECT f.country_code, dc.nombre_pais, dc.region, f.anio,
               {pivot_cases}
        FROM fact_indicator_value f
        JOIN dim_country dc ON dc.country_code = f.country_code
        GROUP BY f.country_code, dc.nombre_pais, dc.region, f.anio
        ORDER BY f.country_code, f.anio
    """
    wide = con.execute(query).fetchdf()
    con.close()
    return wide


def build_features(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.sort_values(["country_code", "anio"]).reset_index(drop=True).copy()

    for col in INDICATOR_COLUMNS:
        df[f"{col}_lag1"] = df.groupby("country_code")[col].shift(1)

    df[TARGET_COLUMN] = df.groupby("country_code")["esperanza_vida"].shift(-1)

    model_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return model_df


def main() -> None:
    wide = query_wide_panel()
    features_df = build_features(wide)
    out_path = PROCESSED_DIR / "consulting_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Panel ancho (warehouse): {len(wide):,} filas país-año")
    print(f"Features (con target, sin nulos): {len(features_df):,} filas x {len(FEATURE_COLUMNS)} features -> {out_path}")
    print(f"Países cubiertos: {features_df['country_code'].nunique()}")


if __name__ == "__main__":
    main()
