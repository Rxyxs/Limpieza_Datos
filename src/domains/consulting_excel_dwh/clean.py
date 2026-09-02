"""Transforma el extracto curado del WDI (ancho: una columna por año) en un
data warehouse real en DuckDB, esquema estrella: `dim_country`, `dim_indicator`,
`fact_indicator_value` (una fila por observación país×indicador×año). Este es
el trabajo típico de una consultora al recibir un Excel de un organismo
público: no se analiza el Excel directamente, se lo transforma en un modelo
dimensional consultable.

Filtrado clave: la hoja `Country` del WDI mezcla países reales con agregados
regionales/de ingreso ("World", "OECD members", "Africa Eastern and Southern")
-- se distinguen por el campo `Region`, vacío en los agregados y poblado en
todo país real. Sin este filtro, un promedio o modelo entrenado sobre el
panel mezclaría observaciones de países con sus propios agregados regionales,
inflando artificialmente la señal (un "país" como "World" está perfectamente
correlacionado con el promedio de los demás, por construcción).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.domains.consulting_excel_dwh.fetch import CURATED_INDICATORS
from src.toolkit.excel_cleaning import wide_years_to_long
from src.toolkit.missing_data import interpolate_within_group, missingness_report

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "consulting"
PROCESSED_DIR = ROOT / "data" / "processed" / "consulting"
WAREHOUSE_PATH = PROCESSED_DIR / "wdi_warehouse.duckdb"


def build_warehouse() -> dict:
    report: dict = {}

    wide = pd.read_csv(RAW_DIR / "wdi_curated_wide.csv")
    report["filas_crudas_ancho"] = len(wide)

    long_df = wide_years_to_long(
        wide, id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        year_pattern=r"^(19|20)\d{2}$", var_name="anio", value_name="valor",
    )
    long_df = long_df.rename(columns={"Country Code": "country_code", "Indicator Code": "indicator_code"})
    report["filas_largas_pre_filtro"] = len(long_df)

    # La hoja `Country` real trae 31 columnas (metadata SNA/BOP/censo que no
    # interesan acá) -- se usan solo las 4 relevantes por nombre real, no por
    # posición (una fila con una columna de menos correría todo el resto).
    country_dim = pd.read_csv(RAW_DIR / "wdi_country_dim.csv", usecols=["Country Code", "Short Name", "Region"])
    country_dim = country_dim.rename(columns={"Country Code": "country_code", "Short Name": "nombre_pais", "Region": "region"})
    real_countries = country_dim[country_dim["region"].notna()]
    report["paises_reales_en_dimension"] = len(real_countries)
    report["agregados_regionales_excluidos"] = len(country_dim) - len(real_countries)

    fact = long_df.merge(real_countries[["country_code", "region"]], on="country_code", how="inner")
    report["filas_tras_excluir_agregados"] = len(fact)
    report["filas_con_valor_real_antes_de_interpolar"] = int(fact["valor"].notna().sum())

    before_missing = missingness_report(fact)

    # Interpola gaps DENTRO de cada serie país×indicador (nunca a través de un
    # cambio de país o de indicador) -- ver docstring de `interpolate_within_group`.
    # Se interpola ANTES de descartar nulos (no después): el melt ancho->largo
    # ya produjo una fila NaN real por cada año sin publicación, y es
    # justamente esa fila la que hay que rellenar -- descartarla antes
    # convertiría este paso en un no-op.
    fact["serie_id"] = fact["country_code"] + "|" + fact["indicator_code"]
    fact = interpolate_within_group(fact, column="valor", group_column="serie_id", sort_by="anio")
    fact = fact.drop(columns=["serie_id"])

    after_missing = missingness_report(fact)
    report["missingness_antes"] = before_missing
    report["missingness_despues"] = after_missing

    # Lo que sigue NaN tras interpolar es una serie país×indicador sin NINGÚN
    # valor real en todo su historial -- ahí no hay nada de qué interpolar, y
    # es la única categoría de nulo que se descarta en vez de rellenarse.
    fact = fact.dropna(subset=["valor"]).reset_index(drop=True)
    report["filas_con_valor_tras_interpolar"] = len(fact)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(WAREHOUSE_PATH))

    dim_country = real_countries.drop_duplicates(subset=["country_code"])
    con.register("dim_country_df", dim_country)
    con.execute("CREATE TABLE dim_country AS SELECT * FROM dim_country_df")

    dim_indicator = pd.DataFrame(
        [{"indicator_code": code, "nombre_es": name} for code, name in CURATED_INDICATORS.items()]
    )
    con.register("dim_indicator_df", dim_indicator)
    con.execute("CREATE TABLE dim_indicator AS SELECT * FROM dim_indicator_df")

    fact_table = fact[["country_code", "indicator_code", "anio", "valor"]]
    con.register("fact_df", fact_table)
    con.execute("CREATE TABLE fact_indicator_value AS SELECT * FROM fact_df")
    con.close()
    report["n_paises_dim"] = len(dim_country)
    report["n_indicadores_dim"] = len(dim_indicator)
    report["n_filas_fact"] = len(fact_table)

    fact_table.to_csv(PROCESSED_DIR / "fact_indicator_value.csv", index=False)
    return report


def main() -> None:
    report = build_warehouse()
    print(f"Warehouse DuckDB -> {WAREHOUSE_PATH}")
    for key, value in report.items():
        if isinstance(value, pd.DataFrame):
            continue
        print(f"  {key}: {value}")

    # Se persiste el reporte (incluidas las tablas de missingness antes/después
    # como registros) para que `charts.py` no tenga que re-ejecutar todo el
    # pipeline de limpieza solo para graficarlo.
    import json
    REPORTS_DIR = ROOT / "outputs" / "consulting"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) for k, v in report.items()
    }
    with open(REPORTS_DIR / "clean_report.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)


if __name__ == "__main__":
    main()
