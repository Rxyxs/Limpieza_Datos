"""Pipeline de limpieza de tickets IT/SaaS.

Orquesta: crudo -> descarte de filas vacías -> JSON aplanado -> fechas normalizadas a
UTC -> limpieza de strings/monedas/nombres -> deteccion de casi-duplicados ->
features derivadas -> imputación -> tratamiento de outliers -> formato ISO 8601 ->
validación de esquema -> exportación a `data/processed/`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaners.datetime_cleaner import clean_datetime_column, to_iso8601
from src.cleaners.duplicate_detector import find_near_duplicate_tickets
from src.cleaners.json_normalizer import normalize_json_column
from src.cleaners.missing_data_imputer import impute_numeric_by_category, interpolate_numeric_column
from src.cleaners.outlier_handler import winsorize_column, winsorize_column_by_group
from src.cleaners.string_cleaner import normalize_case, parse_currency, strip_whitespace, unify_similar_names
from src.validators.schema_validator import validate_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "messy_it_tickets.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_OUTPUT_PATH = PROCESSED_DIR / "clean_it_tickets.csv"
INVALID_OUTPUT_PATH = PROCESSED_DIR / "invalid_it_tickets.csv"
NEAR_DUPLICATES_OUTPUT_PATH = PROCESSED_DIR / "near_duplicate_candidates.csv"

STRING_COLUMNS = ["company_name", "agent_name", "customer_email", "priority", "status", "category"]
CATEGORICAL_CASE_COLUMNS = ["priority", "status", "category"]


def run_pipeline(raw_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Ejecuta el pipeline completo.

    Devuelve `(filas_validas, filas_invalidas, casi_duplicados_candidatos,
    reporte_de_outliers)`.
    """
    df = pd.read_csv(raw_path)

    # Filas completamente vacías (exports corruptos): no aportan señal, se descartan
    # antes de invertir trabajo de parseo en ellas.
    df = df.dropna(how="all").reset_index(drop=True)

    df = normalize_json_column(df, column="user_metadata")

    df = clean_datetime_column(df, "created_at")
    df = clean_datetime_column(df, "resolved_at")

    df = strip_whitespace(df, STRING_COLUMNS)
    df = normalize_case(df, CATEGORICAL_CASE_COLUMNS)
    df["company_name"] = unify_similar_names(df["company_name"])
    df["cost"] = df["cost"].apply(parse_currency)

    # Casi-duplicados (mismo cliente+categoria, creados casi al mismo tiempo, nombre
    # de empresa casi identico): se detectan ANTES de imputar/validar, sobre las
    # columnas ya limpias pero crudas todavia, y se reportan aparte -- no se
    # descartan filas automaticamente (ver docstring del modulo).
    near_duplicates = find_near_duplicate_tickets(df)

    df["response_time_hours"] = (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600

    df = impute_numeric_by_category(df, value_column="cost", category_column="category")
    df = interpolate_numeric_column(df, column="response_time_hours", sort_by="created_at")

    # `cost` no depende de la categoria (ver dirty_data_generator.py), IQR global
    # esta bien. `response_time_hours` SI depende de category+priority a proposito,
    # asi que se winsoriza POR categoria -- un IQR global trataria esa variacion
    # estructural como ruido y sobre-recortaria las categorias de escala mas alta.
    df, cost_outliers = winsorize_column(df, "cost")
    df, response_time_outliers = winsorize_column_by_group(df, "response_time_hours", group_column="category")
    outlier_report = {"cost": cost_outliers, "response_time_hours": response_time_outliers}

    df = to_iso8601(df, "created_at")
    df = to_iso8601(df, "resolved_at")

    valid_df, invalid_df = validate_dataframe(df)
    return valid_df, invalid_df, near_duplicates, outlier_report


if __name__ == "__main__":
    valid_df, invalid_df, near_duplicates, outlier_report = run_pipeline()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(CLEAN_OUTPUT_PATH, index=False)
    if not invalid_df.empty:
        invalid_df.to_csv(INVALID_OUTPUT_PATH, index=False)
    if not near_duplicates.empty:
        near_duplicates.to_csv(NEAR_DUPLICATES_OUTPUT_PATH, index=False)

    total = len(valid_df) + len(invalid_df)
    print(f"Filas procesadas: {total:,}")
    print(f"  Válidas   -> {CLEAN_OUTPUT_PATH} ({len(valid_df):,} filas)")
    print(f"  Inválidas -> {INVALID_OUTPUT_PATH} ({len(invalid_df):,} filas)")
    print(f"  Casi-duplicados candidatos -> {NEAR_DUPLICATES_OUTPUT_PATH} ({len(near_duplicates):,} pares)")
    print(f"  Outliers winsorizados: {outlier_report}")

    if not invalid_df.empty:
        print("\nMotivos de invalidez más comunes:")
        first_lines = invalid_df["validation_error"].str.split("\n").str[0]
        print(first_lines.value_counts().head(10).to_string())
