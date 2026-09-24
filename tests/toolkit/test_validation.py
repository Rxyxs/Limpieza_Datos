"""Pruebas unitarias para src/toolkit/validation.py."""
import pandas as pd
from pydantic import BaseModel, Field


class _Schema(BaseModel):
    pais: str = Field(min_length=2)
    anio: int
    valor: float


def test_validate_dataframe_splits_valid_and_invalid_rows():
    from src.toolkit.validation import validate_dataframe

    df = pd.DataFrame({
        "pais": ["Chile", "P"],  # "P" viola min_length=2
        "anio": [2020, 2021],
        "valor": [1.5, 2.5],
    })
    valid_df, invalid_df = validate_dataframe(df, _Schema)

    assert len(valid_df) == 1
    assert len(invalid_df) == 1
    assert "validation_error" in invalid_df.columns
    assert valid_df.iloc[0]["pais"] == "Chile"


def test_infer_schema_captures_dtype_kind_per_column():
    from src.toolkit.validation import infer_schema

    df = pd.DataFrame({"anio": [2020, 2021], "valor": [1.5, 2.5], "pais": ["Chile", "Perú"]})
    schema = infer_schema(df)

    assert schema == {"anio": "i", "valor": "f", "pais": "O"}


def test_validate_schema_passes_when_columns_and_dtypes_match():
    from src.toolkit.validation import infer_schema, validate_schema

    train_df = pd.DataFrame({"anio": [2020, 2021], "valor": [1.5, 2.5]})
    inference_df = pd.DataFrame({"anio": [2022, 2023], "valor": [3.5, 4.5]})

    resultado = validate_schema(inference_df, infer_schema(train_df))

    assert resultado["esquema_valido"] is True
    assert resultado["columnas_faltantes"] == []
    assert resultado["columnas_inesperadas"] == []
    assert resultado["cambios_de_tipo"] == []


def test_validate_schema_detects_missing_and_unexpected_columns():
    from src.toolkit.validation import validate_schema

    expected_schema = {"anio": "i", "valor": "f"}
    inference_df = pd.DataFrame({"anio": [2022], "pais": ["Chile"]})  # falta "valor", sobra "pais"

    resultado = validate_schema(inference_df, expected_schema)

    assert resultado["esquema_valido"] is False
    assert resultado["columnas_faltantes"] == ["valor"]
    assert resultado["columnas_inesperadas"] == ["pais"]


def test_validate_schema_detects_silent_int_to_float_drift_from_a_single_nan():
    """El caso real que motiva esta función: una columna entera en train se
    vuelve float en inferencia apenas un solo valor llega como NaN -- pandas
    lo hace en silencio, sin avisar ni fallar."""
    from src.toolkit.validation import infer_schema, validate_schema

    train_df = pd.DataFrame({"anio": [2020, 2021, 2022]})
    assert train_df["anio"].dtype.kind == "i"

    inference_df = pd.DataFrame({"anio": [2023, None, 2025]})
    assert inference_df["anio"].dtype.kind == "f"  # promoción silenciosa de pandas

    resultado = validate_schema(inference_df, infer_schema(train_df))

    assert resultado["esquema_valido"] is False
    assert resultado["cambios_de_tipo"] == [{"columna": "anio", "dtype_esperado": "i", "dtype_actual": "f"}]
