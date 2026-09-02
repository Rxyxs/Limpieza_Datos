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
