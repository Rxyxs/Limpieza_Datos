"""Pruebas unitarias para src/toolkit/missing_data.py."""
import numpy as np
import pandas as pd

from src.toolkit.missing_data import (
    impute_numeric_by_category,
    interpolate_numeric_column,
    interpolate_within_group,
    missingness_report,
)


def test_impute_numeric_by_category_uses_conditional_mean():
    df = pd.DataFrame({
        "categoria": ["A", "A", "A", "B"],
        "valor": [100.0, 200.0, np.nan, np.nan],
    })
    result = impute_numeric_by_category(df, value_column="valor", category_column="categoria")

    assert result.loc[2, "valor"] == 150.0
    assert result.loc[3, "valor"] == result["valor"].iloc[:3].mean()


def test_interpolate_numeric_column_fills_gap_linearly():
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "valor": [10.0, np.nan, 30.0],
    })
    result = interpolate_numeric_column(df, column="valor", sort_by="fecha")
    assert result.loc[1, "valor"] == 20.0


def test_interpolate_within_group_does_not_leak_across_groups():
    # Sin agrupar, un unico gap lineal entre el ultimo valor de "CL" y el primero
    # de "PE" produciria un numero sin sentido fisico -- por grupo, cada pais
    # interpola solo contra sus propios vecinos temporales.
    df = pd.DataFrame({
        "pais": ["CL", "CL", "CL", "PE", "PE", "PE"],
        "anio": [2020, 2021, 2022, 2020, 2021, 2022],
        "valor": [10.0, np.nan, 30.0, 1000.0, np.nan, 1002.0],
    })
    result = interpolate_within_group(df, column="valor", group_column="pais", sort_by="anio")

    assert result.loc[1, "valor"] == 20.0
    assert result.loc[4, "valor"] == 1001.0


def test_missingness_report_sorts_by_pct_nulos_desc():
    df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
    report = missingness_report(df)
    assert report.iloc[0]["columna"] == "a"
    assert report.iloc[0]["n_nulos"] == 2
    assert report.iloc[1]["n_nulos"] == 0
