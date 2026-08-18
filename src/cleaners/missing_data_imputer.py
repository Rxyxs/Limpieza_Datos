"""Estrategias de imputación de valores faltantes para el pipeline de limpieza IT/SaaS."""
from __future__ import annotations

import pandas as pd


def impute_numeric_by_category(df: pd.DataFrame, value_column: str, category_column: str) -> pd.DataFrame:
    """Imputa nulos en `value_column` con la media condicional por `category_column`.

    Si una categoría no tiene ningún valor no nulo, recurre a la media global de la
    columna.
    """
    df = df.copy()
    global_mean = df[value_column].mean()
    category_means = df.groupby(category_column)[value_column].transform("mean")
    df[value_column] = df[value_column].fillna(category_means).fillna(global_mean)
    return df


def interpolate_numeric_column(df: pd.DataFrame, column: str, sort_by: str | None = None) -> pd.DataFrame:
    """Imputa nulos en `column` por interpolación lineal respetando el orden temporal.

    Ordena por `sort_by` (típicamente una columna de fecha) antes de interpolar y
    restaura el orden original de las filas al terminar.
    """
    df = df.copy()
    order = df.sort_values(sort_by).index if sort_by else df.index
    interpolated = df.loc[order, column].interpolate(method="linear", limit_direction="both")
    df.loc[order, column] = interpolated
    return df
