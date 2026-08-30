"""Detección y tratamiento de outliers numéricos, vía winsorización IQR.

Distinto de `missing_data_imputer.py`: ese módulo rellena valores *faltantes*;
este trata valores *presentes* pero estadísticamente implausibles (ej. un costo
de $50.000 en una categoría donde el resto factura en cientos) -- outliers reales
en `cost`/`response_time_hours` que sobrevivieron a `string_cleaner.parse_currency`
y a la generación sintética, no errores de parseo.
"""
from __future__ import annotations

import pandas as pd


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Calcula los límites inferior/superior de Tukey: `Q1 - k*IQR`, `Q3 + k*IQR`."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def winsorize_column(df: pd.DataFrame, column: str, k: float = 1.5) -> tuple[pd.DataFrame, int]:
    """Recorta (winsoriza) `column` a los límites IQR de Tukey.

    Valores nulos se ignoran (no se cuentan como outliers, no se tocan) para no
    pisar el trabajo de `missing_data_imputer`, que corre después en el pipeline
    -- este paso trata valores *presentes*.

    Devuelve `(df_modificado, n_valores_recortados)`.
    """
    df = df.copy()
    df[column] = df[column].astype("float64")  # el recorte puede producir no-enteros
    non_null = df[column].dropna()
    if non_null.empty:
        return df, 0

    lower, upper = iqr_bounds(non_null, k=k)
    is_outlier = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
    n_outliers = int(is_outlier.sum())

    df.loc[df[column].notna(), column] = df.loc[df[column].notna(), column].clip(lower, upper)
    return df, n_outliers


def winsorize_columns(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica `winsorize_column` a varias columnas. Devuelve `(df, {columna: n_recortados})`."""
    report: dict[str, int] = {}
    for column in columns:
        df, n_outliers = winsorize_column(df, column, k=k)
        report[column] = n_outliers
    return df, report


def winsorize_column_by_group(
    df: pd.DataFrame, column: str, group_column: str, k: float = 1.5
) -> tuple[pd.DataFrame, int]:
    """Winsoriza `column` con límites IQR calculados *por grupo* de `group_column`.

    Necesario cuando `column` tiene una escala genuinamente distinta según el
    grupo (ej. `response_time_hours` varía por `category` a propósito, ver
    `src/generators/dirty_data_generator.py::_resolution_hours`) -- un IQR
    global trataría la variación *esperada* entre grupos como si fuera ruido,
    sobre-recortando el grupo de escala más alta. Winsorizar dentro de cada
    grupo separa esa variación estructural de los outliers reales dentro de
    cada uno.
    """
    df = df.copy()
    df[column] = df[column].astype("float64")  # el recorte puede producir no-enteros
    total_outliers = 0
    for _, group_index in df.groupby(group_column).groups.items():
        group_slice = df.loc[group_index, [column]]
        winsorized_slice, n_outliers = winsorize_column(group_slice, column, k=k)
        df.loc[group_index, column] = winsorized_slice[column]
        total_outliers += n_outliers
    return df, total_outliers
