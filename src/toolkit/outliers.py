"""Detección y tratamiento de outliers numéricos, vía winsorización IQR.

Distinto de `missing_data.py`: ese módulo rellena valores *faltantes*; este trata
valores *presentes* pero estadísticamente implausibles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Calcula los límites inferior/superior de Tukey: `Q1 - k*IQR`, `Q3 + k*IQR`."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clip_to_bounds(df: pd.DataFrame, column: str, lower: float, upper: float) -> tuple[pd.DataFrame, int]:
    """`transform`: recorta `column` a límites YA CALCULADOS (de `iqr_bounds`) --
    no recalcula nada a partir de `df`, así que es seguro llamarla sobre un
    conjunto de test que nunca participó del cálculo de esos límites.

    Valores nulos se ignoran (no se cuentan como outliers, no se tocan).
    Devuelve `(df_modificado, n_valores_recortados)`.
    """
    df = df.copy()
    df[column] = df[column].astype("float64")  # el recorte puede producir no-enteros
    non_null_mask = df[column].notna()
    is_outlier = non_null_mask & ((df[column] < lower) | (df[column] > upper))
    n_outliers = int(is_outlier.sum())

    df.loc[non_null_mask, column] = df.loc[non_null_mask, column].clip(lower, upper)
    return df, n_outliers


def winsorize_column(df: pd.DataFrame, column: str, k: float = 1.5) -> tuple[pd.DataFrame, int]:
    """Recorta (winsoriza) `column` a los límites IQR de Tukey, calculando y
    aplicando esos límites sobre el mismo `df` en un solo paso.

    Útil cuando no hay una frontera train/test que respetar -- si la hay, usar
    `iqr_bounds` sobre el histórico y `clip_to_bounds` sobre ambos subconjuntos
    con esos mismos límites, como hace `financial_bcch` en este repo.

    Devuelve `(df_modificado, n_valores_recortados)`.
    """
    df = df.copy()
    df[column] = df[column].astype("float64")
    non_null = df[column].dropna()
    if non_null.empty:
        return df, 0

    lower, upper = iqr_bounds(non_null, k=k)
    return clip_to_bounds(df, column, lower, upper)


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
    grupo (ej. producción mensual varía por empresa a propósito, retorno diario
    varía por instrumento) -- un IQR global trataría la variación *esperada* entre
    grupos como si fuera ruido, sobre-recortando el grupo de escala más alta.
    Winsorizar dentro de cada grupo separa esa variación estructural de los
    outliers reales dentro de cada uno.
    """
    df = df.copy()
    df[column] = df[column].astype("float64")
    total_outliers = 0
    for _, group_index in df.groupby(group_column).groups.items():
        group_slice = df.loc[group_index, [column]]
        winsorized_slice, n_outliers = winsorize_column(group_slice, column, k=k)
        df.loc[group_index, column] = winsorized_slice[column]
        total_outliers += n_outliers
    return df, total_outliers


def fix_implausible_level_jumps(
    df: pd.DataFrame, column: str, sort_by: str, threshold: float = 0.5, window: int = 7
) -> tuple[pd.DataFrame, int]:
    """Detecta y corrige un valor de NIVEL implausible en una serie con tendencia
    real (precios, tipos de cambio, índices) comparando cada punto contra la
    MEDIANA MÓVIL centrada de sus `window` vecinos, no contra el día anterior.

    La comparación día-a-día (`valor[t] / valor[t-1]`) tiene un punto ciego
    real: si dos días *consecutivos* están corruptos con valores parecidos
    entre sí (ej. un bug de fuente que persiste un día), su ratio mutuo se ve
    normal y ninguno de los dos se marca -- solo se detectan la entrada y
    salida del tramo corrupto, y al interpolar contra un vecino que también
    está corrupto, la corrección hereda el error en vez de arreglarlo. La
    mediana móvil es robusta a 1-2 puntos corruptos dentro de la ventana (la
    mayoría de los vecinos siguen siendo válidos), así que sigue reflejando
    el nivel real incluso ahí.

    Un desvío de `threshold` en log frente a la mediana local (0.5 ≈ ±65%)
    es casi siempre un error de captura en la fuente, no un movimiento real
    de mercado. Se reemplaza cada punto marcado por `NaN` y se interpola
    linealmente contra los vecinos ya corregidos.

    Devuelve `(df_corregido, n_valores_corregidos)`.
    """
    df = df.sort_values(sort_by).reset_index(drop=True).copy()
    rolling_median = df[column].rolling(window=window, center=True, min_periods=3).median()
    ratio = df[column] / rolling_median
    log_deviation = np.log(ratio.where(ratio > 0))
    is_error = log_deviation.abs() > threshold
    n_errors = int(is_error.sum())
    if n_errors:
        df.loc[is_error, column] = np.nan
        df[column] = df[column].interpolate(method="linear", limit_direction="both")
    return df, n_errors


def zscore_outlier_mask(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Marca valores a más de `threshold` desviaciones estándar de la media.

    Alternativa a IQR más agresiva en colas para datos aproximadamente normales;
    IQR es más robusto cuando la distribución es asimétrica (ej. precios de
    commodities, montos monetarios), que es el caso más común en este proyecto.
    """
    mean, std = series.mean(), series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index)
    z = (series - mean).abs() / std
    return z > threshold
