"""Codificación de variables categóricas y escalado -- el paso final antes de que
un dataset limpio sea consumible por un modelo de ML (los modelos no aceptan texto
crudo ni escalas arbitrariamente distintas entre features).

Separado de `validation.py` a propósito: la validación de esquema garantiza que el
dato limpio es *correcto*; este módulo lo transforma a la representación numérica
que un modelo *necesita*, una responsabilidad distinta que no debería mezclarse
con la de validar.
"""
from __future__ import annotations

import pandas as pd


def encode_ordinal(df: pd.DataFrame, column: str, order: list[str], output_column: str | None = None) -> pd.DataFrame:
    """Codifica `column` como entero ordinal 0..n-1 según el orden explícito en `order`.

    Usar cuando las categorías tienen un orden real (ej. "low" < "medium" < "high",
    o un rating). Valores no reconocidos o nulos quedan como `NaN` -- deben
    resolverse aguas arriba, no silenciarse acá.
    """
    df = df.copy()
    output_column = output_column or f"{column}_encoded"
    order_map = {level: i for i, level in enumerate(order)}
    df[output_column] = df[column].astype(str).str.lower().map(order_map)
    return df


def encode_categorical_onehot(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot-encodea `columns` (categorías sin orden natural).

    A diferencia de una variable ordinal, ninguna categoría es "mayor" que otra,
    así que un entero introduciría una relación falsa que un modelo lineal (o
    cualquiera sensible a magnitud) leería como real.
    """
    return pd.get_dummies(df, columns=columns, prefix=columns, dtype=int)


def zscore_scale(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Estandariza `columns` a media 0 / desviación estándar 1.

    Necesario antes de entrenar una red neuronal: sin escalar, una feature en
    la escala de millones (ej. producción en toneladas) domina el gradiente
    frente a una feature en [0,1] (ej. una tasa), independiente de su relevancia
    real. Devuelve también `(media, std)` por columna para poder revertir la
    transformación sobre las predicciones si el target también fue escalado.
    """
    df = df.copy()
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        std = std if std and std > 1e-9 else 1.0
        df[col] = (df[col] - mean) / std
        stats[col] = (mean, std)
    return df, stats


def inverse_zscore(values, mean: float, std: float):
    """Revierte `zscore_scale` sobre un array/serie de valores (ej. predicciones del modelo)."""
    return values * std + mean
