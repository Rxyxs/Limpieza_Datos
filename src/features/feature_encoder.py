"""Codificación de variables categóricas -- el paso final antes de que el dataset
limpio sea consumible por un modelo de ML (scikit-learn no acepta texto crudo).

Separado de `src/validators/schema_validator.py` a propósito: la validación de
esquema garantiza que el dato limpio es *correcto*; este módulo lo transforma a
la representación numérica que un modelo *necesita*, una responsabilidad
distinta que no debería mezclarse con la de validar.
"""
from __future__ import annotations

import pandas as pd

# Orden real de severidad -- no alfabético -- por eso `priority` se codifica
# ordinal (0..3) y no con one-hot: para un modelo, "critical" ES más que "low"
# de una forma que el one-hot descarta.
PRIORITY_ORDER = ["low", "medium", "high", "critical"]


def encode_priority_ordinal(df: pd.DataFrame, column: str = "priority") -> pd.DataFrame:
    """Codifica `column` como entero ordinal 0..3 según `PRIORITY_ORDER`.

    Valores no reconocidos o nulos quedan como `NaN` (no se inventa un valor
    intermedio) -- deben resolverse aguas arriba, no silenciarse acá.
    """
    df = df.copy()
    order_map = {level: i for i, level in enumerate(PRIORITY_ORDER)}
    df[f"{column}_encoded"] = df[column].str.lower().map(order_map)
    return df


def encode_categorical_onehot(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot-encodea `columns` (categorías sin orden natural: `category`, `status`).

    A diferencia de `priority`, ninguna categoría de soporte es "mayor" que
    otra, así que un entero ordinal introduciría una relación falsa que un
    modelo lineal (o cualquiera sensible a magnitud) leería como real.
    """
    return pd.get_dummies(df, columns=columns, prefix=columns, dtype=int)


def build_ml_ready_dataset(
    df: pd.DataFrame,
    onehot_columns: list[str] | None = None,
    priority_column: str = "priority",
    drop_original_priority: bool = True,
) -> pd.DataFrame:
    """Encadena la codificación ordinal de prioridad + one-hot del resto.

    `onehot_columns` por defecto es `["category", "status"]` si no se especifica.
    Devuelve un DataFrame donde toda columna categórica de entrada quedó
    representada numéricamente, listo para entrenar un modelo.
    """
    onehot_columns = onehot_columns if onehot_columns is not None else ["category", "status"]

    result = encode_priority_ordinal(df, column=priority_column)
    if drop_original_priority:
        result = result.drop(columns=[priority_column])

    return encode_categorical_onehot(result, columns=onehot_columns)
