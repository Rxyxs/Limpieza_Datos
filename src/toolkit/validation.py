"""Validación de esquema genérica vía pydantic: cada dominio define su propio
`BaseModel` (las reglas de negocio son específicas de cada dataset), y usa esta
única función para aplicarlo fila por fila y separar válidas de inválidas.
"""
from __future__ import annotations

from typing import Type

import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_dataframe(df: pd.DataFrame, schema: Type[BaseModel]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida cada fila de `df` contra el modelo pydantic `schema`.

    Devuelve `(filas_validas, filas_invalidas)`; esta última incluye una columna
    adicional `validation_error` con el motivo del rechazo, para auditoría --
    en un pipeline real nunca se descartan filas inválidas en silencio.
    """
    records = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

    valid_rows, invalid_rows = [], []
    for record in records:
        try:
            schema(**record)
            valid_rows.append(record)
        except ValidationError as exc:
            invalid_rows.append({**record, "validation_error": str(exc)})

    valid_df = pd.DataFrame(valid_rows, columns=df.columns)
    invalid_df = pd.DataFrame(invalid_rows, columns=list(df.columns) + ["validation_error"])
    return valid_df, invalid_df


def infer_schema(df: pd.DataFrame) -> dict[str, str]:
    """Congela el esquema de `df` como `{columna: dtype.kind}` (ej. entrenamiento),
    para comparar más tarde contra datos nuevos con `validate_schema`.

    Se guarda el `kind` (entero/float/texto/booleano/fecha) y no el dtype exacto
    porque int32 vs int64 no es un drift que interese acá -- int -> float sí lo
    es, y es precisamente el caso silencioso que produce inyectar un solo NaN en
    una columna que antes era entera.
    """
    return {column: df[column].dtype.kind for column in df.columns}


def validate_schema(df: pd.DataFrame, expected_schema: dict[str, str]) -> dict:
    """Compara `df` (ej. datos de inferencia) contra `expected_schema` (de
    `infer_schema` sobre el set de entrenamiento, o armado a mano).

    Detecta tres formas de drift de esquema, cada una silenciosa si no se
    chequea explícitamente: columnas que el pipeline espera y ya no llegan,
    columnas nuevas no esperadas, y columnas presentes que cambiaron de tipo
    (el caso clásico: una columna entera se vuelve float en cuanto aparece un
    solo NaN, algo que pandas nunca avisa por sí solo).

    Devuelve un dict con las tres listas y `esquema_valido` (True solo si las
    tres están vacías).
    """
    actual_columns = set(df.columns)
    expected_columns = set(expected_schema)

    columnas_faltantes = sorted(expected_columns - actual_columns)
    columnas_inesperadas = sorted(actual_columns - expected_columns)

    cambios_de_tipo = []
    for column in sorted(expected_columns & actual_columns):
        dtype_esperado = expected_schema[column]
        dtype_actual = df[column].dtype.kind
        if dtype_actual != dtype_esperado:
            cambios_de_tipo.append({
                "columna": column, "dtype_esperado": dtype_esperado, "dtype_actual": dtype_actual,
            })

    return {
        "columnas_faltantes": columnas_faltantes,
        "columnas_inesperadas": columnas_inesperadas,
        "cambios_de_tipo": cambios_de_tipo,
        "esquema_valido": not (columnas_faltantes or columnas_inesperadas or cambios_de_tipo),
    }
