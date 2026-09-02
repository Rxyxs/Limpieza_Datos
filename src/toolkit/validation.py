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
