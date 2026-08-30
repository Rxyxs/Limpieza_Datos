"""Filtrado de datos por reglas de negocio -- distinto de la validación de esquema.

`schema_validator.validate_dataframe()` clasifica cada fila en válida/inválida
para *auditoría* (una fila inválida se reporta, no desaparece silenciosamente).
Este módulo, en cambio, *recorta el alcance* del dataset para un análisis o
modelo específico -- ej. "solo tickets ya resueltos" para pronosticar tiempo de
resolución no tiene sentido sobre tickets todavía abiertos. Filtrar no es lo
mismo que invalidar: una fila filtrada acá puede ser una fila perfectamente
válida que simplemente está fuera del alcance del análisis actual.
"""
from __future__ import annotations

import pandas as pd


def filter_resolved_tickets(df: pd.DataFrame, resolved_column: str = "resolved_at") -> pd.DataFrame:
    """Conserva solo tickets con `resolved_column` no nulo.

    Necesario antes de cualquier análisis de tiempo de resolución: un ticket
    todavía abierto no tiene un `response_time_hours` real, solo uno que
    `missing_data_imputer` habrá interpolado -- una fila imputada no debería
    alimentar el *entrenamiento* de un modelo que pronostica ese mismo valor.
    """
    return df[df[resolved_column].notna()].reset_index(drop=True)


def filter_by_date_range(
    df: pd.DataFrame,
    column: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Conserva filas con `column` dentro de `[start, end]` (límites inclusive, opcionales)."""
    series = pd.to_datetime(df[column], utc=True, errors="coerce")
    mask = series.notna()
    if start is not None:
        mask &= series >= pd.to_datetime(start, utc=True)
    if end is not None:
        mask &= series <= pd.to_datetime(end, utc=True)
    return df[mask].reset_index(drop=True)


def filter_positive_cost(df: pd.DataFrame, cost_column: str = "cost") -> pd.DataFrame:
    """Descarta filas con costo negativo (reembolsos) o cero.

    Los reembolsos son un evento de negocio real y válido (por eso
    `string_cleaner.parse_currency` los preserva como negativos en vez de
    tratarlos como error de parseo) pero representan un proceso distinto al
    de "cuánto cuesta resolver un ticket" -- mezclarlos sesgaría un modelo de
    costo/tiempo de resolución hacia abajo de forma artificial.
    """
    return df[df[cost_column] > 0].reset_index(drop=True)


def apply_ml_scope_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Encadena los filtros que definen el alcance típico para modelar
    tiempo de resolución / costo: solo tickets resueltos, con costo positivo.
    """
    df = filter_resolved_tickets(df)
    df = filter_positive_cost(df)
    return df
