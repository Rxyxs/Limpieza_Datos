"""Validación de esquema del dataset limpio de tickets IT/SaaS, vía pydantic."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}
ALLOWED_STATUSES = {"open", "pending", "resolved", "closed"}
ALLOWED_CATEGORIES = {"billing", "technical", "account", "bug report", "feature request"}


class CleanTicketSchema(BaseModel):
    """Esquema esperado de una fila ya limpia, lista para análisis o carga a un DWH."""

    ticket_id: str = Field(min_length=1)
    company_name: str = Field(min_length=2)
    customer_email: str = Field(min_length=1)
    created_at: datetime
    resolved_at: Optional[datetime] = None
    priority: str = Field(min_length=1)
    status: str = Field(min_length=1)
    category: str = Field(min_length=1)
    agent_name: str = Field(min_length=2)
    cost: float

    @field_validator("customer_email")
    @classmethod
    def email_must_look_valid(cls, v: str) -> str:
        if not EMAIL_PATTERN.match(v):
            raise ValueError(f"email con formato inválido: {v!r}")
        return v

    @field_validator("priority")
    @classmethod
    def priority_must_be_known(cls, v: str) -> str:
        if v.lower() not in ALLOWED_PRIORITIES:
            raise ValueError(f"prioridad desconocida: {v!r}")
        return v

    @field_validator("status")
    @classmethod
    def status_must_be_known(cls, v: str) -> str:
        if v.lower() not in ALLOWED_STATUSES:
            raise ValueError(f"estado desconocido: {v!r}")
        return v

    @field_validator("category")
    @classmethod
    def category_must_be_known(cls, v: str) -> str:
        if v.lower() not in ALLOWED_CATEGORIES:
            raise ValueError(f"categoría desconocida: {v!r}")
        return v

    @field_validator("cost")
    @classmethod
    def cost_must_be_plausible(cls, v: float) -> float:
        if v < -100_000 or v > 1_000_000:
            raise ValueError(f"costo fuera de rango plausible: {v}")
        return v


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida cada fila de `df` contra `CleanTicketSchema`.

    Devuelve `(filas_validas, filas_invalidas)`; esta última incluye una columna
    adicional `validation_error` con el motivo del rechazo, para auditoría.
    """
    records = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

    valid_rows, invalid_rows = [], []
    for record in records:
        try:
            CleanTicketSchema(**record)
            valid_rows.append(record)
        except ValidationError as exc:
            invalid_rows.append({**record, "validation_error": str(exc)})

    valid_df = pd.DataFrame(valid_rows, columns=df.columns)
    invalid_df = pd.DataFrame(invalid_rows, columns=list(df.columns) + ["validation_error"])
    return valid_df, invalid_df
