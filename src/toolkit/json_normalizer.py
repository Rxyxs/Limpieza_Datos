"""Aplana una columna JSON anidada en columnas tabulares."""
from __future__ import annotations

import json

import pandas as pd

NULL_LIKE_JSON = {"", "null", "none", "nan", "{}"}


def _safe_parse(value) -> dict:
    """Parsea un valor JSON potencialmente inválido/vacío a dict, sin lanzar excepciones."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if text.lower() in NULL_LIKE_JSON:
        return {}

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_json_column(df: pd.DataFrame, column: str, prefix: str | None = None) -> pd.DataFrame:
    """Reemplaza `column` (JSON anidado de profundidad variable) por columnas planas.

    Cada ruta anidada se convierte en una columna `<prefix>_<ruta>`; las listas se
    serializan como texto separado por comas para conservar una fila por registro.
    """
    prefix = prefix or column
    parsed = df[column].apply(_safe_parse)
    flat = pd.json_normalize(parsed.tolist(), sep="_")

    if flat.empty:
        return df.drop(columns=[column])

    flat = flat.add_prefix(f"{prefix}_")
    for col in flat.columns:
        flat[col] = flat[col].apply(lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v)

    return pd.concat([df.drop(columns=[column]).reset_index(drop=True), flat.reset_index(drop=True)], axis=1)
