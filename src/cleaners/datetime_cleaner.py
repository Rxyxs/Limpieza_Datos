"""Normaliza fechas en formatos y timezones heterogéneos a UTC en formato ISO 8601."""
from __future__ import annotations

import warnings

import pandas as pd

# Offsets fijos para abreviaciones de timezone comunes en los datos de origen. Son
# aproximados (sin reglas de horario de verano) — la misma ambigüedad que existe en
# datos reales cuando llegan con abreviaciones en vez de un offset UTC explícito.
TZ_ABBREVIATION_OFFSETS = {
    "UTC": "+00:00",
    "Z": "+00:00",
    "EST": "-05:00",
    "PST": "-08:00",
    "CET": "+01:00",
}

NULL_LIKE = {"", "nan", "none", "null", "n/a", "na", "-", "?"}


def _strip_named_timezone(text: str) -> tuple[str, str | None]:
    """Separa una abreviación de timezone al final del string (si existe) de su offset."""
    for abbr, offset in TZ_ABBREVIATION_OFFSETS.items():
        suffix = f" {abbr}"
        if text.endswith(suffix):
            return text[: -len(suffix)].strip(), offset
        if text.endswith(abbr) and text[: -len(abbr)] and text[: -len(abbr)][-1].isdigit():
            return text[: -len(abbr)].strip(), offset
    return text, None


def parse_to_utc(value) -> pd.Timestamp | None:
    """Parsea una fecha en cualquier formato/timezone reconocido y la devuelve en UTC.

    Limitación conocida e inherente sin metadatos de locale: una fecha como "07/09/2024"
    es genuinamente ambigua (día/mes vs. mes/día) cuando ambos valores son <= 12; se
    asume mes/día (`dayfirst=False`) y solo se reintenta con `dayfirst=True` si la
    primera lectura falla por completo (p. ej. día > 12), igual que haría un pipeline
    real sin más contexto sobre el origen de cada fecha.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if text.lower() in NULL_LIKE:
        return None

    stripped, offset = _strip_named_timezone(text)
    candidate = f"{stripped} {offset}" if offset else stripped

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(candidate, utc=True, errors="coerce", dayfirst=False)
        if pd.isna(parsed):
            parsed = pd.to_datetime(candidate, utc=True, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed


def clean_datetime_column(df: pd.DataFrame, column: str, output_column: str | None = None) -> pd.DataFrame:
    """Reemplaza `column` (o crea `output_column`) con timestamps UTC (`datetime64[ns, UTC]`)."""
    output_column = output_column or column
    df = df.copy()
    df[output_column] = df[column].apply(parse_to_utc)
    return df


def to_iso8601(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Formatea una columna de timestamps ya normalizados como texto ISO 8601."""
    df = df.copy()
    df[column] = df[column].apply(lambda ts: ts.isoformat() if pd.notna(ts) else None)
    return df
