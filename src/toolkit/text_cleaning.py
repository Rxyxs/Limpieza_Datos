"""Limpieza de texto: espacios, parseo de números/moneda, unificación fuzzy de nombres."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

NULL_LIKE = {"", "nan", "none", "null", "n/a", "na", "-", "?", "..", "...", ":"}
CURRENCY_TOKENS = ["US$", "USD", "EUR", "GBP", "CLP", "$", "€", "£"]


def strip_whitespace(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Recorta espacios externos y colapsa espacios internos repetidos en `columns`."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].apply(lambda v: re.sub(r"\s+", " ", str(v).strip()) if pd.notna(v) else v)
    return df


def parse_currency(value) -> float | None:
    """Convierte un string de moneda/número (símbolos y separadores mezclados) a float.

    Soporta formato US ("$1,200.50", coma de miles / punto decimal) y formato
    latinoamericano/europeo ("1.200,50", punto de miles / coma decimal), detectando
    cuál aplica por la posición relativa de la última coma y el último punto. Trata
    también marcadores de nota al pie comunes en Excel de organismos públicos
    (ej. "1234 (p)" -> 1234.0, provisional) quitando cualquier sufijo no numérico.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if text.lower() in NULL_LIKE:
        return None

    wrapped_in_parens = text.startswith("(") and text.endswith(")")
    negative = text.startswith("-") or wrapped_in_parens
    if wrapped_in_parens:
        text = text[1:-1].strip()
    text = text.lstrip("-").strip()

    for token in CURRENCY_TOKENS:
        text = text.replace(token, "")
    text = text.strip()

    # Marcadores de nota al pie tipo "(p)", "(e)", "*", "n.d." pegados al número.
    text = re.sub(r"\s*\([a-zA-Z]{1,3}\)\s*$", "", text)
    text = text.rstrip("*").strip()

    if not text:
        return None

    has_comma, has_dot = "," in text, "." in text
    if has_comma and has_dot:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma:
        if len(text.split(",")[-1]) == 2:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    try:
        result = float(text)
    except ValueError:
        return None

    return -result if negative else result


def normalize_case(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normaliza la capitalización (Title Case) de columnas categóricas de texto."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].apply(lambda v: str(v).strip().title() if pd.notna(v) else v)
    return df


def unify_similar_names(series: pd.Series, threshold: float = 80) -> pd.Series:
    """Colapsa variantes/typos de un mismo nombre a un único valor canónico.

    Agrupa strings con similitud `rapidfuzz.fuzz.WRatio >= threshold` (tras normalizar
    case/espacios/puntuación vía `rapidfuzz.utils.default_process`); dentro de cada
    grupo se conserva como canónico el nombre más frecuente. Los valores nulos o
    vacíos se preservan como nulos. Útil para nombres de empresa, país o categoría
    que llegan con variantes de un archivo a otro (ej. "SQM S.A." vs "SQM SA").
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    uniques = list(non_null.value_counts().index)

    canonical_map: dict[str, str] = {}
    assigned: set[str] = set()

    for name in uniques:
        if name in assigned:
            continue
        canonical_map[name] = name
        assigned.add(name)

        matches = process.extract(
            name, uniques, scorer=fuzz.WRatio, processor=utils.default_process,
            score_cutoff=threshold, limit=None,
        )
        for match_name, _score, _idx in matches:
            if match_name not in assigned:
                canonical_map[match_name] = name
                assigned.add(match_name)

    def resolve(value):
        if pd.isna(value):
            return value
        stripped = str(value).strip()
        return np.nan if stripped == "" else canonical_map.get(stripped, stripped)

    return series.apply(resolve)
