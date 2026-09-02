"""Técnicas de limpieza específicas de Excel real de organismos públicos/corporativos:
encabezados multi-fila, celdas combinadas (que openpyxl lee como valor solo en la
primera celda y `NaN` en el resto), formato ancho (una columna por año/empresa) que
necesita volverse largo para análisis, y notas al pie mezcladas con los datos.

Este es el módulo que distingue "limpieza de CSV" de "limpieza de Excel real para
un data lake / data warehouse" -- el tipo de trabajo que hace una consultora al
recibir un archivo `.xlsx` de un cliente en vez de un export ya tabular.
"""
from __future__ import annotations

import re

import pandas as pd


def flatten_multirow_header(df_raw: pd.DataFrame, header_rows: int, sep: str = " / ") -> pd.DataFrame:
    """Combina las primeras `header_rows` filas de un DataFrame leído sin encabezado
    (`pd.read_excel(..., header=None)`) en un único encabezado por columna.

    Encabezados multi-fila (ej. fila 0 = "Producción (t)", fila 1 = nombre de
    empresa) son comunes en reportes de organismos públicos porque replican el
    layout visual del Excel original, no un esquema tabular. Las celdas
    combinadas dejan `NaN` en la fila de header bajo la celda madre -- se
    propagan hacia adelante (`ffill`) antes de concatenar, igual que Excel las
    muestra visualmente fusionadas.
    """
    header_block = df_raw.iloc[:header_rows].ffill(axis=1)
    combined = header_block.astype(str).apply(
        lambda col: sep.join(dict.fromkeys(v.strip() for v in col if v.strip().lower() not in {"nan", ""})),
        axis=0,
    )
    body = df_raw.iloc[header_rows:].reset_index(drop=True)
    body.columns = combined.values
    return body


def detect_header_row(df_raw: pd.DataFrame, expected_tokens: list[str], max_scan_rows: int = 15) -> int:
    """Detecta el índice de fila que contiene el encabezado real, escaneando las
    primeras `max_scan_rows` filas de un Excel leído sin encabezado.

    Reportes reales suelen anteponer filas de título/logo/fecha de publicación
    antes de la tabla real -- el encabezado no siempre está en la fila 0.
    Retorna la primera fila cuyo contenido (en minúsculas) incluye la mayoría de
    `expected_tokens`.
    """
    best_row, best_hits = 0, -1
    for i in range(min(max_scan_rows, len(df_raw))):
        row_text = " ".join(str(v).lower() for v in df_raw.iloc[i].tolist())
        hits = sum(1 for token in expected_tokens if token.lower() in row_text)
        if hits > best_hits:
            best_row, best_hits = i, hits
    return best_row


def strip_footnote_markers(value):
    """Quita marcadores de nota al pie pegados a un valor numérico-como-texto:
    "1234 (p)" (provisional), "1234*", "1234 1/" -> "1234". No convierte a
    número -- solo limpia el string; usar junto a `parse_currency` para el
    parseo numérico final.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(value).strip()
    text = re.sub(r"\s*\([a-zA-Z]{1,3}\)\s*$", "", text)  # "(p)", "(e)"
    text = re.sub(r"\s*\d+/\s*$", "", text)  # "1/", "2/" (referencia a nota al pie numerada)
    return text.rstrip("*").strip()


def wide_years_to_long(
    df: pd.DataFrame, id_vars: list[str], year_pattern: str = r"^(19|20)\d{2}$",
    var_name: str = "periodo", value_name: str = "valor",
) -> pd.DataFrame:
    """Convierte un DataFrame ancho (una columna por año, patrón común en reportes
    estadísticos) a formato largo (`id_vars` + `periodo` + `valor`), condición
    necesaria para análisis de series de tiempo o carga a un data warehouse
    (esquema estrella: una fila por hecho, no una columna por período).
    """
    year_cols = [c for c in df.columns if re.match(year_pattern, str(c).strip())]
    long_df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name=var_name, value_name=value_name)
    long_df[var_name] = long_df[var_name].astype(str).str.strip().astype(int)
    return long_df


def drop_subtotal_rows(df: pd.DataFrame, label_column: str, subtotal_markers: list[str]) -> tuple[pd.DataFrame, int]:
    """Quita filas de subtotal/total que Excel reportes suelen intercalar entre
    filas de detalle (ej. "Total", "Subtotal Región", "TOTAL PAÍS") -- si no se
    filtran antes de sumar/promediar, duplican el conteo de lo que agregan.

    Devuelve `(df_filtrado, n_filas_quitadas)`.
    """
    pattern = "|".join(re.escape(m.lower()) for m in subtotal_markers)
    is_subtotal = df[label_column].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
    return df.loc[~is_subtotal].reset_index(drop=True), int(is_subtotal.sum())
