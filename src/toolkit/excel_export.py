"""Escritura del dato ya limpio de vuelta a `.xlsx`, en el formato en que una
consultora lo entrega: un libro multi-hoja, con encabezado congelado, autofiltro,
anchos de columna calculados, formatos numéricos declarados y una hoja
`Diccionario` que documenta cada columna de cada hoja.

`df.to_excel(path)` produce un archivo técnicamente válido y prácticamente
inservible: encabezado que se pierde al scrollear, columnas de 8 caracteres que
muestran `####`, fechas como números de serie y ninguna indicación de qué
significa cada columna ni cuánto le falta. El destinatario de un entregable de
limpieza casi nunca es otro script -- es una persona que abre el archivo en
Excel -- y el diccionario de datos es lo que permite que un tercero audite el
resultado sin leer el código que lo generó.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Excel rechaza estos caracteres en el nombre de una hoja y trunca a 31 chars.
INVALID_SHEET_CHARS = r"[\[\]\:\*\?\/\\]"
MAX_SHEET_NAME = 31

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def safe_sheet_name(name: str, taken: set[str] | None = None) -> str:
    """Normaliza un nombre de hoja a lo que Excel acepta (<=31 chars, sin `[]:*?/\\`),
    resolviendo colisiones con un sufijo numérico.

    Vale la pena hacerlo explícito y no dejárselo a la excepción de openpyxl:
    los nombres de hoja suelen venir de datos (una hoja por región, por cliente,
    por indicador), y dos nombres largos distintos pueden colapsar al mismo
    string al truncarse a 31 caracteres.
    """
    cleaned = re.sub(INVALID_SHEET_CHARS, "-", str(name)).strip() or "Hoja"
    cleaned = cleaned[:MAX_SHEET_NAME]
    if taken is None:
        return cleaned

    candidate, counter = cleaned, 2
    while candidate in taken:
        suffix = f"_{counter}"
        candidate = f"{cleaned[: MAX_SHEET_NAME - len(suffix)]}{suffix}"
        counter += 1
    taken.add(candidate)
    return candidate


def build_data_dictionary(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Construye el diccionario de datos del entregable: una fila por columna de
    cada hoja, con tipo, filas, no-nulos, % faltante, valores únicos, mínimo,
    máximo y un ejemplo real.

    El % faltante y el mínimo/máximo por columna son los dos campos que más
    trabajo ahorran: dejan auditable, sin abrir el dato, si una columna quedó
    vacía tras la limpieza y si su rango es plausible (una tasa que llega a 250%
    o una fecha en 1899 -- típico número de serie de Excel mal convertido --
    saltan a la vista en la misma hoja del entregable).
    """
    rows = []
    for sheet_name, df in sheets.items():
        for col in df.columns:
            series = df[col]
            n_valid = int(series.notna().sum())
            is_ordered = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)
            example = series.dropna().iloc[0] if n_valid else None

            rows.append({
                "hoja": sheet_name,
                "columna": str(col),
                "tipo": str(series.dtype),
                "n_filas": int(len(series)),
                "n_no_nulos": n_valid,
                "pct_faltante": round(100 * (1 - n_valid / len(series)), 2) if len(series) else 0.0,
                "n_unicos": int(series.nunique(dropna=True)),
                "minimo": series.min() if is_ordered and n_valid else None,
                "maximo": series.max() if is_ordered and n_valid else None,
                "ejemplo": str(example)[:80] if example is not None else None,
            })
    return pd.DataFrame(rows)


def _format_worksheet(
    worksheet, df: pd.DataFrame, freeze_header: bool, autofilter: bool,
    number_formats: Mapping[str, str] | None, max_column_width: int,
) -> None:
    """Aplica el formato de presentación a una hoja ya escrita."""
    for cell in worksheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)

    if freeze_header:
        worksheet.freeze_panes = "A2"
    if autofilter and len(df.columns):
        worksheet.auto_filter.ref = worksheet.dimensions

    for position, col in enumerate(df.columns, start=1):
        letter = get_column_letter(position)

        # Ancho: el máximo entre el encabezado y una muestra de los valores. Se
        # muestrean 200 filas y no la columna entera para que el costo no crezca
        # con el largo del dato -- el ancho de columna no necesita ser exacto.
        sample = df[col].dropna().astype(str).head(200)
        widest = max([len(str(col))] + [len(v) for v in sample]) if len(sample) else len(str(col))
        worksheet.column_dimensions[letter].width = min(max(widest + 2, 10), max_column_width)

        fmt = (number_formats or {}).get(col)
        if fmt is None and pd.api.types.is_datetime64_any_dtype(df[col]):
            fmt = "yyyy-mm-dd"
        if fmt:
            for cell in worksheet[letter][1:]:
                cell.number_format = fmt


def write_analysis_workbook(
    path: str | Path,
    sheets: Mapping[str, pd.DataFrame],
    data_dictionary: bool = True,
    freeze_header: bool = True,
    autofilter: bool = True,
    number_formats: Mapping[str, str] | None = None,
    max_column_width: int = 60,
) -> Path:
    """Escribe `sheets` (nombre -> DataFrame) como un `.xlsx` listo para entregar y
    devuelve la ruta escrita.

    `number_formats` mapea nombre de columna -> código de formato de Excel (ej.
    `{"produccion_ton": "#,##0.0", "participacion": "0.00%"}`) y se aplica en
    cualquier hoja donde exista esa columna; las columnas `datetime` reciben
    `yyyy-mm-dd` por defecto, que es lo que evita que una fecha se vea como
    `45292` al abrir el archivo.

    Con `data_dictionary=True` se agrega una hoja `Diccionario` al final,
    construida con `build_data_dictionary` sobre las mismas hojas.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    taken: set[str] = set()
    named = {safe_sheet_name(name, taken): df for name, df in sheets.items()}
    if data_dictionary:
        named[safe_sheet_name("Diccionario", taken)] = build_data_dictionary(named)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in named.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            _format_worksheet(
                writer.sheets[sheet_name], df, freeze_header, autofilter,
                number_formats, max_column_width,
            )
    return path


def split_to_sheets(df: pd.DataFrame, by: str, max_sheets: int = 50) -> dict[str, pd.DataFrame]:
    """Parte un DataFrame largo en un dict de hojas, una por valor de `by`.

    Es la forma en que se pide un entregable en la práctica ("mándame un Excel
    con una pestaña por región"). El tope `max_sheets` es deliberado: partir por
    una columna de alta cardinalidad genera un libro de cientos de pestañas que
    Excel abre lentísimo y nadie navega -- si se supera, conviene entregar el
    dato largo con un autofiltro, y por eso acá falla en vez de generarlo igual.
    """
    groups = list(df.groupby(by, dropna=False))
    if len(groups) > max_sheets:
        raise ValueError(
            f"'{by}' genera {len(groups)} hojas (tope {max_sheets}); "
            "usar una columna de menor cardinalidad o entregar el dato en formato largo."
        )
    return {
        str(key): group.drop(columns=[by]).reset_index(drop=True)
        for key, group in groups
    }
