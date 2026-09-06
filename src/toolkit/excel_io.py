"""Lectura de workbooks `.xlsx` reales a nivel de *archivo*, no de tabla.

`excel_cleaning.py` resuelve la forma de los datos una vez que ya están en un
DataFrame (encabezados multi-fila, formato ancho, subtotales). Este módulo
resuelve el paso anterior, el que rompe antes de llegar ahí: qué hojas trae el
archivo y cuáles están ocultas, qué celdas están combinadas (openpyxl y pandas
dejan `None` en todas menos la superior-izquierda), qué filas/columnas ocultó
un analista, qué celdas son en realidad literales de error de Excel (`#N/A`,
`#REF!`), qué columna de fechas llegó como número de serie, y qué columnas
mezclan texto con números (el triángulo verde de "número almacenado como
texto").

Ninguno de estos problemas existe en un CSV: son propios del formato `.xlsx`,
que guarda una planilla -- con su formato visual, sus fórmulas y sus celdas
ocultas -- y no una tabla.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string

# Literales que Excel escribe en la celda cuando una fórmula falla. Al leer con
# `data_only=True` llegan como estos strings exactos, no como NaN: una columna
# numérica con un solo "#DIV/0!" se lee entera como `object` y todo cálculo
# posterior falla o, peor, se saltea silenciosamente.
EXCEL_ERROR_LITERALS = frozenset({
    "#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NULL!", "#NUM!",
    "#SPILL!", "#CALC!", "#GETTING_DATA", "#FIELD!", "#UNKNOWN!",
})

# Caracteres invisibles que llegan pegados al texto cuando alguien copió desde
# una página web o desde otro Excel: espacio duro (Alt+0160), zero-width space,
# zero-width joiner/non-joiner, word joiner y BOM. `str.strip()` no los toca.
INVISIBLE_CHARS = "\u00a0\u200b\u200c\u200d\u2060\ufeff"


def _resolve_sheet(workbook, sheet):
    """Acepta índice (int) o nombre (str) de hoja, como `pd.read_excel`."""
    if isinstance(sheet, int):
        return workbook.worksheets[sheet]
    return workbook[sheet]


def inventory_workbook(path: str | Path) -> pd.DataFrame:
    """Inventario de las hojas de un `.xlsx` antes de leer nada: nombre, estado
    (`visible` / `hidden` / `veryHidden`), filas y columnas declaradas.

    Es el primer comando a correr sobre un archivo de cliente. Un `.xlsx`
    institucional trae rutinariamente hojas ocultas (borradores, tablas de
    parámetros, una versión anterior del mismo dato) que `pd.read_excel(...,
    sheet_name=None)` levanta como si fueran hojas de datos igual de válidas;
    `veryHidden` además no se puede mostrar desde la interfaz de Excel, así que
    ni el cliente que envió el archivo suele saber que están ahí.

    Se abre en modo `read_only` para no materializar el contenido: sirve igual
    sobre archivos de decenas de MB.
    """
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = [
            {
                "hoja": ws.title,
                "estado": ws.sheet_state,
                "visible": ws.sheet_state == "visible",
                "filas": int(ws.max_row or 0),
                "columnas": int(ws.max_column or 0),
            }
            for ws in workbook.worksheets
        ]
    finally:
        workbook.close()
    return pd.DataFrame(rows)


def _hidden_indices(worksheet) -> tuple[list[int], list[int]]:
    """Índices 0-based de filas y columnas ocultas de una hoja ya abierta."""
    hidden_rows = sorted(idx - 1 for idx, dim in worksheet.row_dimensions.items() if dim.hidden)

    hidden_cols: set[int] = set()
    for letter, dim in worksheet.column_dimensions.items():
        if not dim.hidden:
            continue
        # Una entrada de `column_dimensions` puede describir un rango completo
        # (ej. columnas D a G ocultas de una vez), no una sola columna.
        start = dim.min or column_index_from_string(letter)
        end = dim.max or start
        hidden_cols.update(range(start - 1, end))
    return hidden_rows, sorted(hidden_cols)


def hidden_rows_and_columns(path: str | Path, sheet: int | str = 0) -> tuple[list[int], list[int]]:
    """Devuelve `(filas_ocultas, columnas_ocultas)` como índices 0-based relativos
    a la hoja leída sin encabezado (`header=None`).

    Ocultar una fila en Excel es la forma informal más común de "borrar" un
    registro sin borrarlo: la fila sigue en el archivo y pandas la lee como
    cualquier otra. Si el cliente ocultó 12 filas duplicadas antes de enviar el
    archivo, el conteo que ve él y el que ve el pipeline no coinciden, y la
    diferencia no aparece en ningún log.

    Que esté oculto no dice *por qué* lo está, y por eso esta función reporta en
    vez de filtrar: en el `.xlsx` real de COCHILCO que usa este repo conviven los
    dos motivos opuestos en la misma hoja (ver `read_sheet_expanding_merges`).
    """
    workbook = load_workbook(path, data_only=True)
    try:
        return _hidden_indices(_resolve_sheet(workbook, sheet))
    finally:
        workbook.close()


def read_sheet_expanding_merges(
    path: str | Path,
    sheet: int | str = 0,
    drop_hidden_rows: bool = False,
    drop_hidden_columns: bool = False,
) -> pd.DataFrame:
    """Lee una hoja sin encabezado propagando el valor de cada rango combinado a
    todas sus celdas, opcionalmente descartando filas y/o columnas ocultas.

    Excel guarda una celda combinada como el valor en la celda superior-izquierda
    del rango y `None` en el resto; visualmente el usuario ve una etiqueta que
    abarca cinco filas, pero pandas lee una etiqueta y cuatro nulos. `ffill` no
    es equivalente: propaga también sobre nulos que sí son nulos reales, y solo
    en una dirección. Acá se rellena exactamente el rectángulo que Excel declaró
    combinado, ni una celda más.

    **Las dos banderas de ocultos van separadas y por defecto en `False` a
    propósito**, y el `.xlsx` real de COCHILCO de este repo explica por qué: esa
    hoja tiene 144 filas ocultas y 1 columna oculta, y significan cosas
    opuestas. Las 144 filas ocultas son *todas* observaciones mensuales reales
    (COCHILCO colapsó el detalle mensual para dejar a la vista solo los
    subtotales anuales): descartarlas borra 144 de los 156 meses del archivo. La
    única columna oculta, en cambio, es `Chuqui y R.Tomic`, uno de los subtotales
    encubiertos que este proyecto ya documenta -- ahí ocultar sí marca
    redundancia. Un `drop_hidden=True` único aplicado a los dos ejes habría hecho
    exactamente lo incorrecto en la mitad de los casos, en un solo archivo.

    Devuelve un DataFrame con columnas `RangeIndex`, listo para pasar a
    `excel_cleaning.detect_header_row` / `flatten_multirow_header`.
    """
    workbook = load_workbook(path, data_only=True)
    try:
        worksheet = _resolve_sheet(workbook, sheet)
        grid = [[cell.value for cell in row] for row in worksheet.iter_rows()]
        df = pd.DataFrame(grid)

        for merged in worksheet.merged_cells.ranges:
            anchor = worksheet.cell(row=merged.min_row, column=merged.min_col).value
            if anchor is None:
                continue
            df.iloc[merged.min_row - 1 : merged.max_row, merged.min_col - 1 : merged.max_col] = anchor

        if drop_hidden_rows or drop_hidden_columns:
            hidden_rows, hidden_cols = _hidden_indices(worksheet)
            if drop_hidden_rows:
                df = df.drop(index=[r for r in hidden_rows if r < len(df)])
            if drop_hidden_columns:
                df = df.drop(columns=[c for c in hidden_cols if c < df.shape[1]])
    finally:
        workbook.close()

    df = df.reset_index(drop=True)
    df.columns = pd.RangeIndex(df.shape[1])
    return df


def coerce_excel_errors(
    df: pd.DataFrame, columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reemplaza los literales de error de Excel (`#N/A`, `#REF!`, `#DIV/0!`...)
    por `NaN` y devuelve `(df_limpio, reporte)` con el conteo por columna.

    El reporte importa tanto como la limpieza: un `#N/A` es un dato que nunca
    existió (una fórmula `VLOOKUP` que no encontró su clave) y un `#REF!` es una
    fórmula que apuntaba a celdas que alguien borró -- la segunda situación dice
    que el archivo está roto, no que falte un dato, y merece volver al cliente
    en vez de imputarse. Convertirlos a `NaN` en silencio borra esa distinción.
    """
    target_columns = list(columns) if columns is not None else list(df.columns)
    clean = df.copy()
    rows = []

    for col in target_columns:
        as_text = clean[col].astype(str).str.strip().str.upper()
        is_error = as_text.isin(EXCEL_ERROR_LITERALS)
        n_errors = int(is_error.sum())
        if n_errors:
            rows.append({
                "columna": col,
                "n_errores": n_errors,
                "literales": ", ".join(sorted(as_text[is_error].unique())),
            })
            clean.loc[is_error, col] = np.nan

    report = pd.DataFrame(rows, columns=["columna", "n_errores", "literales"])
    return clean, report


def excel_serial_to_datetime(values, system: int = 1900) -> pd.Series:
    """Convierte números de serie de Excel a fechas reales.

    Una columna de fechas formateada como número (o leída de una celda cuyo
    formato se perdió) llega como `45292.0`, no como `2024-01-01`. La conversión
    tiene dos trampas históricas reales:

    - Excel arrastra desde Lotus 1-2-3 el bug de considerar 1900 año bisiesto:
      existe un día 60 (`1900-02-29`) que nunca existió. Por eso el origen para
      series >= 61 es `1899-12-30` y no `1899-12-31`, y la serie 60 se devuelve
      como `NaT` en vez de inventar una fecha.
    - Excel para Mac usó históricamente el sistema 1904 (origen `1904-01-01`);
      un archivo guardado con ese sistema y leído con el otro se desfasa exactos
      1462 días (4 años y 1 día), un error que pasa desapercibido porque las
      fechas resultantes siguen siendo fechas plausibles.
    """
    serial = pd.to_numeric(pd.Series(values), errors="coerce")

    if system == 1904:
        return pd.to_datetime(serial, unit="D", origin="1904-01-01")

    modern = pd.to_datetime(serial.where(serial >= 61), unit="D", origin="1899-12-30")
    pre_bug = pd.to_datetime(serial.where(serial.between(1, 59)), unit="D", origin="1899-12-31")
    return modern.fillna(pre_bug)


def normalize_excel_whitespace(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Quita caracteres invisibles (espacio duro, zero-width, BOM), colapsa espacios
    repetidos y convierte a `NaN` las celdas que quedan vacías.

    Distinto de `text_cleaning.strip_whitespace`, que asume que el separador es
    un espacio normal: el texto que viene de Excel arrastra `\\xa0` (Alt+0160)
    cada vez que alguien pegó desde una página web, y `"Codelco\\xa0"` y
    `"Codelco"` son dos claves distintas en cualquier `groupby` o `merge`, sin
    que la diferencia se vea en pantalla ni en un `print`.
    """
    target_columns = columns if columns is not None else [
        c for c in df.columns if df[c].dtype == object
    ]
    clean = df.copy()
    pattern = f"[{re.escape(INVISIBLE_CHARS)}]"

    for col in target_columns:
        as_text = clean[col].astype(str).str.replace(pattern, " ", regex=True)
        as_text = as_text.str.replace(r"\s+", " ", regex=True).str.strip()
        clean[col] = as_text.where(clean[col].notna() & (as_text != ""), np.nan)
    return clean


def profile_cell_types(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Perfila el tipo *real* de cada celda por columna: cuántas son número, texto,
    fecha, booleano o nulo, y si la columna mezcla tipos.

    Este es el diagnóstico del triángulo verde de Excel ("número almacenado como
    texto"). Una columna con 4.998 floats y 2 strings se lee como `object`, y
    `.mean()` sobre ella falla o -- según la operación -- devuelve un resultado
    plausible pero incorrecto. Saber que son exactamente 2 celdas, y no 3.000,
    decide si corresponde parsear o devolver el archivo al cliente.
    """
    target_columns = list(columns) if columns is not None else list(df.columns)
    rows = []

    for col in target_columns:
        series = df[col]
        counts = {"numero": 0, "texto": 0, "fecha": 0, "booleano": 0, "otro": 0}

        for value in series.dropna():
            if isinstance(value, (bool, np.bool_)):
                counts["booleano"] += 1
            elif isinstance(value, (int, float, np.integer, np.floating)):
                counts["numero"] += 1
            elif isinstance(value, str):
                counts["texto"] += 1
            elif isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
                counts["fecha"] += 1
            else:
                counts["otro"] += 1

        present = {k: v for k, v in counts.items() if v}
        rows.append({
            "columna": col,
            "dtype_pandas": str(series.dtype),
            "n_nulos": int(series.isna().sum()),
            **{f"n_{k}": v for k, v in counts.items()},
            "tipo_dominante": max(present, key=present.get) if present else "vacio",
            "mixta": len(present) > 1,
        })

    return pd.DataFrame(rows)
