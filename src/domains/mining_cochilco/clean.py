"""Limpieza del panel minero: producción mensual de cobre de mina por empresa,
publicada por COCHILCO en un archivo `.xlsx` que replica el layout visual de un
reporte institucional, no un export tabular. Tres tipos de mezcla real
encontrados y tratados acá (ninguno es "un valor faltante": los tres son
problemas de ESTRUCTURA de fila/columna, no de dato ausente):

1. Filas que no son observaciones mensuales: 6 filas de título/institución
   antes del encabezado real (fila 7), filas de resumen ANUAL intercaladas
   entre los meses (columna A = un año "pelado" como texto, ej. "2024", no
   una fecha) y una fila de cita ("Fuente / Source: ...") al final. Se
   detectan por el TIPO del valor en la columna de fecha: openpyxl
   (`data_only=True`, que es lo que usa `pandas.read_excel` internamente)
   entrega un objeto `datetime` real para una celda-fecha de Excel, pero un
   `str` para una celda de texto como "2024" -- una fila solo se acepta como
   mensual si su columna A es efectivamente un objeto fecha, NO si el texto
   simplemente *parsea* como fecha (`pd.to_datetime("2024")` da
   `2024-01-01`, que colisionaría silenciosamente con la fila real de enero
   2024 si se filtrara solo por parseabilidad -- un filtro más ingenuo se
   habría comido esa fila real).
2. Filas plantilla de meses futuros aún no publicados: COCHILCO deja filas
   pre-creadas para el resto del año en curso con `TOTAL CHILE = 0` (y el
   resto de las columnas en blanco) hasta que el mes efectivamente se
   publique. Se descartan igual que las filas de resumen anual.
3. Columnas subtotal DISFRAZADAS: además de las obvias por nombre
   ("Total Codelco", "TOTAL CHILE"), el archivo tiene subtotales que NO
   siguen el patrón "Total *" -- "Chuqui y R.Tomic" (= Chuquicamata +
   Radomiro Tomic), "Angloamerican Sur" (= Los Bronces + El Soldado) y
   "Capstone Copper" (= Mantos Blancos + Mantoverde). Se confirmaron por
   identidad numérica EXACTA fila a fila (no por el nombre de la columna) --
   sumar "todas las columnas" ingenuamente cuenta cada una de estas 4
   combinaciones dos veces, e infla el total nacional calculado ~2-3% por
   sobre el publicado (`_validate_national_total` lo detecta y frena el
   pipeline si vuelve a pasar con una edición futura del archivo).

Missingness genuina (no un error, ver mensaje del proyecto sobre no imputar
lo que no está realmente ausente): dentro de las filas mensuales reales NO
hay celdas en blanco -- una faena que aún no iniciaba operaciones, o que ya
cerró, se reporta como `0.0` explícito, no como nulo. Winsorizar/imputar esos
ceros como si fueran atípicos trataría un hecho estructural (la faena
literalmente no producía ese mes) como si fuera un dato faltante al azar, así
que se excluyen de la detección de outliers.

Se trabaja con el TOTAL NACIONAL (suma de las 38 columnas reales, ya
validada contra `TOTAL CHILE` publicado) como serie principal para modelar,
en vez de imputar los huecos estructurales de cada faena individualmente --
una faena que no existía en 2014 no tiene un valor "faltante" que estimar, el
total nacional es la serie que sí está completa y bien definida en todo el
período.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src.toolkit.excel_cleaning import detect_header_row
from src.toolkit.missing_data import missingness_report
from src.toolkit.outliers import winsorize_column_by_group
from src.toolkit.text_cleaning import normalize_case, strip_whitespace, unify_similar_names

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "mining"
PROCESSED_DIR = ROOT / "data" / "processed" / "mining"
RAW_FILENAME = "cochilco_produccion_mensual.xlsx"
SHEET_NAME = "Prod.Cu-Mina x Faena (2014+)"

# Columnas subtotal: cada una es una SUMA de otras columnas ya presentes en el
# archivo (ver punto 3 del docstring del módulo). Se excluyen de cualquier
# suma "todas las columnas -> total nacional" para no contar dos veces.
SUBTOTAL_COLUMNS = ["Chuqui y R.Tomic", "Total Codelco", "Angloamerican Sur", "Capstone Copper"]
GRAND_TOTAL_COLUMN = "TOTAL CHILE"
DATE_COLUMN = "fecha"


def _load_raw_wide() -> tuple[pd.DataFrame, dict]:
    """Lee el Excel crudo y arma un DataFrame ancho (una columna por
    empresa/faena) con solo filas mensuales reales publicadas.
    """
    raw = pd.read_excel(RAW_DIR / RAW_FILENAME, sheet_name=SHEET_NAME, header=None)

    header_row_idx = detect_header_row(
        raw, expected_tokens=["mes-a", "chuqui", "escondida", "collahuasi", "total chile"]
    )
    headers = [str(v).strip() for v in raw.iloc[header_row_idx].tolist()]
    headers[0] = DATE_COLUMN

    body = raw.iloc[header_row_idx + 1:].reset_index(drop=True).copy()
    body.columns = headers
    body = body.loc[:, [c for c in body.columns if c and c != "None"]]

    # Mezcla real #1: solo se acepta una fila si su columna de fecha es un
    # objeto fecha real, no un string que *parsea* como fecha.
    is_real_month = body[DATE_COLUMN].apply(lambda v: isinstance(v, (dt.datetime, pd.Timestamp)))
    n_non_month_rows = int((~is_real_month).sum())
    body = body.loc[is_real_month].reset_index(drop=True)
    body[DATE_COLUMN] = pd.to_datetime(body[DATE_COLUMN])

    numeric_cols = [c for c in body.columns if c != DATE_COLUMN]
    body[numeric_cols] = body[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Mezcla real #2: filas plantilla de meses futuros aún no publicados
    # (TOTAL CHILE = 0, resto en blanco).
    is_published = body[GRAND_TOTAL_COLUMN].fillna(0) > 0
    n_unpublished_rows = int((~is_published).sum())
    body = body.loc[is_published].reset_index(drop=True)

    row_report = {
        "filas_no_mensuales_descartadas": n_non_month_rows,
        "filas_futuras_no_publicadas_descartadas": n_unpublished_rows,
    }
    return body, row_report


def _validate_national_total(df: pd.DataFrame, company_cols: list[str]) -> float:
    """Suma las columnas REALES (sin subtotales) y la compara, fila a fila,
    contra `TOTAL CHILE` publicado por COCHILCO. Devuelve la máxima
    desviación absoluta encontrada -- debería ser ~0 (ruido de punto
    flotante) si la lista de columnas subtotal está completa y correcta.
    """
    computed = df[company_cols].sum(axis=1)
    deviation = (computed - df[GRAND_TOTAL_COLUMN]).abs()
    return float(deviation.max())


def build_clean_panel() -> tuple[pd.DataFrame, dict]:
    """Devuelve `(panel_limpio, reporte)`: panel ancho, un registro por mes,
    con las 38 empresas/faenas reales, el total nacional derivado y
    validado, y el desglose Codelco / no-Codelco.
    """
    report: dict = {}
    wide, row_report = _load_raw_wide()
    report.update(row_report)
    report["n_filas_mensuales_reales"] = len(wide)

    wide.columns = [str(c).strip() for c in wide.columns]
    company_cols = [c for c in wide.columns if c not in SUBTOTAL_COLUMNS + [GRAND_TOTAL_COLUMN, DATE_COLUMN]]

    # Limpieza de texto sobre los nombres de empresa/faena: red de seguridad
    # defensiva, no una corrección de un problema observado -- con 38 nombres
    # genuinamente distintos en este archivo, `unify_similar_names` no
    # colapsa ninguno (se deja registrado explícitamente), pero blindaría el
    # pipeline si una futura edición del archivo trajera una variante
    # tipográfica de un nombre ya existente.
    canonical = unify_similar_names(pd.Series(company_cols), threshold=90)
    report["nombres_empresa_unificados"] = int((pd.Series(company_cols).values != canonical.values).sum())

    max_deviation = _validate_national_total(wide, company_cols)
    report["desviacion_max_total_nacional_vs_publicado"] = round(max_deviation, 6)
    if max_deviation > 1.0:
        raise ValueError(
            f"Suma de columnas reales se desvía del TOTAL CHILE publicado en {max_deviation:.3f} "
            "miles de T.M. -- probable columna subtotal disfrazada no listada en SUBTOTAL_COLUMNS "
            "(revisar cabeceras nuevas antes de confiar en el total nacional derivado)."
        )

    wide["total_nacional_miles_ton"] = wide[company_cols].sum(axis=1)
    wide["total_codelco_miles_ton"] = wide["Total Codelco"]
    wide["share_codelco"] = wide["total_codelco_miles_ton"] / wide["total_nacional_miles_ton"]

    before_missing = missingness_report(wide).set_index("columna")["pct_nulos"]

    # Missingness genuina, no un error (ver docstring del módulo): dentro de
    # las filas mensuales reales no hay celdas en blanco.
    report["nulos_en_columnas_de_empresa"] = int(wide[company_cols].isna().sum().sum())

    # Outliers: solo se evalúan meses con producción > 0 por empresa (un 0 es
    # la faena inactiva ese mes, no un valor atípico) y con IQR conservador
    # (k=3.0, no el 1.5 estándar) porque la producción de una faena real
    # tiene tendencia genuina (ley del mineral que declina, expansiones,
    # huelgas, mantenciones programadas) -- un k agresivo trataría esa
    # variación operacional real como si fuera un error de captura.
    long_df = wide.melt(id_vars=[DATE_COLUMN], value_vars=company_cols, var_name="empresa", value_name="produccion_miles_ton")
    long_df = strip_whitespace(long_df, ["empresa"])
    long_df = normalize_case(long_df, ["empresa"])
    active = long_df.loc[long_df["produccion_miles_ton"] > 0].reset_index(drop=True)
    _, n_outliers = winsorize_column_by_group(active, "produccion_miles_ton", "empresa", k=3.0)
    report["outliers_empresa_mes_detectados_no_recortados"] = n_outliers
    report["nota_outliers"] = (
        "Detectados pero NO recortados: dato oficial de un regulador, sin evidencia de error de "
        "captura (a diferencia de la UF corrupta de financial_bcch) -- recortarlos borraría ciclos "
        "operacionales reales (huelgas, mantenciones, rampas de inicio/cierre de faena)."
    )

    after_missing = missingness_report(wide).set_index("columna")["pct_nulos"]
    report["missingness_antes"] = before_missing
    report["missingness_despues"] = after_missing

    panel = wide.sort_values(DATE_COLUMN).reset_index(drop=True)
    report["n_filas_finales"] = len(panel)
    report["rango_fechas"] = f"{panel[DATE_COLUMN].min().date()} a {panel[DATE_COLUMN].max().date()}"
    report["total_nacional_min_mensual"] = round(panel["total_nacional_miles_ton"].min(), 1)
    report["total_nacional_max_mensual"] = round(panel["total_nacional_miles_ton"].max(), 1)
    report["total_nacional_medio_mensual"] = round(panel["total_nacional_miles_ton"].mean(), 1)

    return panel, report


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel, report = build_clean_panel()
    out_path = PROCESSED_DIR / "mining_panel_clean.csv"
    panel.to_csv(out_path, index=False)

    print(f"Panel limpio: {len(panel):,} filas ({report['rango_fechas']})")
    print(f"  -> {out_path}")
    for key, value in report.items():
        if isinstance(value, pd.Series):
            continue
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
