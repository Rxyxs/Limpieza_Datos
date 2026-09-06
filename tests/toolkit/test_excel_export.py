"""Pruebas unitarias para src/toolkit/excel_export.py.

Cada prueba escribe un .xlsx real en `tmp_path` y lo vuelve a abrir con openpyxl
o pandas: se verifica el archivo producido, no las llamadas hechas para producirlo.
"""
import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from src.toolkit.excel_export import (
    build_data_dictionary,
    safe_sheet_name,
    split_to_sheets,
    write_analysis_workbook,
)


@pytest.fixture
def panel():
    return pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        "empresa": ["Codelco", "Escondida", "Codelco", "Escondida"],
        "produccion_ton": [140.5, 98.2, np.nan, 101.0],
    })


def test_write_analysis_workbook_roundtrips_the_data(tmp_path, panel):
    path = write_analysis_workbook(tmp_path / "entregable.xlsx", {"Panel": panel})

    back = pd.read_excel(path, sheet_name="Panel")
    assert list(back.columns) == ["fecha", "empresa", "produccion_ton"]
    assert len(back) == 4
    assert back["produccion_ton"].isna().sum() == 1
    assert back["fecha"].iloc[0] == pd.Timestamp("2024-01-01")


def test_write_analysis_workbook_applies_presentation_formatting(tmp_path, panel):
    path = write_analysis_workbook(
        tmp_path / "entregable.xlsx", {"Panel": panel},
        number_formats={"produccion_ton": "#,##0.0"},
    )
    ws = load_workbook(path)["Panel"]

    assert ws.freeze_panes == "A2"
    assert ws.auto_filter.ref == ws.dimensions
    assert ws["A1"].font.bold is True
    assert ws["C2"].number_format == "#,##0.0"
    assert ws["A2"].number_format == "yyyy-mm-dd"  # default para columnas de fecha
    assert ws.column_dimensions["C"].width >= len("produccion_ton")


def test_write_analysis_workbook_appends_data_dictionary_sheet(tmp_path, panel):
    path = write_analysis_workbook(tmp_path / "entregable.xlsx", {"Panel": panel})

    sheets = load_workbook(path).sheetnames
    assert sheets == ["Panel", "Diccionario"]

    diccionario = pd.read_excel(path, sheet_name="Diccionario").set_index("columna")
    assert diccionario.loc["produccion_ton", "pct_faltante"] == 25.0
    assert diccionario.loc["produccion_ton", "n_no_nulos"] == 3
    assert diccionario.loc["empresa", "n_unicos"] == 2


def test_write_analysis_workbook_can_omit_the_dictionary(tmp_path, panel):
    path = write_analysis_workbook(
        tmp_path / "entregable.xlsx", {"Panel": panel}, data_dictionary=False,
    )
    assert load_workbook(path).sheetnames == ["Panel"]


def test_build_data_dictionary_reports_range_only_for_ordered_columns(panel):
    diccionario = build_data_dictionary({"Panel": panel}).set_index("columna")

    assert diccionario.loc["produccion_ton", "minimo"] == 98.2
    assert diccionario.loc["produccion_ton", "maximo"] == 140.5
    assert diccionario.loc["fecha", "minimo"] == pd.Timestamp("2024-01-01")
    assert diccionario.loc["empresa", "minimo"] is None
    assert diccionario.loc["empresa", "ejemplo"] == "Codelco"


def test_safe_sheet_name_truncates_and_resolves_collisions():
    taken = set()
    largo = "Produccion mensual por empresa 2014-2026"

    first = safe_sheet_name(largo, taken)
    second = safe_sheet_name(largo, taken)

    assert len(first) <= 31 and len(second) <= 31
    assert first != second
    assert safe_sheet_name("Region/Zona: Norte[1]") == "Region-Zona- Norte-1-"


def test_split_to_sheets_creates_one_sheet_per_group(panel):
    sheets = split_to_sheets(panel, by="empresa")

    assert set(sheets) == {"Codelco", "Escondida"}
    assert "empresa" not in sheets["Codelco"].columns
    assert len(sheets["Codelco"]) == 2


def test_split_to_sheets_refuses_high_cardinality_columns(panel):
    with pytest.raises(ValueError, match="hojas"):
        split_to_sheets(panel, by="fecha", max_sheets=3)


def test_split_to_sheets_feeds_write_analysis_workbook(tmp_path, panel):
    path = write_analysis_workbook(
        tmp_path / "por_empresa.xlsx", split_to_sheets(panel, by="empresa"),
    )
    assert load_workbook(path).sheetnames == ["Codelco", "Escondida", "Diccionario"]
