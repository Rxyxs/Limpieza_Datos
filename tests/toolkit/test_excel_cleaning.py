"""Pruebas unitarias para src/toolkit/excel_cleaning.py."""
import numpy as np
import pandas as pd

from src.toolkit.excel_cleaning import (
    detect_header_row,
    drop_subtotal_rows,
    flatten_multirow_header,
    strip_footnote_markers,
    wide_years_to_long,
)


def test_flatten_multirow_header_combines_rows_and_ffills_merged_cells():
    # Simula una celda combinada: "Produccion" solo aparece en la primera columna
    # del grupo, NaN en las siguientes -- exactamente como openpyxl lee un merge.
    raw = pd.DataFrame([
        ["Produccion", np.nan, "Precio", np.nan],
        ["Empresa A", "Empresa B", "Empresa A", "Empresa B"],
        [100, 200, 3.5, 3.6],
        [110, 210, 3.6, 3.7],
    ])
    result = flatten_multirow_header(raw, header_rows=2)

    assert list(result.columns) == [
        "Produccion / Empresa A", "Produccion / Empresa B", "Precio / Empresa A", "Precio / Empresa B",
    ]
    assert len(result) == 2
    assert result.iloc[0, 0] == 100


def test_detect_header_row_finds_row_past_title_lines():
    raw = pd.DataFrame([
        ["Anuario Estadistico 2025", np.nan, np.nan],
        ["Publicado por COCHILCO", np.nan, np.nan],
        ["Empresa", "Anio", "Produccion"],
        ["Codelco", 2024, 1500000],
    ])
    row = detect_header_row(raw, expected_tokens=["empresa", "anio", "produccion"])
    assert row == 2


def test_strip_footnote_markers_removes_common_patterns():
    assert strip_footnote_markers("1234 (p)") == "1234"
    assert strip_footnote_markers("1234*") == "1234"
    assert strip_footnote_markers("1234 1/") == "1234"
    assert strip_footnote_markers("1234") == "1234"


def test_wide_years_to_long_melts_year_columns_only():
    df = pd.DataFrame({
        "empresa": ["Codelco", "Escondida"],
        "2020": [100, 200],
        "2021": [110, 210],
        "unidad": ["ton", "ton"],  # no deberia tratarse como año
    })
    long_df = wide_years_to_long(df, id_vars=["empresa", "unidad"])

    assert set(long_df["periodo"]) == {2020, 2021}
    assert long_df["periodo"].dtype == int
    assert len(long_df) == 4


def test_drop_subtotal_rows_removes_matching_labels():
    df = pd.DataFrame({"empresa": ["Codelco", "Escondida", "Total Pais"], "produccion": [100, 200, 300]})
    result, n_removed = drop_subtotal_rows(df, "empresa", subtotal_markers=["total"])
    assert n_removed == 1
    assert "Total Pais" not in result["empresa"].values
