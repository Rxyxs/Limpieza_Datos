"""Pruebas unitarias para src/toolkit/text_cleaning.py."""
import numpy as np
import pandas as pd

from src.toolkit.text_cleaning import normalize_case, parse_currency, strip_whitespace, unify_similar_names


def test_strip_whitespace_collapses_and_trims_spaces():
    df = pd.DataFrame({"name": ["  Nova   Systems  ", np.nan]})
    result = strip_whitespace(df, ["name"])
    assert result.loc[0, "name"] == "Nova Systems"
    assert pd.isna(result.loc[1, "name"])


def test_parse_currency_handles_us_format():
    assert parse_currency("$1,200.50") == 1200.50


def test_parse_currency_handles_latam_format():
    assert parse_currency("1.200,50") == 1200.50


def test_parse_currency_handles_negative_parentheses():
    assert parse_currency("(20.00)") == -20.00


def test_parse_currency_strips_footnote_marker():
    assert parse_currency("1234 (p)") == 1234.0


def test_parse_currency_returns_none_for_null_like_values():
    assert parse_currency("N/A") is None
    assert parse_currency("") is None
    assert parse_currency(np.nan) is None


def test_normalize_case_title_cases_and_preserves_nulls():
    df = pd.DataFrame({"category": ["LOW", "high", np.nan]})
    result = normalize_case(df, ["category"])
    assert result.loc[0, "category"] == "Low"
    assert pd.isna(result.loc[2, "category"])


def test_unify_similar_names_clusters_typo_variants():
    series = pd.Series(["Nova Systems", "NOVA SYSTEMS", "Nova Systems Inc.", "Quanta Networks"])
    result = unify_similar_names(series, threshold=80)
    assert result.nunique() == 2
    assert result[0] == result[1] == result[2]
    assert result[3] == "Quanta Networks"


def test_unify_similar_names_preserves_missing_values():
    series = pd.Series(["Nova Systems", np.nan, "  "])
    result = unify_similar_names(series)
    assert pd.isna(result[1])
    assert pd.isna(result[2])
