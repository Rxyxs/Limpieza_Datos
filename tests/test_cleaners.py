"""Pruebas unitarias para los módulos de limpieza (src/cleaners)."""
import numpy as np
import pandas as pd

from src.cleaners.datetime_cleaner import clean_datetime_column, parse_to_utc, to_iso8601
from src.cleaners.json_normalizer import normalize_json_column
from src.cleaners.missing_data_imputer import impute_numeric_by_category, interpolate_numeric_column
from src.cleaners.string_cleaner import normalize_case, parse_currency, strip_whitespace, unify_similar_names


# ---------------------------------------------------------------------------
# json_normalizer
# ---------------------------------------------------------------------------

def test_normalize_json_column_flattens_nested_dict():
    df = pd.DataFrame({
        "id": [1],
        "user_metadata": ['{"os": "Windows", "browser": {"name": "Chrome", "version": "120"}}'],
    })
    result = normalize_json_column(df, column="user_metadata")

    assert "user_metadata" not in result.columns
    assert result.loc[0, "user_metadata_os"] == "Windows"
    assert result.loc[0, "user_metadata_browser_name"] == "Chrome"


def test_normalize_json_column_handles_null_like_values():
    df = pd.DataFrame({"id": [1, 2, 3], "user_metadata": ["null", "", np.nan]})
    result = normalize_json_column(df, column="user_metadata")

    assert "user_metadata" not in result.columns
    assert len(result) == 3


def test_normalize_json_column_serializes_lists_as_text():
    df = pd.DataFrame({"id": [1], "user_metadata": ['{"tags": ["vip", "beta"]}']})
    result = normalize_json_column(df, column="user_metadata")

    assert result.loc[0, "user_metadata_tags"] == "vip, beta"


# ---------------------------------------------------------------------------
# datetime_cleaner
# ---------------------------------------------------------------------------

def test_parse_to_utc_handles_named_timezone_abbreviation():
    parsed = parse_to_utc("27/06/2025 09:10 EST")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.hour == 14  # 09:10 EST (-05:00) -> 14:10 UTC


def test_parse_to_utc_handles_iso_offset_format():
    parsed = parse_to_utc("2025-03-11T19:27:00+02:00")
    assert parsed is not None
    assert parsed.hour == 17  # 19:27+02:00 -> 17:27 UTC


def test_parse_to_utc_returns_none_for_null_like_values():
    assert parse_to_utc("N/A") is None
    assert parse_to_utc("") is None
    assert parse_to_utc(None) is None
    assert parse_to_utc(np.nan) is None


def test_clean_datetime_column_and_to_iso8601_roundtrip():
    df = pd.DataFrame({"created_at": ["2024-01-01 12:00:00", "not a date"]})
    df = clean_datetime_column(df, "created_at")
    assert pd.notna(df.loc[0, "created_at"])
    assert pd.isna(df.loc[1, "created_at"])

    df = to_iso8601(df, "created_at")
    assert df.loc[0, "created_at"] == "2024-01-01T12:00:00+00:00"
    assert pd.isna(df.loc[1, "created_at"])  # pandas almacena el None devuelto como NaN


# ---------------------------------------------------------------------------
# string_cleaner
# ---------------------------------------------------------------------------

def test_strip_whitespace_collapses_and_trims_spaces():
    df = pd.DataFrame({"name": ["  Nova   Systems  ", np.nan]})
    result = strip_whitespace(df, ["name"])
    assert result.loc[0, "name"] == "Nova Systems"
    assert pd.isna(result.loc[1, "name"])


def test_parse_currency_handles_us_format():
    assert parse_currency("$1,200.50") == 1200.50


def test_parse_currency_handles_european_format():
    assert parse_currency("1200,50 €") == 1200.50


def test_parse_currency_handles_currency_code_suffix():
    assert parse_currency("1,050.75 USD") == 1050.75


def test_parse_currency_handles_negative_refund():
    assert parse_currency("-$20.00") == -20.00


def test_parse_currency_returns_none_for_null_like_values():
    assert parse_currency("N/A") is None
    assert parse_currency("") is None
    assert parse_currency(np.nan) is None


def test_normalize_case_title_cases_and_preserves_nulls():
    df = pd.DataFrame({"priority": ["LOW", "high", np.nan]})
    result = normalize_case(df, ["priority"])
    assert result.loc[0, "priority"] == "Low"
    assert result.loc[1, "priority"] == "High"
    assert pd.isna(result.loc[2, "priority"])


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


# ---------------------------------------------------------------------------
# missing_data_imputer
# ---------------------------------------------------------------------------

def test_impute_numeric_by_category_uses_conditional_mean():
    df = pd.DataFrame({
        "category": ["Billing", "Billing", "Billing", "Technical"],
        "cost": [100.0, 200.0, np.nan, np.nan],
    })
    result = impute_numeric_by_category(df, value_column="cost", category_column="category")

    assert result.loc[2, "cost"] == 150.0  # media de Billing (100, 200)
    assert result.loc[3, "cost"] == result["cost"].iloc[:3].mean()  # sin datos en Technical -> media global previa


def test_interpolate_numeric_column_fills_gap_linearly():
    df = pd.DataFrame({
        "created_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "response_time_hours": [10.0, np.nan, 30.0],
    })
    result = interpolate_numeric_column(df, column="response_time_hours", sort_by="created_at")
    assert result.loc[1, "response_time_hours"] == 20.0
