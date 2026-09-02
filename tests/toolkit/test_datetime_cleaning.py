"""Pruebas unitarias para src/toolkit/datetime_cleaning.py."""
import numpy as np
import pandas as pd

from src.toolkit.datetime_cleaning import clean_datetime_column, parse_to_utc, reindex_to_full_calendar, to_iso8601


def test_parse_to_utc_handles_named_timezone_abbreviation():
    parsed = parse_to_utc("27/06/2025 09:10 EST")
    assert parsed is not None
    assert parsed.hour == 14  # 09:10 EST (-05:00) -> 14:10 UTC


def test_parse_to_utc_handles_iso_offset_format():
    parsed = parse_to_utc("2025-03-11T19:27:00+02:00")
    assert parsed.hour == 17


def test_parse_to_utc_returns_none_for_null_like_values():
    assert parse_to_utc("N/A") is None
    assert parse_to_utc("") is None
    assert parse_to_utc(None) is None
    assert parse_to_utc(np.nan) is None


def test_clean_datetime_column_and_to_iso8601_roundtrip():
    df = pd.DataFrame({"fecha": ["2024-01-01 12:00:00", "not a date"]})
    df = clean_datetime_column(df, "fecha")
    assert pd.notna(df.loc[0, "fecha"])
    assert pd.isna(df.loc[1, "fecha"])

    df = to_iso8601(df, "fecha")
    assert df.loc[0, "fecha"] == "2024-01-01T12:00:00+00:00"


def test_reindex_to_full_calendar_fills_weekend_gaps_with_ffill():
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-05", "2024-01-08"]),  # viernes -> lunes, salta fin de semana
        "valor": [100.0, 110.0],
    })
    result = reindex_to_full_calendar(df, date_column="fecha", freq="D", fill_method="ffill")

    assert len(result) == 4  # vie, sab, dom, lun
    saturday = result[result["fecha"] == pd.Timestamp("2024-01-06")]
    assert saturday["valor"].iloc[0] == 100.0  # sostiene el ultimo valor real (viernes)
