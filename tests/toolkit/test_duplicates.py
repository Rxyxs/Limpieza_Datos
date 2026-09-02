"""Pruebas unitarias para src/toolkit/duplicates.py."""
import pandas as pd

from src.toolkit.duplicates import exact_duplicate_report, find_near_duplicate_rows


def test_finds_a_near_duplicate_pair_with_typo():
    df = pd.DataFrame({
        "id": ["A-1", "A-2"],
        "cliente": ["user1@corp.com", "user1@corp.com"],
        "empresa": ["Nova Systems", "Nova Systms"],
        "categoria": ["Billing", "Billing"],
        "fecha": pd.to_datetime(["2024-01-01T10:00:00Z", "2024-01-01T10:02:00Z"]),
    })
    result = find_near_duplicate_rows(
        df, key_column="cliente", fuzzy_column="empresa", group_column="categoria",
        timestamp_column="fecha", id_column="id",
    )
    assert len(result) == 1
    assert set(result.iloc[0][["id_a", "id_b"]]) == {"A-1", "A-2"}


def test_does_not_flag_different_keys_as_duplicates():
    df = pd.DataFrame({
        "id": ["A-1", "A-2"],
        "cliente": ["user1@corp.com", "user2@corp.com"],
        "empresa": ["Nova Systems", "Nova Systems"],
    })
    result = find_near_duplicate_rows(df, key_column="cliente", fuzzy_column="empresa", id_column="id")
    assert result.empty


def test_does_not_flag_tickets_outside_the_time_window():
    df = pd.DataFrame({
        "id": ["A-1", "A-2"],
        "cliente": ["user1@corp.com", "user1@corp.com"],
        "empresa": ["Nova Systems", "Nova Systems"],
        "fecha": pd.to_datetime(["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"]),
    })
    result = find_near_duplicate_rows(
        df, key_column="cliente", fuzzy_column="empresa", timestamp_column="fecha", time_window_minutes=10,
    )
    assert result.empty


def test_raises_on_missing_required_columns():
    df = pd.DataFrame({"id": ["A-1"]})
    try:
        find_near_duplicate_rows(df, key_column="cliente", fuzzy_column="empresa")
        assert False, "deberia haber lanzado KeyError"
    except KeyError:
        pass


def test_exact_duplicate_report_counts_removed_rows():
    df = pd.DataFrame({"id": ["A-1", "A-1", "A-2"], "valor": [1, 1, 2]})
    deduped, n_removed = exact_duplicate_report(df, subset=["id"])
    assert len(deduped) == 2
    assert n_removed == 1
