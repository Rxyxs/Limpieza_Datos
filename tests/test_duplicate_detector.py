"""Pruebas unitarias para src/cleaners/duplicate_detector.py."""
import pandas as pd

from src.cleaners.duplicate_detector import find_near_duplicate_tickets


def test_finds_a_near_duplicate_pair_with_typo_company_name():
    df = pd.DataFrame({
        "ticket_id": ["TCK-1", "TCK-2"],
        "customer_email": ["user1@corp.com", "user1@corp.com"],
        "company_name": ["Nova Systems", "Nova Systms"],  # typo introducido en el reenvio
        "category": ["Billing", "Billing"],
        "created_at": pd.to_datetime(["2024-01-01T10:00:00Z", "2024-01-01T10:02:00Z"]),
    })
    result = find_near_duplicate_tickets(df)

    assert len(result) == 1
    assert set(result.iloc[0][["ticket_id_a", "ticket_id_b"]]) == {"TCK-1", "TCK-2"}


def test_does_not_flag_different_customers_as_duplicates():
    df = pd.DataFrame({
        "ticket_id": ["TCK-1", "TCK-2"],
        "customer_email": ["user1@corp.com", "user2@corp.com"],
        "company_name": ["Nova Systems", "Nova Systems"],
        "category": ["Billing", "Billing"],
        "created_at": pd.to_datetime(["2024-01-01T10:00:00Z", "2024-01-01T10:02:00Z"]),
    })
    result = find_near_duplicate_tickets(df)
    assert result.empty


def test_does_not_flag_tickets_outside_the_time_window():
    df = pd.DataFrame({
        "ticket_id": ["TCK-1", "TCK-2"],
        "customer_email": ["user1@corp.com", "user1@corp.com"],
        "company_name": ["Nova Systems", "Nova Systems"],
        "category": ["Billing", "Billing"],
        "created_at": pd.to_datetime(["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"]),
    })
    result = find_near_duplicate_tickets(df, time_window_minutes=10)
    assert result.empty


def test_does_not_flag_dissimilar_company_names_even_if_close_in_time():
    df = pd.DataFrame({
        "ticket_id": ["TCK-1", "TCK-2"],
        "customer_email": ["user1@corp.com", "user1@corp.com"],
        "company_name": ["Nova Systems", "Quanta Networks"],
        "category": ["Billing", "Billing"],
        "created_at": pd.to_datetime(["2024-01-01T10:00:00Z", "2024-01-01T10:01:00Z"]),
    })
    result = find_near_duplicate_tickets(df)
    assert result.empty


def test_raises_on_missing_required_columns():
    df = pd.DataFrame({"ticket_id": ["TCK-1"]})
    try:
        find_near_duplicate_tickets(df)
        assert False, "deberia haber lanzado KeyError"
    except KeyError:
        pass
