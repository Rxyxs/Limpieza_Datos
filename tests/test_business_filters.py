"""Pruebas unitarias para src/filters/business_filters.py."""
import numpy as np
import pandas as pd

from src.filters.business_filters import (
    apply_ml_scope_filters,
    filter_by_date_range,
    filter_positive_cost,
    filter_resolved_tickets,
)


def test_filter_resolved_tickets_drops_open_tickets():
    df = pd.DataFrame({"resolved_at": ["2024-01-01", None, "2024-01-02"]})
    result = filter_resolved_tickets(df)
    assert len(result) == 2
    assert result["resolved_at"].notna().all()


def test_filter_by_date_range_keeps_only_inside_the_window():
    df = pd.DataFrame({"created_at": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"], utc=True)})
    result = filter_by_date_range(df, "created_at", start="2024-03-01", end="2024-09-01")
    assert len(result) == 1
    assert result.iloc[0]["created_at"] == pd.Timestamp("2024-06-01", tz="UTC")


def test_filter_positive_cost_drops_refunds_and_zero():
    df = pd.DataFrame({"cost": [100.0, -20.0, 0.0, 50.0]})
    result = filter_positive_cost(df)
    assert result["cost"].tolist() == [100.0, 50.0]


def test_apply_ml_scope_filters_chains_resolved_and_positive_cost():
    df = pd.DataFrame({
        "resolved_at": ["2024-01-01", None, "2024-01-02", "2024-01-03"],
        "cost": [100.0, 50.0, -10.0, 0.0],
    })
    result = apply_ml_scope_filters(df)
    assert len(result) == 1
    assert result.iloc[0]["cost"] == 100.0
