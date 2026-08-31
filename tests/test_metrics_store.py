"""Pruebas unitarias para src/models/metrics_store.py."""
import numpy as np
import pandas as pd
import pytest

from src.models.metrics_store import persist_sla_comparison_metrics, read_latest_run


def _fake_outcome() -> dict:
    return {
        "n_train": 100,
        "n_test": 25,
        "breach_rate_test": 0.4,
        "results_by_model": {
            "relu": {"accuracy": 0.80, "precision": 0.78, "recall": 0.70, "f1": 0.74},
            "tanh": {"accuracy": 0.82, "precision": 0.80, "recall": 0.72, "f1": 0.76},
            "gradient_boosting": {"accuracy": 0.85, "precision": 0.83, "recall": 0.79, "f1": 0.81},
        },
    }


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_metrics.duckdb"


def test_persist_creates_db_file_and_returns_its_path(db_path):
    outcome = _fake_outcome()
    result_path = persist_sla_comparison_metrics(outcome, db_path=db_path)

    assert result_path == db_path
    assert db_path.exists()


def test_persist_writes_one_row_per_model(db_path):
    outcome = _fake_outcome()
    persist_sla_comparison_metrics(outcome, db_path=db_path)

    latest = read_latest_run(db_path)
    assert len(latest) == 3
    assert set(latest["model_name"]) == {"relu", "tanh", "gradient_boosting"}


def test_persist_appends_across_multiple_runs_without_overwriting(db_path):
    outcome = _fake_outcome()
    persist_sla_comparison_metrics(outcome, db_path=db_path)
    persist_sla_comparison_metrics(outcome, db_path=db_path)

    import duckdb
    con = duckdb.connect(str(db_path))
    total_rows = con.execute("SELECT count(*) FROM sla_model_metrics").fetchone()[0]
    con.close()

    assert total_rows == 6  # 3 modelos x 2 corridas, ninguna sobreescribe a la otra


def test_read_latest_run_is_sorted_by_f1_descending(db_path):
    outcome = _fake_outcome()
    persist_sla_comparison_metrics(outcome, db_path=db_path)

    latest = read_latest_run(db_path)
    assert list(latest["f1"]) == sorted(latest["f1"], reverse=True)
    assert latest.iloc[0]["model_name"] == "gradient_boosting"


def test_persist_raises_without_results_key(db_path):
    with pytest.raises(ValueError):
        persist_sla_comparison_metrics({"n_train": 1, "n_test": 1, "breach_rate_test": 0.1}, db_path=db_path)
