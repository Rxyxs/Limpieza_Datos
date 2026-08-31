"""Pruebas unitarias para src/models/sla_breach_classifier.py."""
import numpy as np
import pandas as pd

from src.models.sla_breach_classifier import (
    SLA_THRESHOLD_HOURS,
    prepare_classification_frame,
    train_and_compare_activations,
    train_and_compare_all_models,
)


def _synthetic_clean_dataset(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    categories = ["Billing", "Technical", "Account", "Bug Report", "Feature Request"]
    priorities = ["low", "medium", "high", "critical"]
    category = rng.choice(categories, n)
    priority = rng.choice(priorities, n)

    base_hours = pd.Series(category).map(
        {"Billing": 8.0, "Account": 12.0, "Technical": 30.0, "Bug Report": 48.0, "Feature Request": 60.0}
    ).to_numpy()
    priority_mult = pd.Series(priority).map(
        {"critical": 0.3, "high": 0.6, "medium": 1.0, "low": 1.8}
    ).to_numpy()
    response_time_hours = base_hours * priority_mult * rng.lognormal(0, 0.3, n)

    created = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "ticket_id": [f"TCK-{i}" for i in range(n)],
        "company_name": ["Nova Systems"] * n,
        "customer_email": [f"user{i}@corp.com" for i in range(n)],
        "created_at": created,
        "resolved_at": created,  # basta con que no sea nulo para pasar el filtro de alcance
        "priority": priority,
        "status": rng.choice(["Open", "Pending", "Resolved", "Closed"], n),
        "category": category,
        "agent_name": ["A. Rivera"] * n,
        "cost": rng.uniform(5, 5000, n),
        "response_time_hours": response_time_hours,
    })


def test_prepare_classification_frame_binarizes_correctly():
    df = _synthetic_clean_dataset()
    X, y = prepare_classification_frame(df)

    assert set(y.unique()) <= {0, 1}
    assert "response_time_hours" not in X.columns  # el target no debe filtrarse a las features


def test_train_and_compare_activations_returns_both_activations():
    df = _synthetic_clean_dataset()
    outcome = train_and_compare_activations(df)

    assert set(outcome["results_by_activation"].keys()) == {"relu", "tanh"}
    for activation, result in outcome["results_by_activation"].items():
        assert 0.0 <= result["accuracy"] <= 1.0
        assert len(result["predictions"]) == outcome["n_test"]


def test_breach_rate_reflects_the_sla_threshold():
    df = _synthetic_clean_dataset()
    _, y = prepare_classification_frame(df)
    expected_rate = (df["response_time_hours"] > SLA_THRESHOLD_HOURS).mean()
    # Redondeado por el filtrado de alcance (resolved+costo positivo), deberia ser cercano.
    assert abs(y.mean() - expected_rate) < 0.05


def test_train_and_compare_all_models_adds_gradient_boosting_as_third_arm():
    df = _synthetic_clean_dataset()
    outcome = train_and_compare_all_models(df)

    assert set(outcome["results_by_model"].keys()) == {"relu", "tanh", "gradient_boosting"}
    for model_name, result in outcome["results_by_model"].items():
        assert 0.0 <= result["accuracy"] <= 1.0
        assert len(result["predictions"]) == outcome["n_test"]


def test_train_and_compare_activations_contract_unchanged():
    # La función original no debe verse afectada por agregar el tercer modelo.
    df = _synthetic_clean_dataset()
    outcome = train_and_compare_activations(df)
    assert set(outcome["results_by_activation"].keys()) == {"relu", "tanh"}
