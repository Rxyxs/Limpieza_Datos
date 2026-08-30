"""Pruebas unitarias para src/features/feature_encoder.py."""
import numpy as np
import pandas as pd

from src.features.feature_encoder import build_ml_ready_dataset, encode_categorical_onehot, encode_priority_ordinal


def test_encode_priority_ordinal_respects_severity_order():
    df = pd.DataFrame({"priority": ["low", "critical", "medium", "high"]})
    result = encode_priority_ordinal(df)

    assert result.loc[0, "priority_encoded"] == 0
    assert result.loc[1, "priority_encoded"] == 3
    assert result.loc[2, "priority_encoded"] == 1
    assert result.loc[3, "priority_encoded"] == 2


def test_encode_priority_ordinal_is_case_insensitive():
    df = pd.DataFrame({"priority": ["LOW", "Critical"]})
    result = encode_priority_ordinal(df)
    assert result.loc[0, "priority_encoded"] == 0
    assert result.loc[1, "priority_encoded"] == 3


def test_encode_priority_ordinal_leaves_unknown_values_as_nan():
    df = pd.DataFrame({"priority": ["urgent"]})  # no es un nivel valido
    result = encode_priority_ordinal(df)
    assert pd.isna(result.loc[0, "priority_encoded"])


def test_encode_categorical_onehot_creates_one_column_per_category():
    df = pd.DataFrame({"category": ["Billing", "Technical"]})
    result = encode_categorical_onehot(df, columns=["category"])

    assert "category_Billing" in result.columns
    assert "category_Technical" in result.columns
    assert result.loc[0, "category_Billing"] == 1
    assert result.loc[0, "category_Technical"] == 0


def test_build_ml_ready_dataset_has_no_remaining_categorical_columns():
    df = pd.DataFrame({
        "priority": ["low", "high"],
        "category": ["Billing", "Technical"],
        "status": ["Open", "Closed"],
        "cost": [10.0, 20.0],
    })
    result = build_ml_ready_dataset(df)

    assert "priority" not in result.columns  # reemplazada por priority_encoded
    assert "category" not in result.columns  # reemplazada por dummies
    assert "status" not in result.columns
    assert result.select_dtypes(include="object").empty  # nada de texto queda
