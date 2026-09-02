"""Pruebas unitarias para src/toolkit/encoding.py."""
import numpy as np
import pandas as pd

from src.toolkit.encoding import encode_categorical_onehot, encode_ordinal, inverse_zscore, zscore_scale

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def test_encode_ordinal_respects_explicit_order():
    df = pd.DataFrame({"nivel": ["low", "critical", "medium", "high"]})
    result = encode_ordinal(df, "nivel", order=SEVERITY_ORDER)
    assert result["nivel_encoded"].tolist() == [0, 3, 1, 2]


def test_encode_ordinal_is_case_insensitive():
    df = pd.DataFrame({"nivel": ["LOW", "Critical"]})
    result = encode_ordinal(df, "nivel", order=SEVERITY_ORDER)
    assert result["nivel_encoded"].tolist() == [0, 3]


def test_encode_ordinal_leaves_unknown_values_as_nan():
    df = pd.DataFrame({"nivel": ["urgent"]})
    result = encode_ordinal(df, "nivel", order=SEVERITY_ORDER)
    assert pd.isna(result.loc[0, "nivel_encoded"])


def test_encode_categorical_onehot_creates_one_column_per_category():
    df = pd.DataFrame({"pais": ["Chile", "Peru"]})
    result = encode_categorical_onehot(df, columns=["pais"])
    assert "pais_Chile" in result.columns
    assert result.loc[0, "pais_Chile"] == 1
    assert result.loc[0, "pais_Peru"] == 0


def test_zscore_scale_produces_mean_zero_and_reversible():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
    scaled, stats = zscore_scale(df, ["x"])
    assert abs(scaled["x"].mean()) < 1e-9

    mean, std = stats["x"]
    recovered = inverse_zscore(scaled["x"].values, mean, std)
    assert np.allclose(recovered, df["x"].values)


def test_zscore_scale_handles_zero_variance_column():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    scaled, stats = zscore_scale(df, ["x"])
    assert scaled["x"].tolist() == [0.0, 0.0, 0.0]
