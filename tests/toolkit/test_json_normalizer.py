"""Pruebas unitarias para src/toolkit/json_normalizer.py."""
import numpy as np
import pandas as pd

from src.toolkit.json_normalizer import normalize_json_column


def test_normalize_json_column_flattens_nested_dict():
    df = pd.DataFrame({
        "id": [1],
        "metadata": ['{"os": "Windows", "browser": {"name": "Chrome", "version": "120"}}'],
    })
    result = normalize_json_column(df, column="metadata")

    assert "metadata" not in result.columns
    assert result.loc[0, "metadata_os"] == "Windows"
    assert result.loc[0, "metadata_browser_name"] == "Chrome"


def test_normalize_json_column_handles_null_like_values():
    df = pd.DataFrame({"id": [1, 2, 3], "metadata": ["null", "", np.nan]})
    result = normalize_json_column(df, column="metadata")
    assert "metadata" not in result.columns
    assert len(result) == 3


def test_normalize_json_column_serializes_lists_as_text():
    df = pd.DataFrame({"id": [1], "metadata": ['{"tags": ["vip", "beta"]}']})
    result = normalize_json_column(df, column="metadata")
    assert result.loc[0, "metadata_tags"] == "vip, beta"
