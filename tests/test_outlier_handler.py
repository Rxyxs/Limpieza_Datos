"""Pruebas unitarias para src/cleaners/outlier_handler.py."""
import numpy as np
import pandas as pd

from src.cleaners.outlier_handler import iqr_bounds, winsorize_column, winsorize_column_by_group


def test_iqr_bounds_flags_a_known_extreme_value():
    series = pd.Series([10, 11, 12, 13, 14, 15, 1000])
    lower, upper = iqr_bounds(series)
    assert 1000 > upper
    assert 10 >= lower  # ningún valor "normal" debería quedar bajo el límite inferior


def test_winsorize_column_clips_extreme_value_and_counts_it():
    df = pd.DataFrame({"cost": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1000.0]})
    result, n_outliers = winsorize_column(df, "cost")

    assert n_outliers == 1
    assert result["cost"].max() < 1000.0


def test_winsorize_column_ignores_nulls():
    df = pd.DataFrame({"cost": [10.0, 11.0, 12.0, np.nan, 1000.0]})
    result, n_outliers = winsorize_column(df, "cost")

    assert pd.isna(result.loc[3, "cost"])  # el nulo no se toca, sigue nulo
    assert n_outliers == 1


def test_winsorize_column_by_group_uses_separate_bounds_per_group():
    # Grupo "A" tiene escala ~10-15, grupo "B" tiene escala ~500-600 -- un IQR
    # global marcaria TODO el grupo B como outlier; por grupo, ninguno lo es.
    df = pd.DataFrame({
        "value": [10, 11, 12, 13, 14, 500, 510, 520, 530, 540],
        "group": ["A"] * 5 + ["B"] * 5,
    })
    result, n_outliers = winsorize_column_by_group(df, "value", group_column="group")

    assert n_outliers == 0
    assert result["value"].tolist() == df["value"].tolist()


def test_winsorize_column_by_group_still_catches_within_group_outliers():
    df = pd.DataFrame({
        "value": [10, 11, 12, 13, 14, 9000],
        "group": ["A"] * 5 + ["A"],
    })
    result, n_outliers = winsorize_column_by_group(df, "value", group_column="group")

    assert n_outliers == 1
    assert result["value"].max() < 9000
