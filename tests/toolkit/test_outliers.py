"""Pruebas unitarias para src/toolkit/outliers.py."""
import numpy as np
import pandas as pd

from src.toolkit.outliers import (
    fix_implausible_level_jumps,
    iqr_bounds,
    winsorize_column,
    winsorize_column_by_group,
    zscore_outlier_mask,
)


def test_iqr_bounds_flags_a_known_extreme_value():
    series = pd.Series([10, 11, 12, 13, 14, 15, 1000])
    lower, upper = iqr_bounds(series)
    assert 1000 > upper
    assert 10 >= lower


def test_winsorize_column_clips_extreme_value_and_counts_it():
    df = pd.DataFrame({"valor": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1000.0]})
    result, n_outliers = winsorize_column(df, "valor")
    assert n_outliers == 1
    assert result["valor"].max() < 1000.0


def test_winsorize_column_ignores_nulls():
    df = pd.DataFrame({"valor": [10.0, 11.0, 12.0, np.nan, 1000.0]})
    result, n_outliers = winsorize_column(df, "valor")
    assert pd.isna(result.loc[3, "valor"])
    assert n_outliers == 1


def test_winsorize_column_by_group_uses_separate_bounds_per_group():
    df = pd.DataFrame({
        "valor": [10, 11, 12, 13, 14, 500, 510, 520, 530, 540],
        "grupo": ["A"] * 5 + ["B"] * 5,
    })
    result, n_outliers = winsorize_column_by_group(df, "valor", group_column="grupo")
    assert n_outliers == 0
    assert result["valor"].tolist() == df["valor"].tolist()


def test_fix_implausible_level_jumps_corrects_single_bad_point_via_interpolation():
    # Simula el bug real encontrado en la UF de mindicador.cl: un valor cae a
    # ~1/40 de su nivel real por un dia, y se "recupera" al dia siguiente --
    # el patron delator (salto + reversion) de un error de captura, no de un
    # movimiento de mercado real.
    df = pd.DataFrame({
        "fecha": pd.date_range("2024-01-01", periods=9),
        "valor": [24600.0, 24610.0, 24615.0, 608.15, 24630.0, 24635.0, 24640.0, 24645.0, 24650.0],
    })
    result, n_fixed = fix_implausible_level_jumps(df, "valor", sort_by="fecha", threshold=0.5)
    assert n_fixed == 1
    assert result.loc[3, "valor"] > 20000  # interpolado entre vecinos, no el valor corrupto


def test_fix_implausible_level_jumps_catches_two_consecutive_bad_points_similar_to_each_other():
    # El caso real que motivo reemplazar "ratio contra el dia anterior" por
    # "ratio contra la mediana movil": dos dias SEGUIDOS corruptos con valores
    # parecidos ENTRE SI (24627 -> 608.15 -> 607.38 -> 24627) -- el ratio
    # dia-a-dia entre los dos puntos malos (608.15 / 607.38) es ~normal, asi
    # que una deteccion dia-a-dia solo atrapa la entrada/salida del tramo, no
    # ambos puntos corruptos. Encontrado de verdad en la UF real de
    # mindicador.cl (2014-12-29/30).
    good = [24627.10] * 6
    df = pd.DataFrame({
        "fecha": pd.date_range("2014-12-20", periods=14),
        "valor": good + [608.15, 607.38] + good,
    })
    result, n_fixed = fix_implausible_level_jumps(df, "valor", sort_by="fecha", threshold=0.5)
    assert n_fixed == 2
    assert result.loc[6, "valor"] > 20000
    assert result.loc[7, "valor"] > 20000


def test_fix_implausible_level_jumps_leaves_normal_series_untouched():
    df = pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=5), "valor": [100.0, 101.0, 99.5, 100.2, 100.8]})
    result, n_fixed = fix_implausible_level_jumps(df, "valor", sort_by="fecha")
    assert n_fixed == 0
    assert result["valor"].tolist() == df["valor"].tolist()


def test_zscore_outlier_mask_flags_beyond_threshold():
    series = pd.Series([10, 11, 9, 10, 12, 200])
    mask = zscore_outlier_mask(series, threshold=2.0)
    assert mask.iloc[-1]
    assert not mask.iloc[0]


def test_zscore_outlier_mask_handles_zero_variance():
    series = pd.Series([5, 5, 5, 5])
    mask = zscore_outlier_mask(series)
    assert not mask.any()
