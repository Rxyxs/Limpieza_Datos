"""Pruebas unitarias para src/toolkit/viz.py -- verifican que cada función
genera un archivo real y no vacío, no la estética del gráfico."""
import numpy as np
import pandas as pd

from src.toolkit.viz import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution_before_after,
    plot_funnel,
    plot_missingness_before_after,
    plot_model_comparison_bars,
    plot_regression_diagnostics,
    plot_timeseries,
    plot_training_curve,
)


def _assert_written(path):
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_correlation_heatmap_writes_a_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": ["x", "y", "z", "x", "y"]})
    _assert_written(plot_correlation_heatmap(df, output_path=tmp_path / "heatmap.png"))


def test_plot_confusion_matrix_writes_a_file(tmp_path):
    _assert_written(plot_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1], labels=["No", "Si"], output_path=tmp_path / "cm.png"))


def test_plot_model_comparison_bars_writes_a_file(tmp_path):
    results = {"baseline": {"r2": 0.10, "rmse": 5.0}, "xgboost": {"r2": 0.65, "rmse": 2.1}}
    _assert_written(plot_model_comparison_bars(results, tmp_path / "cmp.png", metrics=("r2", "rmse"), title="t"))


def test_plot_funnel_writes_a_file(tmp_path):
    stages = {"crudo": 172900, "sin agregados": 141050, "con valor": 79597, "tras interpolar": 132600}
    _assert_written(plot_funnel(stages, tmp_path / "funnel.png", title="t"))


def test_plot_missingness_before_after_writes_a_file(tmp_path):
    before = pd.Series({"a": 40.0, "b": 10.0})
    after = pd.Series({"a": 0.0, "b": 0.0})
    _assert_written(plot_missingness_before_after(before, after, tmp_path / "miss.png"))


def test_plot_distribution_before_after_writes_a_file(tmp_path):
    before = pd.Series(np.random.default_rng(0).normal(0, 1, 200))
    after = pd.Series(np.random.default_rng(1).normal(0, 0.8, 200))
    _assert_written(plot_distribution_before_after(before, after, tmp_path / "dist.png", title="t"))


def test_plot_regression_diagnostics_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.normal(100, 10, 50)
    y_pred = y_true + rng.normal(0, 2, 50)
    _assert_written(plot_regression_diagnostics(y_true, y_pred, tmp_path / "reg.png", title="t"))


def test_plot_training_curve_writes_a_file(tmp_path):
    train = [1.0 / (i + 1) for i in range(120)]
    val = [1.1 / (i + 1) for i in range(120)]
    _assert_written(plot_training_curve(train, val, tmp_path / "curve.png", title="t", best_epoch=100))


def test_plot_timeseries_writes_a_file(tmp_path):
    df = pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=30), "valor": np.arange(30.0)})
    _assert_written(plot_timeseries(df, x="fecha", y="valor", output_path=tmp_path / "ts.png", title="t", rolling_window=7))
