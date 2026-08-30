"""Pruebas unitarias para src/visualization/plots.py."""
import numpy as np
import pandas as pd

from src.visualization.plots import plot_confusion_matrix, plot_correlation_heatmap


def test_plot_correlation_heatmap_writes_a_file(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1],
        "c": ["x", "y", "z", "x", "y"],  # no numerica, debe ignorarse
    })
    output_path = tmp_path / "heatmap.png"
    result_path = plot_correlation_heatmap(df, output_path=output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_correlation_heatmap_respects_explicit_columns(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [9, 9, 9]})
    output_path = tmp_path / "heatmap_subset.png"
    plot_correlation_heatmap(df, columns=["a", "b"], output_path=output_path)
    assert output_path.exists()


def test_plot_confusion_matrix_writes_a_file(tmp_path):
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0]
    output_path = tmp_path / "confusion.png"
    result_path = plot_confusion_matrix(y_true, y_pred, labels=["No", "Si"], output_path=output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
