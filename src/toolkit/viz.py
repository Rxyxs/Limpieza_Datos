"""Librería de gráficos reusable ("caja negra") para los 4 dominios: cada función
recibe datos ya limpios/resultados de modelo y un `output_path`, y no sabe nada
del dominio que la llama -- se reusa igual para el panel financiero, minero,
agrícola o el data warehouse de consultoría.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_missingness_before_after(
    pct_before: pd.Series, pct_after: pd.Series, output_path: Path, title: str = "Nulos por columna: antes vs. después de limpiar",
) -> Path:
    """Bar chart comparando el % de nulos por columna antes y después del pipeline
    de limpieza -- la primera evidencia visual de que la limpieza realmente hizo
    algo, no solo una afirmación en el README.
    """
    columns = sorted(set(pct_before.index) | set(pct_after.index), key=lambda c: -pct_before.get(c, 0))
    before_vals = [pct_before.get(c, 0) for c in columns]
    after_vals = [pct_after.get(c, 0) for c in columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(columns) * 0.55), 5))
    x = np.arange(len(columns))
    ax.bar(x - 0.2, before_vals, width=0.4, label="antes", color="#C1440E")
    ax.bar(x + 0.2, after_vals, width=0.4, label="después", color="#2A6F97")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_ylabel("% de filas con nulo")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_distribution_before_after(
    before: pd.Series, after: pd.Series, output_path: Path, title: str, xlabel: str = "",
) -> Path:
    """Histograma superpuesto de una columna numérica antes vs. después de tratar
    outliers/imputar -- muestra si el tratamiento realmente movió la distribución
    (ej. la winsorización recorta la cola) sin necesitar leer código.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(before.dropna(), color="#C1440E", label="antes", stat="density", kde=True, alpha=0.45, ax=ax)
    sns.histplot(after.dropna(), color="#2A6F97", label="después", stat="density", kde=True, alpha=0.45, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(
    df: pd.DataFrame, output_path: Path, columns: list[str] | None = None, title: str = "Correlación entre features numéricas",
) -> Path:
    """Heatmap de correlación (Pearson) sobre `columns` (o todas las numéricas)."""
    numeric_df = df[columns] if columns else df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns) * 0.8), max(5, len(corr.columns) * 0.65)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={"fontsize": 7})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(y_true, y_pred, labels: list[str], output_path: Path, title: str = "Matriz de confusión") -> Path:
    cm = confusion_matrix(y_true, y_pred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison_bars(
    results_by_model: dict, output_path: Path, metrics: tuple[str, ...], title: str,
) -> Path:
    """Bar chart agrupado comparando modelos entre sí sobre las métricas dadas
    (dict de nombre -> dict de métricas)."""
    model_names = list(results_by_model.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(metrics))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("deep", n_colors=len(model_names))
    for i, name in enumerate(model_names):
        values = [results_by_model[name][m] for m in metrics]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=palette[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_regression_diagnostics(
    y_true, y_pred, output_path: Path, title: str, unit: str = "",
) -> Path:
    """Panel de 2 gráficos: real vs. predicho (con línea y=x de referencia) y
    residuales vs. predicho -- el par estándar para auditar un modelo de
    regresión más allá de mirar solo el R².
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    residuals = y_true - y_pred

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_true, y_pred, alpha=0.5, s=18, color="#2A6F97")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "--", color="#C1440E", linewidth=1.5, label="y = x")
    axes[0].set_xlabel(f"Real {unit}")
    axes[0].set_ylabel(f"Predicho {unit}")
    axes[0].set_title("Real vs. predicho")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.5, s=18, color="#2A6F97")
    axes[1].axhline(0, linestyle="--", color="#C1440E", linewidth=1.5)
    axes[1].set_xlabel(f"Predicho {unit}")
    axes[1].set_ylabel("Residual (real - predicho)")
    axes[1].set_title("Residuales")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_training_curve(
    train_losses: list[float], val_losses: list[float], output_path: Path, title: str, best_epoch: int | None = None,
) -> Path:
    """Curva de pérdida (train/val) vs. época -- evidencia de que el modelo
    efectivamente entrenó >=100 épocas y de dónde quedó el mejor checkpoint.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train loss", color="#2A6F97")
    ax.plot(epochs, val_losses, label="val loss", color="#C1440E")
    if best_epoch:
        ax.axvline(best_epoch, linestyle="--", color="#6C757D", linewidth=1.2, label=f"mejor época = {best_epoch}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_funnel(stages: dict, output_path: Path, title: str, xlabel: str = "filas") -> Path:
    """Barra horizontal mostrando cuántas filas sobreviven cada etapa de un
    pipeline ETL (dict ordenado de nombre de etapa -> conteo de filas) -- útil
    para pipelines de varias etapas de filtrado (ej. warehouse: crudo ancho ->
    largo -> tras excluir agregados -> con valor real -> tras interpolar),
    donde una tabla de números no comunica tan rápido dónde se pierden filas.
    """
    labels = list(stages.keys())
    values = list(stages.values())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.7)))
    bars = ax.barh(labels, values, color="#2A6F97")
    ax.bar_label(bars, fmt="{:,.0f}", padding=4)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_timeseries(
    df: pd.DataFrame, x: str, y: str, output_path: Path, title: str, rolling_window: int | None = None, ylabel: str = "",
) -> Path:
    """Serie de tiempo simple, con media móvil opcional superpuesta."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df[x], df[y], color="#2A6F97", linewidth=0.9, alpha=0.7, label=y)
    if rolling_window:
        ax.plot(df[x], df[y].rolling(rolling_window).mean(), color="#C1440E", linewidth=1.8,
                label=f"media móvil ({rolling_window})")
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
