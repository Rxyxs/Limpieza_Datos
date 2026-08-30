"""Visualizaciones del pipeline: heatmap de correlación y matrices de confusión.

Separado de los módulos de modelado a propósito -- graficar no es responsabilidad
de un modelo, y mantenerlo aparte permite reusar `plot_confusion_matrix` para
cualquier clasificador futuro sin duplicar código de ploteo.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    output_path: Path | None = None,
    title: str = "Correlación entre features numéricas",
) -> Path:
    """Genera un heatmap de correlación (Pearson) sobre `columns` (o todas las
    numéricas de `df` si no se especifica) y lo guarda en `output_path`.

    Sirve tanto de EDA (¿qué features numéricas se mueven juntas antes de
    modelar?) como de chequeo de sanidad: `cost` no debería correlacionar con
    nada relevante (se genera de forma independiente en el generador
    sintético), mientras que `priority_encoded` sí debería mostrar relación
    con `response_time_hours` -- la estructura causal inyectada tiene que
    aparecer acá antes de que aparezca en la importancia de features de un
    modelo.
    """
    numeric_df = df[columns] if columns else df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)

    output_path = output_path or (FIGURES_DIR / "correlation_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns) * 0.9), max(5, len(corr.columns) * 0.75)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    output_path: Path,
    title: str = "Matriz de confusión",
) -> Path:
    """Genera y guarda una matriz de confusión como heatmap anotado."""
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


if __name__ == "__main__":
    from src.features.feature_encoder import build_ml_ready_dataset
    from src.filters.business_filters import apply_ml_scope_filters

    clean_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "clean_it_tickets.csv")
    scoped_df = apply_ml_scope_filters(clean_df)
    encoded_df = build_ml_ready_dataset(scoped_df)

    numeric_columns = ["cost", "response_time_hours", "priority_encoded"] + [
        c for c in encoded_df.columns if c.startswith("category_") or c.startswith("status_")
    ]
    saved_path = plot_correlation_heatmap(encoded_df, columns=numeric_columns)
    print(f"Heatmap de correlación guardado en {saved_path}")

    correlations = encoded_df[numeric_columns].corr(numeric_only=True)["response_time_hours"].sort_values(ascending=False)
    print("\nCorrelación de cada feature con response_time_hours:")
    print(correlations.to_string())
