"""Clasificador de incumplimiento de SLA: ¿este ticket va a tardar más de
`SLA_THRESHOLD_HOURS` en resolverse? -- ReLU vs. Tanh, comparación controlada.

Reutiliza `prepare_training_frame` de `forecast_response_time.py` (mismas
features, mismo alcance filtrado) y binariza el target continuo
`response_time_hours` en vez de duplicar la lógica de preparación de datos.
Igual que en `forecast_response_time.py`, `category`+`priority` están
causalmente ligadas al tiempo de resolución en el generador sintético, así
que el clasificador tiene señal real que aprender, no ruido.

Comparación de arquitecturas idénticas (`MLPClassifier`, misma cantidad de
capas/neuronas, mismo seed) que solo difieren en la función de activación de
las capas ocultas -- ReLU vs. Tanh -- el mismo patrón de comparación
controlada usado en otro repo del portafolio
(`crypto-direction-deep-learning`), aplicado acá a un problema real y
distinto: decisión de negocio binaria (breach/no-breach) en vez de dirección
de precio.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.models.forecast_response_time import CLEAN_DATA_PATH, prepare_training_frame
from src.visualization.plots import FIGURES_DIR, plot_confusion_matrix

SLA_THRESHOLD_HOURS = 24.0
CLASS_LABELS = ["Dentro de SLA", "Incumple SLA"]
HIDDEN_LAYER_SIZES = (16, 8)


def prepare_classification_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reusa la preparación de `forecast_response_time.py` y binariza el target:
    `1` si `response_time_hours` supera `SLA_THRESHOLD_HOURS`, `0` si no.
    """
    X, y_hours = prepare_training_frame(df)
    y_breach = (y_hours > SLA_THRESHOLD_HOURS).astype(int)
    return X, y_breach


def _train_single_activation(
    activation: str, X_train, X_test, y_train, y_test, random_state: int
) -> dict:
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=activation,
        solver="adam",
        max_iter=500,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "activation": activation,
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "report": classification_report(y_test, predictions, target_names=CLASS_LABELS, zero_division=0),
    }


def train_and_compare_activations(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Entrena dos `MLPClassifier` idénticos salvo por la función de activación
    de las capas ocultas (ReLU vs. Tanh) y devuelve las métricas de ambos.

    Las features se escalan con `StandardScaler` (ajustado solo en train) --
    a diferencia de un RandomForest, una red neuronal es sensible a la escala
    de entrada, y `cost` (0-5000) sin escalar dominaría sobre
    `priority_encoded` (0-3) por magnitud, no por señal real.
    """
    X, y = prepare_classification_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {
        activation: _train_single_activation(activation, X_train_scaled, X_test_scaled, y_train, y_test, random_state)
        for activation in ("relu", "tanh")
    }

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "breach_rate_train": float(y_train.mean()),
        "breach_rate_test": float(y_test.mean()),
        "y_test": y_test,
        "results_by_activation": results,
    }


if __name__ == "__main__":
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    outcome = train_and_compare_activations(clean_df)

    print(f"Entrenamiento: {outcome['n_train']:,} filas -- Test: {outcome['n_test']:,} filas")
    print(f"Tasa de incumplimiento de SLA (>{SLA_THRESHOLD_HOURS:.0f}h) -- train: {outcome['breach_rate_train']:.1%}  test: {outcome['breach_rate_test']:.1%}")

    for activation, result in outcome["results_by_activation"].items():
        print(f"\n=== Activación: {activation} ===")
        print(f"Accuracy: {result['accuracy']:.3f}  Precision: {result['precision']:.3f}  Recall: {result['recall']:.3f}  F1: {result['f1']:.3f}")
        print(result["report"])

        output_path = FIGURES_DIR / f"confusion_matrix_{activation}.png"
        plot_confusion_matrix(
            outcome["y_test"], result["predictions"], CLASS_LABELS, output_path,
            title=f"Matriz de confusión -- MLPClassifier ({activation})",
        )
        print(f"Matriz de confusión guardada en {output_path}")

    relu_f1 = outcome["results_by_activation"]["relu"]["f1"]
    tanh_f1 = outcome["results_by_activation"]["tanh"]["f1"]
    winner = "relu" if relu_f1 >= tanh_f1 else "tanh"
    print(f"\nMejor F1: {winner} (relu={relu_f1:.3f} vs. tanh={tanh_f1:.3f})")
