"""Modelo baseline de pronóstico: horas hasta resolución de un ticket.

Consume el dataset ya limpio, filtrado a alcance (`src/filters/business_filters.py`)
y codificado (`src/features/feature_encoder.py`) para demostrar, con un modelo real
entrenado y evaluado -- no solo "los datos quedaron listos para ML" en abstracto --
que la limpieza produce señal aprovechable. `category` y `priority` están ligadas
causalmente a `response_time_hours` en el generador sintético
(`src/generators/dirty_data_generator.py::_resolution_hours`), así que un modelo
correctamente entrenado debería recuperar ese patrón; la importancia de features
reportada al final es la verificación de que efectivamente lo hace, contra un
terreno conocido, no una afirmación sin evidencia.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.features.feature_encoder import build_ml_ready_dataset
from src.filters.business_filters import apply_ml_scope_filters

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_it_tickets.csv"

TARGET_COLUMN = "response_time_hours"
FEATURE_COLUMNS_TO_DROP = [
    "ticket_id", "company_name", "customer_email", "agent_name",
    "created_at", "resolved_at", TARGET_COLUMN,
]


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica el filtrado de alcance + codificación, y separa `(X, y)` para entrenar."""
    scoped = apply_ml_scope_filters(df)
    encoded = build_ml_ready_dataset(scoped)

    y = encoded[TARGET_COLUMN]
    X = encoded.drop(columns=[c for c in FEATURE_COLUMNS_TO_DROP if c in encoded.columns])
    # user_metadata_* quedan como texto (browser/os/plan) -- no son parte de este
    # modelo baseline, se excluyen explícitamente en vez de dejar que get_dummies
    # las intente encodear como si fueran categorías de bajo cardinality.
    text_like = [c for c in X.columns if c.startswith("user_metadata")]
    X = X.drop(columns=text_like)

    return X, y


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Entrena un RandomForestRegressor y lo compara contra un baseline de media.

    Devuelve un dict con MAE/R² del modelo y del baseline, más importancia de
    features -- todo a partir de una corrida real, no estimado.
    """
    X, y = prepare_training_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_prediction = y_train.mean()
    baseline_mae = mean_absolute_error(y_test, [baseline_prediction] * len(y_test))
    baseline_r2 = r2_score(y_test, [baseline_prediction] * len(y_test))

    importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_mae": mean_absolute_error(y_test, predictions),
        "model_r2": r2_score(y_test, predictions),
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2,
        "feature_importance": importance,
    }


if __name__ == "__main__":
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    results = train_and_evaluate(clean_df)

    print(f"Entrenamiento: {results['n_train']:,} filas -- Test: {results['n_test']:,} filas")
    print(f"\nRandomForest -- MAE: {results['model_mae']:.2f}h  R²: {results['model_r2']:.3f}")
    print(f"Baseline (media) -- MAE: {results['baseline_mae']:.2f}h  R²: {results['baseline_r2']:.3f}")
    print(f"\nMejora de MAE sobre baseline: {(1 - results['model_mae'] / results['baseline_mae']) * 100:.1f}%")
    print("\nImportancia de features (top 10):")
    print(results["feature_importance"].head(10).to_string())
