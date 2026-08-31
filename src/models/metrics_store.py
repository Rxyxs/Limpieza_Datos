"""Persistencia de métricas comparativas de modelos en DuckDB.

Hasta ahora las métricas de cada corrida (`train_and_compare_activations`,
`train_and_compare_all_models`) solo vivían impresas en consola -- útil para
una corrida puntual, pero no permite comparar corridas a lo largo del tiempo
(por ejemplo, tras cambiar hiperparámetros o el dataset sintético). Este
módulo agrega una tabla local `sla_model_metrics` en un archivo DuckDB
embebido (`outputs/metrics.duckdb`, no versionado) donde cada llamada a
`persist_sla_comparison_metrics` agrega una fila por modelo con un timestamp
de corrida, sin sobreescribir corridas anteriores.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "outputs" / "metrics.duckdb"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sla_model_metrics (
    run_id TIMESTAMP,
    model_name VARCHAR,
    n_train INTEGER,
    n_test INTEGER,
    breach_rate_test DOUBLE,
    accuracy DOUBLE,
    precision DOUBLE,
    recall DOUBLE,
    f1 DOUBLE
)
"""


def persist_sla_comparison_metrics(outcome: dict, db_path: Path | None = None) -> Path:
    """Agrega una fila por modelo de `outcome["results_by_model"]` (o
    `outcome["results_by_activation"]`, para compatibilidad con la comparación
    de dos activaciones original) a la tabla `sla_model_metrics` de DuckDB.

    Devuelve la ruta del archivo DuckDB usado. No sobreescribe corridas
    anteriores -- cada llamada es un `INSERT`, identificado por `run_id`.
    """
    db_path = db_path or DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    results = outcome.get("results_by_model") or outcome.get("results_by_activation")
    if results is None:
        raise ValueError("outcome debe traer 'results_by_model' o 'results_by_activation'")

    run_id = datetime.now(timezone.utc)
    rows = [
        {
            "run_id": run_id,
            "model_name": model_name,
            "n_train": outcome["n_train"],
            "n_test": outcome["n_test"],
            "breach_rate_test": outcome["breach_rate_test"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        }
        for model_name, result in results.items()
    ]
    rows_df = pd.DataFrame(rows)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(_CREATE_TABLE_SQL)
        con.execute("INSERT INTO sla_model_metrics SELECT * FROM rows_df")
    finally:
        con.close()

    return db_path


def read_latest_run(db_path: Path | None = None) -> pd.DataFrame:
    """Devuelve las métricas de la corrida más reciente (todas las filas que
    comparten el `run_id` máximo) como DataFrame, ordenadas por F1 descendente.
    """
    db_path = db_path or DEFAULT_DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        return con.execute(
            """
            SELECT * FROM sla_model_metrics
            WHERE run_id = (SELECT max(run_id) FROM sla_model_metrics)
            ORDER BY f1 DESC
            """
        ).df()
    finally:
        con.close()


if __name__ == "__main__":
    from src.models.forecast_response_time import CLEAN_DATA_PATH
    from src.models.sla_breach_classifier import train_and_compare_all_models

    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    run_outcome = train_and_compare_all_models(clean_df)
    saved_path = persist_sla_comparison_metrics(run_outcome)
    print(f"Métricas persistidas en {saved_path}")
    print(read_latest_run(saved_path).to_string(index=False))
