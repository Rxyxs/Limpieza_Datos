"""Detección de filas casi-duplicadas (no solo duplicados exactos).

Distinto de `text_cleaning.unify_similar_names`: ese módulo unifica variantes
del *valor* de una columna en todo el dataset. Este módulo detecta *filas* que
probablemente son el mismo registro capturado dos veces -- misma clave lógica,
en el mismo instante o casi, con un campo de texto casi idéntico (typo de
reenvío/doble carga). Un dedup exacto (`DataFrame.duplicated()`) no atrapa este
caso porque el id siempre difiere y el campo de texto difiere por el typo.
"""
from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz


def find_near_duplicate_rows(
    df: pd.DataFrame,
    key_column: str,
    fuzzy_column: str,
    group_column: str | None = None,
    timestamp_column: str | None = None,
    id_column: str | None = None,
    time_window_minutes: float = 10.0,
    similarity_threshold: float = 80.0,
) -> pd.DataFrame:
    """Marca pares de filas candidatas a ser el mismo registro duplicado.

    Agrupa por `key_column` (+ `group_column` si se da) y, dentro de cada grupo,
    compara pares creados a menos de `time_window_minutes` de diferencia (si hay
    `timestamp_column`) vía similitud fuzzy (`rapidfuzz.fuzz.token_sort_ratio`)
    sobre `fuzzy_column`. Si no se da `timestamp_column`, compara todos los pares
    dentro del grupo sin restricción temporal.

    No elimina ni fusiona nada -- devuelve un DataFrame de pares candidatos con
    su score, dejando la decisión de fusionar/descartar a revisión humana o una
    regla de negocio explícita aguas abajo.
    """
    group_cols = [key_column] + ([group_column] if group_column else [])
    required = set(group_cols + [fuzzy_column] + ([timestamp_column] if timestamp_column else []))
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Columnas requeridas ausentes: {missing}")

    working = df.dropna(subset=group_cols).copy()
    if timestamp_column:
        working[timestamp_column] = pd.to_datetime(working[timestamp_column], utc=True, errors="coerce")
        working = working.dropna(subset=[timestamp_column])

    candidates: list[dict] = []
    grouped = working.groupby(group_cols)

    for _, group in grouped:
        if len(group) < 2:
            continue
        if timestamp_column:
            group = group.sort_values(timestamp_column)
        rows = group.to_dict(orient="records")
        indices = group.index.tolist()

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if timestamp_column:
                    delta_minutes = abs(
                        (rows[j][timestamp_column] - rows[i][timestamp_column]).total_seconds()
                    ) / 60
                    if delta_minutes > time_window_minutes:
                        break  # ordenado por tiempo: filas siguientes solo se alejan más
                else:
                    delta_minutes = None

                val_a, val_b = rows[i].get(fuzzy_column), rows[j].get(fuzzy_column)
                if pd.isna(val_a) or pd.isna(val_b):
                    continue

                score = fuzz.token_sort_ratio(str(val_a), str(val_b))
                if score >= similarity_threshold:
                    candidates.append({
                        "idx_a": indices[i],
                        "idx_b": indices[j],
                        "id_a": rows[i].get(id_column) if id_column else indices[i],
                        "id_b": rows[j].get(id_column) if id_column else indices[j],
                        "similarity_score": score,
                        "minutes_apart": round(delta_minutes, 2) if delta_minutes is not None else None,
                    })

    return pd.DataFrame(
        candidates,
        columns=["idx_a", "idx_b", "id_a", "id_b", "similarity_score", "minutes_apart"],
    )


def exact_duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Elimina duplicados exactos (misma clave en `subset`, o toda la fila) y reporta cuántos se quitaron."""
    n_before = len(df)
    deduped = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return deduped, n_before - len(deduped)
