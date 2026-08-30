"""Detección de tickets casi-duplicados (double-submission), no solo exactos.

Distinto de `string_cleaner.unify_similar_names`: ese módulo unifica variantes
del *nombre* de una misma empresa en toda la columna. Este módulo detecta
*filas* que probablemente son el mismo ticket capturado dos veces -- mismo
cliente (`customer_email`, un identificador más estable que el nombre de
empresa, que es justo el campo que suele llevar el typo introducido en el
reenvío), misma categoría, creado en el mismo instante o casi, con un
`company_name` casi idéntico. Un dedup exacto (`DataFrame.duplicated()`) no
atrapa este caso porque `ticket_id` siempre difiere y `company_name` difiere
por el typo.
"""
from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz


def find_near_duplicate_tickets(
    df: pd.DataFrame,
    email_column: str = "customer_email",
    company_column: str = "company_name",
    category_column: str = "category",
    timestamp_column: str = "created_at",
    time_window_minutes: float = 10.0,
    similarity_threshold: float = 80.0,
) -> pd.DataFrame:
    """Marca pares de filas candidatas a ser el mismo ticket duplicado.

    Agrupa por `email_column` + `category_column` (un cliente y una categoría
    real rara vez generan dos tickets genuinamente distintos en la misma
    ventana de minutos), y dentro de cada grupo compara pares creados a menos
    de `time_window_minutes` de diferencia via similitud fuzzy
    (`rapidfuzz.fuzz.token_sort_ratio`) sobre `company_column` -- que es
    justamente el campo que un reenvío con typo altera.

    No elimina ni fusiona nada -- devuelve un DataFrame de pares candidatos
    con su score, dejando la decisión de fusionar/descartar a una revisión
    humana o una regla de negocio explícita aguas abajo.

    Devuelve columnas: `idx_a`, `idx_b`, `ticket_id_a`, `ticket_id_b`,
    `similarity_score`, `minutes_apart`.
    """
    required = {email_column, company_column, category_column, timestamp_column}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Columnas requeridas ausentes: {missing}")

    working = df.dropna(subset=[email_column, category_column, timestamp_column]).copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], utc=True, errors="coerce")
    working = working.dropna(subset=[timestamp_column])

    candidates: list[dict] = []
    grouped = working.groupby([email_column, category_column])

    for _, group in grouped:
        if len(group) < 2:
            continue
        group = group.sort_values(timestamp_column)
        rows = group.to_dict(orient="records")
        indices = group.index.tolist()

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                delta_minutes = abs((rows[j][timestamp_column] - rows[i][timestamp_column]).total_seconds()) / 60
                if delta_minutes > time_window_minutes:
                    break  # ordenado por tiempo: filas siguientes solo se alejan más

                company_a, company_b = rows[i].get(company_column), rows[j].get(company_column)
                if pd.isna(company_a) or pd.isna(company_b):
                    continue

                score = fuzz.token_sort_ratio(str(company_a), str(company_b))
                if score >= similarity_threshold:
                    candidates.append({
                        "idx_a": indices[i],
                        "idx_b": indices[j],
                        "ticket_id_a": rows[i].get("ticket_id"),
                        "ticket_id_b": rows[j].get("ticket_id"),
                        "similarity_score": score,
                        "minutes_apart": round(delta_minutes, 2),
                    })

    return pd.DataFrame(
        candidates,
        columns=["idx_a", "idx_b", "ticket_id_a", "ticket_id_b", "similarity_score", "minutes_apart"],
    )
