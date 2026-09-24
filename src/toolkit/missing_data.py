"""Estrategias de imputación de valores faltantes, reutilizables entre dominios."""
from __future__ import annotations

import pandas as pd


def compute_category_means(df: pd.DataFrame, value_column: str, category_column: str) -> dict:
    """`fit`: aprende la media condicional por `category_column` (más la media
    global como respaldo) a partir de `df`, sin modificar nada.

    Separado de `apply_category_means` a propósito -- es lo que permite ajustar
    estos estadísticos **solo** sobre el histórico/entrenamiento y aplicar los
    mismos números, ya congelados, sobre datos que este `df` nunca vio (un
    período de test posterior). Ver `impute_numeric_by_category` para el uso
    conveniente de un solo paso cuando no hace falta esa separación.
    """
    return {
        "category_means": df.groupby(category_column)[value_column].mean().to_dict(),
        "global_mean": float(df[value_column].mean()),
    }


def apply_category_means(
    df: pd.DataFrame, value_column: str, category_column: str, stats: dict,
) -> pd.DataFrame:
    """`transform`: rellena nulos en `value_column` con los estadísticos YA
    APRENDIDOS por `compute_category_means` -- no recalcula nada a partir de
    `df`, así que es seguro llamarla sobre un conjunto de test que nunca
    participó del ajuste.

    Una categoría presente en `df` pero ausente en `stats["category_means"]`
    (una categoría nueva, no vista en el ajuste) cae directo a `global_mean`,
    igual que una categoría sin ningún valor no nulo en el ajuste original.
    """
    df = df.copy()
    mapped_means = df[category_column].map(stats["category_means"])
    df[value_column] = df[value_column].fillna(mapped_means).fillna(stats["global_mean"])
    return df


def impute_numeric_by_category(df: pd.DataFrame, value_column: str, category_column: str) -> pd.DataFrame:
    """Imputa nulos en `value_column` con la media condicional por `category_column`,
    ajustando y aplicando sobre el mismo `df` en un solo paso.

    Si una categoría no tiene ningún valor no nulo, recurre a la media global de la
    columna. Útil cuando el valor esperado varía estructuralmente según un grupo
    (ej. producción de cobre por empresa, costo de ticket por categoría) **y no
    hay una frontera train/test que respetar** -- si la hay, usar
    `compute_category_means` sobre el histórico y `apply_category_means` sobre
    ambos subconjuntos, como hacen los dominios de este repo que sí tienen esa
    frontera.
    """
    stats = compute_category_means(df, value_column, category_column)
    return apply_category_means(df, value_column, category_column, stats)


def interpolate_numeric_column(df: pd.DataFrame, column: str, sort_by: str | None = None) -> pd.DataFrame:
    """Imputa nulos en `column` por interpolación lineal respetando el orden temporal.

    Ordena por `sort_by` (típicamente una columna de fecha) antes de interpolar y
    restaura el orden original de las filas al terminar.
    """
    df = df.copy()
    order = df.sort_values(sort_by).index if sort_by else df.index
    interpolated = df.loc[order, column].interpolate(method="linear", limit_direction="both")
    df.loc[order, column] = interpolated
    return df


def interpolate_within_group(df: pd.DataFrame, column: str, group_column: str, sort_by: str) -> pd.DataFrame:
    """Interpola `column` por interpolación lineal, por separado dentro de cada grupo.

    Necesario en paneles (país×año, empresa×mes): interpolar a través de un cambio
    de grupo mezclaría series que no tienen relación temporal real entre sí (ej. el
    último año de Chile con el primer año de Perú si el DataFrame está ordenado por
    fecha global en vez de por grupo).

    Implementado con `groupby(...).transform` (no `.apply`) a propósito: con
    muchos grupos, `.apply` devolviendo un DataFrame de 2 columnas puede
    producir una forma ambigua (a veces Series, a veces DataFrame según el
    tamaño/contenido de cada grupo) y fallar con `ValueError: Columns must be
    same length as key` -- `transform` siempre devuelve un resultado alineado
    al índice de entrada, sin esa ambigüedad.
    """
    df = df.copy()
    order = df.sort_values([group_column, sort_by]).index
    sorted_df = df.loc[order]
    interpolated = sorted_df.groupby(group_column)[column].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    df.loc[order, column] = interpolated
    return df


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen tabular de nulos por columna: conteo, porcentaje y dtype.

    Primer paso diagnóstico antes de decidir qué estrategia de imputación aplicar
    a cada columna -- no toda columna con nulos debería tratarse igual.
    """
    total = len(df)
    report = pd.DataFrame({
        "columna": df.columns,
        "n_nulos": df.isna().sum().values,
        "pct_nulos": (df.isna().sum().values / total * 100).round(2) if total else 0,
        "dtype": df.dtypes.astype(str).values,
    })
    return report.sort_values("pct_nulos", ascending=False).reset_index(drop=True)
