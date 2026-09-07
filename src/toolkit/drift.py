"""Detección de *drift*: cuánto se movió la distribución de cada columna entre
dos muestras del mismo dato -- el archivo del mes pasado contra el de este mes,
o el período de entrenamiento contra el de test.

Es el chequeo que no aparece en ningún reporte de missingness ni de outliers,
porque no hay nada roto: cada valor individual es válido y plausible. Lo que
cambió es la *forma* de la distribución, y esa es exactamente la condición que
un modelo ya entrenado no puede absorber -- fue ajustado sobre la distribución
vieja y se lo evalúa sobre la nueva.

Los tres dominios de panel de este repo lo muestran de la forma más directa
posible: sus baselines de "predecir el promedio histórico" dan R² NEGATIVO, que
es peor que no modelar nada. No es un error de cálculo. Es drift: el rendimiento
de cereales y la esperanza de vida tienen tendencia real al alza, así que la
media del período de entrenamiento subestima sistemáticamente el período de
test. Este módulo pone un número sobre eso antes de entrenar, en vez de
descubrirlo después mirando una métrica mala.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Umbrales convencionales del PSI, heredados del scoring de crédito, donde el
# índice se usa desde hace décadas para decidir si un modelo sigue vigente. Son
# una convención de la industria, no un resultado estadístico: no hay un test de
# hipótesis detrás ni un nivel de significancia asociado, así que sirven para
# ordenar y priorizar columnas, no para afirmar que un cambio es "significativo".
PSI_MODERATE = 0.10
PSI_SEVERE = 0.25

# Se suma a cada proporción antes del logaritmo: un bin sin observaciones en una
# de las dos muestras haría log(0) = -inf y contaminaría el PSI de la columna
# entera con un infinito, escondiendo el aporte real de los demás bins.
EPSILON = 1e-6


def population_stability_index(
    expected, actual, bins: int = 10,
) -> float:
    """Population Stability Index entre una muestra de referencia (`expected`) y
    una nueva (`actual`). 0 = distribuciones idénticas; más alto = más movida.

    **Los cortes de los bins salen de `expected`, nunca de `actual`.** Es la
    decisión que define la medida: si cada muestra se bineara por sus propios
    cuantiles, ambas quedarían con ~10% de los datos en cada bin por
    construcción y el PSI daría cerca de cero *justamente cuando la
    distribución más se movió*. La muestra vieja define la regla; la nueva se
    mide contra ella.

    Se usan cuantiles y no bins de ancho fijo porque una sola cola larga
    (habitual en producción minera o en PIB per cápita) dejaría casi todos los
    datos en un bin y el índice no vería nada.
    """
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.empty or actual.empty:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # columna casi constante: no hay distribución que comparar
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_share = np.histogram(expected, bins=edges)[0] / len(expected) + EPSILON
    actual_share = np.histogram(actual, bins=edges)[0] / len(actual) + EPSILON

    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def categorical_drift(expected, actual) -> dict:
    """Drift de una columna categórica: distancia de variación total entre las dos
    distribuciones de frecuencia, más las categorías que aparecen o desaparecen.

    Las categorías nuevas importan aparte del número: una categoría que el
    modelo nunca vio en entrenamiento no es "un poco de drift", es una entrada
    que su encoder no sabe representar -- con `encode_categorical_onehot` cae en
    todo ceros, indistinguible de cualquier otra categoría desconocida.
    """
    expected_counts = pd.Series(expected).dropna().astype(str).value_counts(normalize=True)
    actual_counts = pd.Series(actual).dropna().astype(str).value_counts(normalize=True)

    todas = expected_counts.index.union(actual_counts.index)
    p = expected_counts.reindex(todas, fill_value=0.0)
    q = actual_counts.reindex(todas, fill_value=0.0)

    return {
        "distancia_variacion_total": float(0.5 * np.abs(p - q).sum()),
        "categorias_nuevas": sorted(set(actual_counts.index) - set(expected_counts.index)),
        "categorias_ausentes": sorted(set(expected_counts.index) - set(actual_counts.index)),
    }


def classify_psi(psi: float) -> str:
    """Traduce un PSI a `estable` / `moderado` / `severo` según los umbrales."""
    if pd.isna(psi):
        return "sin datos"
    if psi < PSI_MODERATE:
        return "estable"
    return "moderado" if psi < PSI_SEVERE else "severo"


def drift_report(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Reporte de drift columna por columna entre dos DataFrames del mismo esquema.

    Para columnas numéricas devuelve PSI, el estadístico KS y su p-valor; para
    las categóricas, la distancia de variación total y el conteo de categorías
    nuevas. Ordena por severidad, que es como se usa: la pregunta operativa no
    es "¿hay drift?" (casi siempre hay algo) sino "¿qué columna miro primero?".

    KS y PSI responden cosas distintas y por eso van los dos. KS trae un p-valor
    y con muestras grandes marca como significativo un desplazamiento
    minúsculo, irrelevante para el modelo; el PSI no tiene p-valor pero mide la
    MAGNITUD del movimiento. Una columna con p<0.001 y PSI de 0.02 se movió de
    forma detectable y da lo mismo; una con PSI de 0.4 hay que mirarla aunque el
    p-valor no impresione.
    """
    target_columns = columns if columns is not None else list(expected_df.columns)
    rows = []

    for col in target_columns:
        if col not in actual_df.columns:
            continue

        expected, actual = expected_df[col], actual_df[col]
        es_numerica = pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual)

        if es_numerica:
            psi = population_stability_index(expected, actual, bins=bins)
            limpio_e, limpio_a = expected.dropna(), actual.dropna()
            if len(limpio_e) and len(limpio_a):
                ks = stats.ks_2samp(limpio_e, limpio_a)
                ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
            else:
                ks_stat = ks_p = float("nan")
            rows.append({
                "columna": col, "tipo": "numerica", "psi": psi, "veredicto": classify_psi(psi),
                "ks_estadistico": ks_stat, "ks_pvalor": ks_p,
                "media_expected": float(limpio_e.mean()) if len(limpio_e) else float("nan"),
                "media_actual": float(limpio_a.mean()) if len(limpio_a) else float("nan"),
                "categorias_nuevas": 0,
            })
        else:
            cat = categorical_drift(expected, actual)
            tvd = cat["distancia_variacion_total"]
            rows.append({
                "columna": col, "tipo": "categorica", "psi": tvd,
                "veredicto": classify_psi(tvd), "ks_estadistico": float("nan"),
                "ks_pvalor": float("nan"), "media_expected": float("nan"),
                "media_actual": float("nan"),
                "categorias_nuevas": len(cat["categorias_nuevas"]),
            })

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def target_shift(y_expected, y_actual) -> dict:
    """Cuantifica el desplazamiento del propio target entre dos períodos, y cuánto
    de él explicaría por sí solo un R² negativo del baseline "predigo la media".

    `r2_de_predecir_la_media_vieja` es el R² que obtiene, sobre el período
    nuevo, un modelo que predice constantemente la media del período viejo --
    exactamente lo que hacen los baselines de este proyecto. Si da negativo, no
    es un bug: significa que la media vieja quedó fuera del rango típico del
    período nuevo, que es la definición operativa de drift del target.
    """
    expected = pd.Series(y_expected).dropna().astype(float)
    actual = pd.Series(y_actual).dropna().astype(float)

    media_vieja = float(expected.mean())
    residuales = actual - media_vieja
    varianza_actual = float(((actual - actual.mean()) ** 2).sum())
    r2 = float("nan") if varianza_actual == 0 else 1 - float((residuales**2).sum()) / varianza_actual

    desvio_viejo = float(expected.std())
    return {
        "media_expected": media_vieja,
        "media_actual": float(actual.mean()),
        "desplazamiento_en_desvios": (float(actual.mean()) - media_vieja) / desvio_viejo if desvio_viejo else float("nan"),
        "r2_de_predecir_la_media_vieja": r2,
        "psi": population_stability_index(expected, actual),
    }
