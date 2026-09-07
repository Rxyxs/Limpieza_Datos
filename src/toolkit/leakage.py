"""Detección de fuga de target (*target leakage*): features que ya contienen la
respuesta, y filas que aparecen en entrenamiento y en test a la vez.

La fuga es el único error de este proyecto que no se ve en ninguna métrica,
porque su síntoma es un resultado *bueno*. Un R² de 0.99 no dispara ninguna
alarma; se celebra, se pone en el README, y falla recién en producción, cuando
la feature que traía la respuesta no está disponible en el momento de predecir.
Por eso el chequeo tiene que correrse contra el resultado bueno, no contra el
malo.

Este módulo nació de un hallazgo real de este mismo repo: el Random Forest del
dominio de consultoría pone el **97,2% de su importancia en una sola feature**
(la esperanza de vida del año en curso, para predecir la del año siguiente).
Ese caso concreto NO es fuga -- el valor del año actual está genuinamente
disponible cuando se predice el siguiente, y la autocorrelación es real -- pero
tiene la firma numérica exacta de una fuga, y distinguir los dos casos requiere
mirar el dato, no la métrica. Lo que sigue produce esa evidencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Un solo predictor que explica esta proporción del target ya merece revisión
# manual: puede ser autocorrelación legítima (una serie contra su propio rezago)
# o la respuesta filtrada con otro nombre, y la métrica no distingue una de otra.
SINGLE_FEATURE_R2_ALERT = 0.95

# Tolerancia relativa para decidir si una relación entre feature y target es
# EXACTA. No se compara contra cero: un target derivado de la feature arrastra
# error de redondeo de punto flotante, y exigir igualdad exacta no encontraría
# nunca la fuga más obvia de todas.
EXACT_RELATION_TOLERANCE = 1e-9


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    varianza = float(((y_true - y_true.mean()) ** 2).sum())
    if varianza == 0:
        return float("nan")
    return 1 - float(((y_true - y_pred) ** 2).sum()) / varianza


def single_feature_power(
    X: pd.DataFrame, y, columns: list[str] | None = None,
) -> pd.DataFrame:
    """R² de una regresión lineal simple de cada feature contra el target, por
    separado, ordenado de mayor a menor.

    Una regresión de UNA variable y no un modelo completo, a propósito: la
    pregunta acá no es cuánto explica el conjunto de features (eso lo responde
    el modelo), sino si existe **alguna feature que sola ya sabe la respuesta**.
    Un R² individual de 0.99 tiene dos explicaciones posibles y hay que decidir
    cuál es: rezago legítimo de una serie autocorrelacionada, o la respuesta
    filtrada con otro nombre.

    `correlacion` va con signo porque ayuda a decidir: una correlación de -0.99
    con la mortalidad infantil es un driver epidemiológico real; una de +1.00
    con una columna llamada `total_final` es la respuesta.
    """
    objetivo = pd.Series(y).astype(float).reset_index(drop=True)
    target_columns = columns if columns is not None else [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
    ]

    filas = []
    for col in target_columns:
        feature = pd.to_numeric(X[col], errors="coerce").reset_index(drop=True)
        validas = feature.notna() & objetivo.notna()
        if validas.sum() < 3 or feature[validas].std() == 0:
            continue

        f, t = feature[validas].to_numpy(), objetivo[validas].to_numpy()
        pendiente, intercepto = np.polyfit(f, t, 1)
        r2 = _r2(t, pendiente * f + intercepto)

        filas.append({
            "feature": col,
            "r2_individual": r2,
            "correlacion": float(np.corrcoef(f, t)[0, 1]),
            "sospechosa": bool(r2 >= SINGLE_FEATURE_R2_ALERT),
        })

    reporte = pd.DataFrame(filas)
    if reporte.empty:
        return reporte
    return reporte.sort_values("r2_individual", ascending=False).reset_index(drop=True)


def exact_relations(X: pd.DataFrame, y, columns: list[str] | None = None) -> pd.DataFrame:
    """Busca features que sean una función determinística exacta del target:
    iguales, desplazadas por una constante, o escaladas por un factor constante.

    Es la fuga que un R² alto sugiere pero no prueba. Si `target - feature` es
    la misma constante en todas las filas, o si `target / feature` lo es, no hay
    nada que aprender: la columna ES el target en otras unidades o con otro
    origen (el mismo monto en miles, el mismo valor antes de sumarle un cargo
    fijo). Se detecta por identidad numérica fila a fila, igual que este
    proyecto ya identifica las columnas subtotal disfrazadas del Excel de
    COCHILCO, y no por el nombre de la columna.
    """
    objetivo = pd.Series(y).astype(float).reset_index(drop=True)
    target_columns = columns if columns is not None else [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
    ]

    filas = []
    for col in target_columns:
        feature = pd.to_numeric(X[col], errors="coerce").reset_index(drop=True)
        validas = feature.notna() & objetivo.notna()
        if validas.sum() < 3:
            continue
        f, t = feature[validas].to_numpy(), objetivo[validas].to_numpy()

        diferencia = t - f
        escala = max(np.abs(t).max(), 1.0)
        if np.ptp(diferencia) <= EXACT_RELATION_TOLERANCE * escala:
            relacion = "identica" if abs(diferencia[0]) <= EXACT_RELATION_TOLERANCE * escala else "desplazada"
            filas.append({"feature": col, "relacion": relacion, "constante": float(diferencia[0])})
            continue

        if np.all(np.abs(f) > EXACT_RELATION_TOLERANCE):
            razon = t / f
            if np.ptp(razon) <= EXACT_RELATION_TOLERANCE * max(np.abs(razon).max(), 1.0):
                filas.append({"feature": col, "relacion": "escalada", "constante": float(razon[0])})

    return pd.DataFrame(filas, columns=["feature", "relacion", "constante"])


def train_test_overlap(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str] | None = None,
) -> dict:
    """Cuenta las filas que aparecen idénticas en entrenamiento y en test.

    Es la fuga más grosera y la más fácil de introducir sin darse cuenta: basta
    un `drop_duplicates` que nunca se corrió, o un panel con la misma
    observación cargada dos veces desde dos archivos distintos. El modelo
    memoriza esas filas en entrenamiento y las "predice" perfecto en test, y el
    R² resultante mide memoria, no capacidad de generalizar.

    Los 4 dominios de este repo parten cronológicamente, así que un solapamiento
    debería ser exactamente 0 -- y por eso vale la pena medirlo: si diera
    distinto de cero, el split cronológico está mal armado.
    """
    cols = columns if columns is not None else [c for c in train.columns if c in test.columns]
    claves_train = set(map(tuple, train[cols].astype(str).itertuples(index=False, name=None)))
    claves_test = list(map(tuple, test[cols].astype(str).itertuples(index=False, name=None)))

    solapadas = [k for k in claves_test if k in claves_train]
    return {
        "n_filas_test": len(claves_test),
        "n_solapadas": len(solapadas),
        "proporcion_solapada": len(solapadas) / len(claves_test) if claves_test else 0.0,
        "columnas_comparadas": len(cols),
    }


def leakage_report(
    X_train: pd.DataFrame, y_train, X_test: pd.DataFrame | None = None,
    columns: list[str] | None = None,
) -> dict:
    """Corre los tres chequeos juntos y devuelve un veredicto legible.

    El veredicto es deliberadamente `revisar` y no `fuga detectada` cuando lo
    único que aparece es un R² individual alto: ese resultado no distingue una
    fuga de una autocorrelación legítima, y afirmar lo primero convertiría una
    herramienta de diagnóstico en una fuente de falsos positivos que se terminan
    ignorando. `fuga` se afirma solo ante evidencia determinística: una relación
    exacta con el target, o filas compartidas entre train y test.
    """
    poder = single_feature_power(X_train, y_train, columns)
    exactas = exact_relations(X_train, y_train, columns)
    solapamiento = train_test_overlap(X_train, X_test, columns) if X_test is not None else None

    sospechosas = poder[poder["sospechosa"]]["feature"].tolist() if not poder.empty else []
    hay_fuga = not exactas.empty or bool(solapamiento and solapamiento["n_solapadas"])

    if hay_fuga:
        veredicto = "fuga"
    elif sospechosas:
        veredicto = "revisar"
    else:
        veredicto = "sin senales"

    return {
        "veredicto": veredicto,
        "features_sospechosas": sospechosas,
        "relaciones_exactas": exactas.to_dict("records"),
        "solapamiento_train_test": solapamiento,
        "poder_individual": poder,
    }
