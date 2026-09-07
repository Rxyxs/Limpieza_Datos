"""Tres familias de modelos adicionales, reusables sin cambios por los 4
dominios, para el mismo objetivo que ya resuelven su baseline, su MLP y su
XGBoost:

1. `fit_elasticnet` -- lineal regularizado (Ridge / ElasticNet / Lasso). Es el
   modelo que más seguido gana cuando el dataset es realmente chico, y el único
   de los seis que deja leer directamente qué features usa y con qué signo.
2. `fit_random_forest` -- bagging de árboles. Contraste directo contra el
   boosting de XGBoost: misma materia prima (árboles), estrategia opuesta
   (promediar muchos árboles profundos e independientes en vez de encadenar
   muchos árboles débiles que corrigen al anterior).
3. `fit_lstm` -- red recurrente que lee la ventana de meses/años previos como
   SECUENCIA, no como columnas de lag aplanadas. Es la única de las seis que
   ve el orden temporal como tal.

**Selección de hiperparámetros sobre el split de validación, nunca con
`GridSearchCV`.** La validación cruzada por defecto de scikit-learn baraja las
filas, y barajar una serie de tiempo entrena con datos posteriores a los que
después evalúa: el modelo "aprende" el futuro y el score de validación queda
inflado sin que nada falle visiblemente. Acá cada candidato se ajusta en train
y se elige por RMSE en el mismo split de validación cronológico que ya usan el
MLP y XGBoost, así que los seis modelos se comparan bajo exactamente la misma
regla.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from torch import nn

from src.toolkit.torch_trainer import train_with_early_stopping

DEFAULT_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_L1_RATIOS = (0.0, 0.5, 1.0)  # 0.0 = Ridge puro, 1.0 = Lasso puro
DEFAULT_RF_DEPTHS = (None, 4, 8, 16)
DEFAULT_RF_MIN_LEAF = (1, 2, 5)


@dataclass
class ModelFit:
    """Resultado de ajustar un modelo: predicciones sobre test, en las unidades
    reales del target, más los metadatos que el dominio publica en `metrics.json`."""

    predictions: np.ndarray
    metadata: dict = field(default_factory=dict)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# 1. Lineal regularizado
# ---------------------------------------------------------------------------

def fit_elasticnet(
    X_train, y_train, X_val, y_val, X_test,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    l1_ratios: tuple[float, ...] = DEFAULT_L1_RATIOS,
    max_iter: int = 20000,
) -> ModelFit:
    """Ajusta una malla de modelos lineales regularizados y devuelve el mejor
    según RMSE de validación.

    `l1_ratio=0` se resuelve con `Ridge` y no con `ElasticNet(l1_ratio=0)`:
    scikit-learn acepta el segundo pero advierte y converge peor, porque el
    solver de coordinate descent de ElasticNet no está pensado para el caso
    puramente L2, que tiene solución cerrada.

    **Las features tienen que llegar ya escaladas** (z-score con media/desvío de
    TRAIN). La penalización es sobre la magnitud de los coeficientes, así que
    una feature medida en miles se penaliza mucho más que la misma información
    medida en unidades: sin escalar, la regularización no elige las features
    más informativas sino las de escala más grande.

    **`alphas` se interpreta relativo al desvío estándar del target**, no en
    términos absolutos. El `alpha` de scikit-learn penaliza contra la escala de
    `y`: los cuatro dominios de este repo tienen targets que van de ~0.005
    (retorno diario del dólar) a ~5.000 (kg/hectárea), así que una misma malla
    absoluta significaría "sin regularizar" en uno y "todos los coeficientes a
    cero" en el otro. Multiplicar por `std(y_train)` hace que la malla cubra el
    mismo rango real de regularización en los cuatro -- y de paso resuelve que
    el coordinate descent no converja cuando la penalización efectiva es
    despreciable frente a la escala del target.

    `n_coeficientes_no_nulos` en los metadatos es lo que hace útil a este modelo
    aun cuando pierde: con `l1_ratio` alto, el propio ajuste dice cuántas de las
    features sobreviven, que es una respuesta directa a "¿cuánta señal real hay
    acá?" que ni el MLP ni los árboles dan.
    """
    target_scale = float(np.std(y_train))
    target_scale = target_scale if target_scale > 1e-12 else 1.0

    best_fit, best_rmse, best_params = None, float("inf"), {}

    for alpha in alphas:
        effective_alpha = alpha * target_scale
        for l1_ratio in l1_ratios:
            model = (
                Ridge(alpha=effective_alpha)
                if l1_ratio <= 0
                else ElasticNet(
                    alpha=effective_alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=0
                )
            )
            model.fit(X_train, y_train)
            rmse = _rmse(y_val, model.predict(X_val))
            if rmse < best_rmse:
                best_fit, best_rmse = model, rmse
                best_params = {
                    "alpha_relativo": float(alpha),
                    "alpha_efectivo": float(effective_alpha),
                    "l1_ratio": float(l1_ratio),
                }

    coefficients = np.asarray(best_fit.coef_).ravel()
    return ModelFit(
        predictions=np.asarray(best_fit.predict(X_test), dtype=float),
        metadata={
            **best_params,
            "val_rmse": best_rmse,
            "n_coeficientes": int(coefficients.size),
            "n_coeficientes_no_nulos": int(np.sum(np.abs(coefficients) > 1e-10)),
        },
    )


# ---------------------------------------------------------------------------
# 2. Bagging de árboles
# ---------------------------------------------------------------------------

def fit_random_forest(
    X_train, y_train, X_val, y_val, X_test,
    n_estimators: int = 300,
    max_depths: tuple = DEFAULT_RF_DEPTHS,
    min_samples_leaf: tuple[int, ...] = DEFAULT_RF_MIN_LEAF,
    random_state: int = 42,
    feature_names: list[str] | None = None,
) -> ModelFit:
    """Ajusta un Random Forest eligiendo profundidad y hoja mínima por RMSE de
    validación, y devuelve sus predicciones sobre test.

    Se le pasan las features **sin escalar**, igual que a XGBoost: un árbol
    parte por umbrales sobre cada variable por separado, así que un cambio de
    escala solo mueve el umbral, nunca la partición elegida.

    El valor de tener este modelo al lado de XGBoost no es ganarle, es que
    cuando el bosque le gana al boosting sobre las mismas features eso dice algo
    concreto: el boosting estaba sobreajustando la señal que quedaba, porque
    promediar árboles independientes (bagging) reduce varianza sin reducir
    sesgo, y encadenar árboles (boosting) hace exactamente lo contrario.
    """
    best_model, best_rmse, best_params = None, float("inf"), {}

    for depth in max_depths:
        for leaf in min_samples_leaf:
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=depth, min_samples_leaf=leaf,
                random_state=random_state, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            rmse = _rmse(y_val, model.predict(X_val))
            if rmse < best_rmse:
                best_model, best_rmse = model, rmse
                best_params = {"max_depth": depth, "min_samples_leaf": int(leaf)}

    metadata = {
        **best_params,
        "n_estimators": int(n_estimators),
        "val_rmse": best_rmse,
    }
    if feature_names is not None:
        ranked = sorted(zip(feature_names, best_model.feature_importances_), key=lambda p: -p[1])
        metadata["features_mas_importantes"] = [
            {"feature": name, "importancia": round(float(value), 4)} for name, value in ranked[:5]
        ]

    return ModelFit(
        predictions=np.asarray(best_model.predict(X_test), dtype=float),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 3. Red recurrente sobre secuencias
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    window: int,
    time_column: str,
    group_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Arma, para cada fila de `df`, la secuencia de las `window` observaciones
    que la preceden (incluyéndola). Devuelve `(secuencias, completas)`:

    - `secuencias`: array `(len(df), window, n_features)`, alineado **fila a
      fila con `df`**, para que el dominio pueda cortarlo con exactamente los
      mismos índices de train/val/test que ya usa para los otros modelos.
    - `completas`: máscara booleana de las filas que tienen las `window`
      observaciones reales. Las que no (las primeras de cada serie) igual
      reciben una secuencia, rellenada repitiendo su observación más antigua
      disponible.

    **Toda fila recibe secuencia, y por eso se rellena en vez de descartarse.**
    Si el LSTM se evaluara sobre un test más chico que el de los otros cinco
    modelos -- por haber perdido las filas sin historia suficiente -- su R² no
    sería comparable con el de ellos, y la tabla del README estaría comparando
    modelos sobre conjuntos distintos sin decirlo. Rellenar repitiendo el dato
    más antiguo (y no con ceros, que tras el z-score significan "el valor
    promedio de train", una observación inventada) mantiene el conjunto de
    evaluación idéntico para los seis. Para *entrenar* conviene quedarse solo
    con las filas `completas`; para *predecir*, usarlas todas.

    Con `group_column` las secuencias **nunca cruzan un límite de grupo**: en un
    panel de países, la "historia" de la primera fila de Chile no es la última
    fila de Brasil. Es la misma disciplina que `missing_data.interpolate_within_group`
    aplica a la imputación, acá aplicada a la ventana temporal.
    """
    if window < 1:
        raise ValueError("window tiene que ser >= 1")

    values = df[feature_columns].to_numpy(dtype=np.float32)
    sequences = np.zeros((len(df), window, len(feature_columns)), dtype=np.float32)
    complete = np.zeros(len(df), dtype=bool)

    sort_columns = ([group_column] if group_column else []) + [time_column]
    ordered_positions = (
        df.reset_index(drop=True).sort_values(sort_columns, kind="stable").index.to_numpy()
    )
    groups = df[group_column].to_numpy() if group_column else np.zeros(len(df), dtype=np.int8)

    history: list[int] = []
    previous_group = object()
    for position in ordered_positions:
        current_group = groups[position]
        if current_group != previous_group:
            history = []
            previous_group = current_group

        history.append(position)
        if len(history) > window:
            history.pop(0)

        padding = [history[0]] * (window - len(history))
        sequences[position] = values[padding + history]
        complete[position] = len(history) == window

    return sequences, complete


class SequenceLSTM(nn.Module):
    """LSTM de una capa + cabezal lineal sobre el último paso de la secuencia.

    Una sola capa y pocas unidades a propósito: los cuatro dominios de este repo
    tienen entre 137 y ~10.000 filas de entrenamiento, tres órdenes de magnitud
    menos de lo que necesita una recurrente profunda para no sobreajustar. Un
    modelo más grande daría una curva de entrenamiento más bonita y un test peor.
    """

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        output, _state = self.lstm(x)
        return self.head(output[:, -1, :])


def fit_lstm(
    seq_train: np.ndarray, y_train: np.ndarray,
    seq_val: np.ndarray, y_val: np.ndarray,
    seq_test: np.ndarray,
    hidden: int = 32,
    dropout: float = 0.1,
    max_epochs: int = 400,
    min_epochs: int = 100,
    patience: int = 25,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> ModelFit:
    """Entrena la LSTM con el mismo loop de early stopping que usan los MLP de
    los 4 dominios (piso de `min_epochs`, restauración del mejor checkpoint) y
    devuelve sus predicciones sobre test, ya en unidades reales del target.

    **El target se z-scorea internamente** (media y desvío de TRAIN únicamente) y
    se revierte antes de devolver. No es un detalle de estilo: es el bug que este
    proyecto ya documentó dos veces en dominios distintos -- una red inicializada
    cerca de cero contra un target de escala ~450 (minería) o ~70 (esperanza de
    vida) converge a una constante y da un R² catastrófico, y el síntoma no
    apunta a la escala sino a la arquitectura. Dejarlo adentro del trainer
    significa que ningún dominio nuevo puede volver a pisarlo.

    Las features, en cambio, tienen que llegar escaladas desde afuera, con la
    misma media/desvío de train que usan el MLP y el lineal, para que los seis
    modelos vean exactamente la misma entrada.
    """
    torch.manual_seed(seed)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    y_std = y_std if y_std > 1e-9 else 1.0

    X_train_t = torch.tensor(seq_train, dtype=torch.float32)
    y_train_t = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(seq_val, dtype=torch.float32)
    y_val_t = torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(seq_test, dtype=torch.float32)

    model = SequenceLSTM(n_features=seq_train.shape[2], hidden=hidden, dropout=dropout)
    result = train_with_early_stopping(
        model, X_train_t, y_train_t, X_val_t, y_val_t, loss_fn=nn.MSELoss(),
        max_epochs=max_epochs, min_epochs=min_epochs, patience=patience,
        lr=lr, weight_decay=weight_decay,
    )

    model.eval()
    with torch.no_grad():
        scaled = model(X_test_t).numpy().ravel()

    return ModelFit(
        predictions=scaled * y_std + y_mean,
        metadata={
            "window": int(seq_train.shape[1]),
            "hidden": int(hidden),
            "epochs_run": result.epochs_run,
            "best_epoch": result.best_epoch,
            "early_stopped": result.early_stopped,
            "n_secuencias_train": int(seq_train.shape[0]),
        },
    )


# ---------------------------------------------------------------------------
# Lectura del ganador
# ---------------------------------------------------------------------------

# Cómo se llama, dentro del `metrics.json` de cada dominio, el vector de
# predicciones de cada modelo.
PREDICTION_KEYS = {
    "mlp_pytorch": "mlp_pred",
    "xgboost": "xgb_pred",
    "elasticnet": "elasticnet_pred",
    "random_forest": "rf_pred",
    "lstm": "lstm_pred",
}


def best_model_predictions(metrics: dict) -> tuple[str, list]:
    """Devuelve `(nombre, predicciones)` del modelo con mejor R² real en test.

    Los baselines entran en la comparación como cualquier otro modelo. Excluirlos
    "porque no son modelos de verdad" haría que el gráfico de diagnóstico de un
    dominio donde el baseline gana mostrara igual a un modelo peor, con el
    rótulo de mejor -- exactamente el maquillaje que este proyecto no hace.
    """
    results = metrics["results"]
    best = max(results, key=lambda name: results[name]["r2"])
    key = PREDICTION_KEYS.get(best, "baseline_pred")
    return best, metrics[key]
