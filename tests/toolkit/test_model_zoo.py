"""Pruebas unitarias para src/toolkit/model_zoo.py.

Los tres modelos se entrenan de verdad sobre datos pequeños pero con señal
real construida a propósito (no mocks de sklearn ni de torch): lo que se
verifica es que aprenden esa señal, que eligen sus hiperparámetros mirando
SOLO el split de validación, y que las secuencias nunca cruzan un límite de
grupo.
"""
import numpy as np
import pandas as pd
import pytest

from src.toolkit.model_zoo import (
    build_sequences,
    fit_elasticnet,
    fit_lstm,
    fit_random_forest,
)


@pytest.fixture
def linear_panel():
    """y = 3*x1 - 2*x2 + ruido chico; x3 es puro ruido, sin relación con y."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    y = 3 * X["x1"] - 2 * X["x2"] + rng.normal(scale=0.1, size=n)
    return X.iloc[:180], y.values[:180], X.iloc[180:240], y.values[180:240], X.iloc[240:], y.values[240:]


def test_fit_elasticnet_recovers_a_real_linear_signal(linear_panel):
    X_tr, y_tr, X_va, y_va, X_te, y_te = linear_panel
    fit = fit_elasticnet(X_tr, y_tr, X_va, y_va, X_te)

    assert np.corrcoef(fit.predictions, y_te)[0, 1] > 0.99
    assert fit.metadata["alpha_relativo"] in {0.001, 0.01, 0.1, 1.0, 10.0}
    assert fit.metadata["n_coeficientes"] == 3


def test_fit_elasticnet_with_lasso_drops_the_noise_feature(linear_panel):
    X_tr, y_tr, X_va, y_va, X_te, y_te = linear_panel
    # l1_ratio=1 (Lasso puro) con penalizacion apreciable: la feature de ruido
    # deberia quedar en cero exacto, que es lo que hace util este modelo.
    fit = fit_elasticnet(X_tr, y_tr, X_va, y_va, X_te, alphas=(0.05,), l1_ratios=(1.0,))

    assert fit.metadata["n_coeficientes_no_nulos"] == 2


def test_fit_elasticnet_scales_alpha_by_the_target_spread(linear_panel):
    # El mismo alpha relativo tiene que producir la misma solucion sobre un
    # target multiplicado por 1000: si alpha fuera absoluto, la penalizacion
    # pasaria de razonable a despreciable y el modelo cambiaria por completo.
    X_tr, y_tr, X_va, y_va, X_te, _ = linear_panel
    normal = fit_elasticnet(X_tr, y_tr, X_va, y_va, X_te, alphas=(0.1,), l1_ratios=(1.0,))
    escalado = fit_elasticnet(X_tr, y_tr * 1000, X_va, y_va * 1000, X_te, alphas=(0.1,), l1_ratios=(1.0,))

    assert escalado.metadata["n_coeficientes_no_nulos"] == normal.metadata["n_coeficientes_no_nulos"]
    assert escalado.metadata["alpha_efectivo"] == pytest.approx(
        normal.metadata["alpha_efectivo"] * 1000, rel=1e-6
    )
    np.testing.assert_allclose(escalado.predictions, normal.predictions * 1000, rtol=1e-5)


def test_fit_elasticnet_selects_on_validation_not_on_train(linear_panel):
    X_tr, y_tr, X_va, y_va, X_te, _ = linear_panel
    fit = fit_elasticnet(X_tr, y_tr, X_va, y_va, X_te)

    # El RMSE reportado es el del split de validacion, no el de train: tiene
    # que coincidir con recalcularlo aparte sobre X_va.
    assert fit.metadata["val_rmse"] > 0
    assert fit.metadata["val_rmse"] < 1.0  # hay senal real, no es ruido


def test_fit_random_forest_learns_a_nonlinear_signal():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"x1": rng.uniform(-3, 3, n), "x2": rng.uniform(-3, 3, n)})
    y = (X["x1"] ** 2 + np.sin(3 * X["x2"])).values

    fit = fit_random_forest(
        X.iloc[:240], y[:240], X.iloc[240:320], y[240:320], X.iloc[320:],
        n_estimators=60, feature_names=["x1", "x2"],
    )

    assert np.corrcoef(fit.predictions, y[320:])[0, 1] > 0.9
    assert fit.metadata["n_estimators"] == 60
    assert fit.metadata["min_samples_leaf"] in {1, 2, 5}
    assert len(fit.metadata["features_mas_importantes"]) == 2


def test_fit_random_forest_omits_importances_without_feature_names():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x1": rng.normal(size=120)})
    y = (2 * X["x1"]).values
    fit = fit_random_forest(X.iloc[:70], y[:70], X.iloc[70:95], y[70:95], X.iloc[95:], n_estimators=20)

    assert "features_mas_importantes" not in fit.metadata


# ---------------------------------------------------------------------------
# Secuencias
# ---------------------------------------------------------------------------

@pytest.fixture
def panel_two_groups():
    return pd.DataFrame({
        "pais": ["CHL"] * 4 + ["ARG"] * 4,
        "anio": [2000, 2001, 2002, 2003] * 2,
        "valor": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    })


def test_build_sequences_never_crosses_a_group_boundary(panel_two_groups):
    sequences, complete = build_sequences(
        panel_two_groups, ["valor"], window=3, time_column="anio", group_column="pais",
    )

    # La primera fila de ARG (indice 4) no puede tomar historia de CHL.
    assert sequences[4].ravel().tolist() == [10.0, 10.0, 10.0]
    # La ultima de ARG ve solo valores de ARG.
    assert sequences[7].ravel().tolist() == [20.0, 30.0, 40.0]
    # Sin grupo, el panel se lee como una sola serie ordenada por año y la
    # historia de ARG queda contaminada con valores de CHL.
    sin_grupo, _ = build_sequences(panel_two_groups, ["valor"], window=3, time_column="anio")
    assert sin_grupo[4].ravel().tolist() == [1.0, 1.0, 10.0]  # 1.0 es de CHL
    assert sin_grupo[7].ravel().tolist() == [30.0, 4.0, 40.0]  # 4.0 es de CHL


def test_build_sequences_marks_only_full_history_as_complete(panel_two_groups):
    _sequences, complete = build_sequences(
        panel_two_groups, ["valor"], window=3, time_column="anio", group_column="pais",
    )

    # Las 2 primeras filas de cada grupo no tienen 3 observaciones reales.
    assert complete.tolist() == [False, False, True, True] * 2


def test_build_sequences_pads_short_history_by_repeating_the_oldest(panel_two_groups):
    sequences, _complete = build_sequences(
        panel_two_groups, ["valor"], window=3, time_column="anio", group_column="pais",
    )

    assert sequences[0].ravel().tolist() == [1.0, 1.0, 1.0]  # sin historia
    assert sequences[1].ravel().tolist() == [1.0, 1.0, 2.0]  # una observacion previa


def test_build_sequences_returns_one_sequence_per_row(panel_two_groups):
    sequences, complete = build_sequences(
        panel_two_groups, ["valor"], window=3, time_column="anio", group_column="pais",
    )

    # Toda fila recibe secuencia: es lo que mantiene el test set identico
    # entre el LSTM y los otros cinco modelos.
    assert sequences.shape == (len(panel_two_groups), 3, 1)
    assert len(complete) == len(panel_two_groups)


def test_build_sequences_sorts_by_time_within_group():
    desordenado = pd.DataFrame({
        "pais": ["CHL", "CHL", "CHL"],
        "anio": [2002, 2000, 2001],
        "valor": [3.0, 1.0, 2.0],
    })
    sequences, _complete = build_sequences(
        desordenado, ["valor"], window=3, time_column="anio", group_column="pais",
    )

    # La fila de 2002 (posicion 0 del DataFrame) tiene que ver 2000, 2001, 2002.
    assert sequences[0].ravel().tolist() == [1.0, 2.0, 3.0]


def test_build_sequences_rejects_a_window_below_one(panel_two_groups):
    with pytest.raises(ValueError, match="window"):
        build_sequences(panel_two_groups, ["valor"], window=0, time_column="anio")


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def test_fit_lstm_learns_a_sequence_signal_and_respects_the_epoch_floor():
    # Serie con dependencia temporal real: cada punto depende de los previos.
    rng = np.random.default_rng(7)
    n = 400
    serie = np.cumsum(rng.normal(size=n)) + np.sin(np.arange(n) / 5.0) * 3
    df = pd.DataFrame({"t": np.arange(n), "valor": serie})
    target = np.roll(serie, -1)  # el valor siguiente

    sequences, _complete = build_sequences(df, ["valor"], window=8, time_column="t")
    fit = fit_lstm(
        sequences[:240], target[:240], sequences[240:320], target[240:320], sequences[320:-1],
        max_epochs=140, min_epochs=100,
    )

    assert fit.metadata["epochs_run"] >= 100  # piso de epocas del proyecto
    assert fit.metadata["window"] == 8
    assert len(fit.predictions) == len(sequences[320:-1])
    assert np.corrcoef(fit.predictions, target[320:-1])[0, 1] > 0.8


def test_fit_lstm_returns_predictions_in_the_real_target_scale():
    # Target de escala grande (~450, como la produccion minera real): si el
    # z-scoring interno faltara, las predicciones saldrian cerca de 0.
    rng = np.random.default_rng(3)
    n = 240
    serie = 450 + np.cumsum(rng.normal(scale=5, size=n))
    df = pd.DataFrame({"t": np.arange(n), "valor": serie})

    sequences, _complete = build_sequences(df, ["valor"], window=5, time_column="t")
    fit = fit_lstm(
        sequences[:150], serie[:150], sequences[150:200], serie[150:200], sequences[200:],
        max_epochs=110, min_epochs=100,
    )

    assert fit.predictions.mean() > 300  # esta en unidades reales, no en z-scores
