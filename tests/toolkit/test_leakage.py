"""Pruebas unitarias para src/toolkit/leakage.py."""
import numpy as np
import pandas as pd
import pytest

from src.toolkit.leakage import (
    exact_relations,
    leakage_report,
    single_feature_power,
    train_test_overlap,
)


@pytest.fixture
def panel_con_fuga():
    rng = np.random.default_rng(0)
    n = 200
    driver = rng.normal(size=n)
    ruido = rng.normal(size=n)
    y = 3 * driver + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({
        "driver": driver,
        "ruido": ruido,
        "target_en_miles": y / 1000,  # el target escalado: fuga exacta
    }), y


def test_single_feature_power_ranks_the_leaking_feature_first(panel_con_fuga):
    X, y = panel_con_fuga
    reporte = single_feature_power(X, y)

    assert reporte.iloc[0]["feature"] == "target_en_miles"
    assert reporte.iloc[0]["r2_individual"] > 0.99
    assert bool(reporte.iloc[0]["sospechosa"]) is True
    assert bool(reporte.set_index("feature").loc["ruido", "sospechosa"]) is False


def test_single_feature_power_keeps_the_sign_of_the_correlation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=150)
    reporte = single_feature_power(pd.DataFrame({"x": x}), -2 * x)

    assert reporte.iloc[0]["correlacion"] < -0.99
    assert reporte.iloc[0]["r2_individual"] > 0.99  # el R2 no distingue el signo


def test_single_feature_power_skips_constant_columns():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"util": rng.normal(size=100), "constante": [7.0] * 100})
    reporte = single_feature_power(X, rng.normal(size=100))

    assert list(reporte["feature"]) == ["util"]


def test_exact_relations_detects_a_scaled_copy_of_the_target(panel_con_fuga):
    X, y = panel_con_fuga
    relaciones = exact_relations(X, y).set_index("feature")

    assert relaciones.loc["target_en_miles", "relacion"] == "escalada"
    assert relaciones.loc["target_en_miles", "constante"] == pytest.approx(1000.0)
    assert "driver" not in relaciones.index


def test_exact_relations_detects_an_offset_copy():
    rng = np.random.default_rng(3)
    y = rng.normal(size=120) * 10
    X = pd.DataFrame({"y_menos_5": y - 5, "otra": rng.normal(size=120)})
    relaciones = exact_relations(X, y).set_index("feature")

    assert relaciones.loc["y_menos_5", "relacion"] == "desplazada"
    assert relaciones.loc["y_menos_5", "constante"] == pytest.approx(5.0)


def test_exact_relations_detects_an_identical_copy():
    rng = np.random.default_rng(4)
    y = rng.normal(size=80)
    relaciones = exact_relations(pd.DataFrame({"copia": y}), y)

    assert relaciones.iloc[0]["relacion"] == "identica"


def test_exact_relations_ignores_a_merely_strong_correlation():
    # Correlacion 0.999 pero NO deterministica: no es una relacion exacta.
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    y = 2 * x + rng.normal(scale=0.01, size=300)

    assert exact_relations(pd.DataFrame({"x": x}), y).empty


def test_train_test_overlap_counts_repeated_rows():
    train = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    test = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})
    resultado = train_test_overlap(train, test)

    assert resultado["n_solapadas"] == 1
    assert resultado["proporcion_solapada"] == pytest.approx(0.5)


def test_train_test_overlap_is_zero_for_a_chronological_split():
    panel = pd.DataFrame({"anio": range(2000, 2020), "valor": range(20)})
    resultado = train_test_overlap(panel.iloc[:15], panel.iloc[15:])

    assert resultado["n_solapadas"] == 0
    assert resultado["proporcion_solapada"] == 0.0


def test_leakage_report_says_fuga_only_with_deterministic_evidence(panel_con_fuga):
    X, y = panel_con_fuga
    reporte = leakage_report(X, y)

    assert reporte["veredicto"] == "fuga"
    assert reporte["relaciones_exactas"][0]["feature"] == "target_en_miles"


def test_leakage_report_says_revisar_for_a_strong_but_legitimate_predictor():
    # Autocorrelacion real (una serie contra su propio rezago): R2 individual
    # altisimo, pero ninguna relacion exacta. El veredicto no puede ser "fuga".
    rng = np.random.default_rng(6)
    serie = np.cumsum(rng.normal(size=400)) + 100
    X = pd.DataFrame({"lag1": serie[:-1]})
    y = serie[1:]
    reporte = leakage_report(X, y)

    assert reporte["veredicto"] == "revisar"
    assert reporte["features_sospechosas"] == ["lag1"]
    assert reporte["relaciones_exactas"] == []


def test_leakage_report_says_sin_senales_on_a_clean_panel():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = 0.5 * X["a"] + rng.normal(size=200)
    reporte = leakage_report(X, y)

    assert reporte["veredicto"] == "sin senales"
    assert reporte["features_sospechosas"] == []


def test_leakage_report_flags_rows_shared_between_train_and_test():
    rng = np.random.default_rng(8)
    X = pd.DataFrame({"a": rng.normal(size=100)})
    y = rng.normal(size=100)
    reporte = leakage_report(X, y, X_test=X.iloc[:10])

    assert reporte["veredicto"] == "fuga"
    assert reporte["solapamiento_train_test"]["n_solapadas"] == 10
