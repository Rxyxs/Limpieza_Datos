"""Pruebas unitarias para src/toolkit/drift.py."""
import numpy as np
import pandas as pd

from src.toolkit.drift import (
    PSI_SEVERE,
    categorical_drift,
    classify_psi,
    drift_report,
    population_stability_index,
    target_shift,
)


def test_psi_is_near_zero_for_two_samples_of_the_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)

    assert population_stability_index(a, b) < 0.02


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(size=5000)
    poco = population_stability_index(base, rng.normal(loc=0.3, size=5000))
    mucho = population_stability_index(base, rng.normal(loc=1.5, size=5000))

    assert poco < mucho
    assert mucho > PSI_SEVERE


def test_psi_bins_come_from_the_reference_sample():
    # Si cada muestra se bineara por sus propios cuantiles, ambas quedarian con
    # ~10% por bin y el PSI daria ~0 justo cuando mas se movio la distribucion.
    rng = np.random.default_rng(2)
    vieja = rng.normal(loc=0, scale=1, size=4000)
    nueva = rng.normal(loc=0, scale=4, size=4000)  # misma media, otra dispersion

    assert population_stability_index(vieja, nueva) > PSI_SEVERE


def test_psi_is_zero_for_a_constant_column():
    # Sin distribucion que comparar, la respuesta honesta es 0, no un error.
    assert population_stability_index([5.0] * 100, [5.0] * 100) == 0.0


def test_psi_handles_bins_empty_in_one_sample():
    # Un bin sin observaciones daria log(0) = -inf sin el epsilon.
    valor = population_stability_index(np.arange(1000), np.arange(1000) + 5000)
    assert np.isfinite(valor)
    assert valor > PSI_SEVERE


def test_categorical_drift_reports_new_and_missing_categories():
    resultado = categorical_drift(
        ["a", "a", "b", "b", "c"],
        ["a", "b", "b", "d", "d"],
    )

    assert resultado["categorias_nuevas"] == ["d"]
    assert resultado["categorias_ausentes"] == ["c"]
    assert 0 < resultado["distancia_variacion_total"] <= 1


def test_categorical_drift_is_zero_for_identical_distributions():
    resultado = categorical_drift(["a", "b", "b"], ["a", "b", "b"])

    assert resultado["distancia_variacion_total"] == 0.0
    assert resultado["categorias_nuevas"] == []


def test_classify_psi_maps_to_the_three_conventional_bands():
    assert classify_psi(0.05) == "estable"
    assert classify_psi(0.15) == "moderado"
    assert classify_psi(0.40) == "severo"
    assert classify_psi(float("nan")) == "sin datos"


def test_drift_report_sorts_the_worst_column_first():
    rng = np.random.default_rng(3)
    vieja = pd.DataFrame({
        "estable": rng.normal(size=2000),
        "movida": rng.normal(size=2000),
        "categoria": rng.choice(["x", "y"], size=2000),
    })
    nueva = pd.DataFrame({
        "estable": rng.normal(size=2000),
        "movida": rng.normal(loc=2.0, size=2000),
        "categoria": rng.choice(["x", "y"], size=2000),
    })
    reporte = drift_report(vieja, nueva)

    assert reporte.iloc[0]["columna"] == "movida"
    assert reporte.iloc[0]["veredicto"] == "severo"
    assert reporte.set_index("columna").loc["estable", "veredicto"] == "estable"
    assert reporte.set_index("columna").loc["categoria", "tipo"] == "categorica"


def test_drift_report_flags_a_category_that_did_not_exist_before():
    vieja = pd.DataFrame({"region": ["norte"] * 50 + ["sur"] * 50})
    nueva = pd.DataFrame({"region": ["norte"] * 40 + ["sur"] * 40 + ["centro"] * 20})
    reporte = drift_report(vieja, nueva)

    assert reporte.iloc[0]["categorias_nuevas"] == 1


def test_drift_report_ignores_columns_absent_from_the_new_sample():
    vieja = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    nueva = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    assert list(drift_report(vieja, nueva)["columna"]) == ["a"]


def test_target_shift_explains_a_negative_baseline_r2():
    # Serie con tendencia al alza: el promedio del periodo viejo queda por
    # debajo de todo el periodo nuevo, y "predecir la media vieja" da R2 < 0.
    viejo = np.arange(100, dtype=float)
    nuevo = np.arange(100, 140, dtype=float)
    resultado = target_shift(viejo, nuevo)

    assert resultado["media_actual"] > resultado["media_expected"]
    assert resultado["desplazamiento_en_desvios"] > 1
    assert resultado["r2_de_predecir_la_media_vieja"] < 0


def test_target_shift_is_neutral_when_nothing_moved():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=3000), rng.normal(size=3000)
    resultado = target_shift(a, b)

    assert abs(resultado["desplazamiento_en_desvios"]) < 0.15
    assert resultado["r2_de_predecir_la_media_vieja"] > -0.05
