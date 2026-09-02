"""Pruebas de humo (smoke tests) para el dominio `agriculture_worldbank`,
contra los datos reales ya descargados/procesados por el pipeline
(`fetch.py` -> `clean.py` -> `features.py` -> `model.py`). No hay mocks: si
los artefactos no existen, correr el pipeline primero:

    .venv\\Scripts\\python.exe -m src.domains.agriculture_worldbank.fetch
    .venv\\Scripts\\python.exe -m src.domains.agriculture_worldbank.clean
    .venv\\Scripts\\python.exe -m src.domains.agriculture_worldbank.features
    .venv\\Scripts\\python.exe -m src.domains.agriculture_worldbank.model
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.domains.agriculture_worldbank.features import COUNTRY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.domains.agriculture_worldbank.fetch import COUNTRIES, INDICATORS, RAW_DIR

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / "agriculture"
REPORTS_DIR = ROOT / "outputs" / "agriculture"

# Rango real plausible de rendimiento de cereales (kg/hectárea) para estos 9
# países sudamericanos en 1990-2025, según literatura agro-económica.
PLAUSIBLE_YIELD_MIN = 500
PLAUSIBLE_YIELD_MAX = 8000


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"artefacto real no encontrado: {path} -- correr el pipeline del dominio primero")
    return path


# ---------------------------------------------------------------------------
# fetch.py: disposición real de los datos crudos
# ---------------------------------------------------------------------------

def test_raw_indicator_files_cover_all_9_countries():
    for slug in INDICATORS.values():
        raw = pd.read_csv(_require(RAW_DIR / f"{slug}.csv"))
        countries_present = set(raw["country_iso3"].unique())
        assert countries_present == set(COUNTRIES), f"{slug}: países presentes {countries_present} != esperados {set(COUNTRIES)}"


def test_raw_indicator_files_span_plausible_year_range():
    for slug in INDICATORS.values():
        raw = pd.read_csv(_require(RAW_DIR / f"{slug}.csv"))
        assert raw["year"].min() <= 1995, f"{slug}: no cubre décadas tempranas reales (min={raw['year'].min()})"
        assert raw["year"].max() >= 2020, f"{slug}: no cubre años recientes reales (max={raw['year'].max()})"


def test_irrigated_land_has_genuine_sparse_coverage_including_peru_fully_missing():
    """Documenta el hallazgo real de missingness más severo del dominio: Perú
    no tiene NI UN SOLO valor real de tierra irrigada en 1990-2025 en la API
    pública del Banco Mundial. Si este assert alguna vez falla porque el
    Banco Mundial retro-publicó datos para Perú, es una mejora real de la
    fuente, no un bug -- habría que actualizar el comentario en clean.py.
    """
    raw = pd.read_csv(_require(RAW_DIR / "irrigated_land_pct.csv"))
    peru = raw[raw["country_iso3"] == "PER"]
    assert peru["value"].notna().sum() == 0
    # Cobertura genuinamente rala en todo el panel (no es un caso aislado de Perú).
    assert raw["value"].notna().mean() < 0.25


# ---------------------------------------------------------------------------
# clean.py: el panel final no debe tener nulos en las columnas de modelado
# ---------------------------------------------------------------------------

def test_clean_panel_has_no_remaining_nulls_in_modeling_columns():
    panel = pd.read_csv(_require(PROCESSED_DIR / "agriculture_panel_clean.csv"))
    indicator_columns = list(INDICATORS.values())
    assert panel[indicator_columns].isna().sum().sum() == 0


def test_clean_panel_is_a_balanced_country_year_grid():
    panel = pd.read_csv(_require(PROCESSED_DIR / "agriculture_panel_clean.csv"))
    assert panel["country_iso3"].nunique() == 9
    counts_per_country = panel.groupby("country_iso3").size()
    assert (counts_per_country == counts_per_country.iloc[0]).all(), "el panel debería ser un grid país-año balanceado"


def test_cereal_yield_is_within_plausible_real_range():
    panel = pd.read_csv(_require(PROCESSED_DIR / "agriculture_panel_clean.csv"))
    yields = panel["cereal_yield_kg_ha"]
    assert yields.min() >= PLAUSIBLE_YIELD_MIN, f"rendimiento mínimo implausible: {yields.min()}"
    assert yields.max() <= PLAUSIBLE_YIELD_MAX, f"rendimiento máximo implausible: {yields.max()}"


# ---------------------------------------------------------------------------
# features.py: sin nulos, sin look-ahead evidente en los lags
# ---------------------------------------------------------------------------

def test_features_table_has_no_nulls_in_feature_or_target_columns():
    features_df = pd.read_csv(_require(PROCESSED_DIR / "agriculture_features.csv"))
    assert features_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum() == 0


def test_features_table_has_all_9_country_dummies_and_they_are_mutually_exclusive():
    features_df = pd.read_csv(_require(PROCESSED_DIR / "agriculture_features.csv"))
    assert set(COUNTRY_COLUMNS).issubset(features_df.columns)
    # cada fila pertenece a exactamente un país (one-hot real, no corrupto).
    assert (features_df[COUNTRY_COLUMNS].sum(axis=1) == 1).all()


def test_yield_lag1_matches_previous_year_within_same_country_no_leakage():
    """Chequeo directo de la disciplina no-look-ahead: el lag1 de una fila
    debe ser igual al rendimiento REAL del mismo país en el año anterior
    (tomado del panel limpio, no del propio archivo de features)."""
    panel = pd.read_csv(_require(PROCESSED_DIR / "agriculture_panel_clean.csv"))
    features_df = pd.read_csv(_require(PROCESSED_DIR / "agriculture_features.csv"))

    panel_lookup = panel.set_index(["country_iso3", "year"])["cereal_yield_kg_ha"]
    country_col = features_df[COUNTRY_COLUMNS].idxmax(axis=1).str.replace("country_iso3_", "", regex=False)

    sample = features_df.sample(n=min(20, len(features_df)), random_state=42)
    for idx in sample.index:
        country, year = country_col.loc[idx], int(features_df.loc[idx, "year"])
        expected_prev_yield = panel_lookup.loc[(country, year - 1)]
        assert features_df.loc[idx, "cereal_yield_lag1"] == pytest.approx(expected_prev_yield)


# ---------------------------------------------------------------------------
# model.py: el reclamo real del modelo
# ---------------------------------------------------------------------------

def test_model_metrics_file_has_expected_shape():
    metrics = json.loads(_require(REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["results"].keys()) == {"baseline_media_pais", "mlp_pytorch", "xgboost"}
    for name, m in metrics["results"].items():
        assert {"r2", "rmse", "mae"}.issubset(m.keys())


def test_mlp_trained_at_least_100_epochs():
    metrics = json.loads(_require(REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["results"]["mlp_pytorch"]["epochs_run"] >= 100
    assert len(metrics["train_losses"]) >= 100


def test_real_drivers_beat_per_country_baseline_by_a_real_margin():
    """Reclamo real medido: a diferencia del dominio financiero (retorno
    cambiario, señal débil por diseño), acá hay drivers agro-económicos
    genuinos (fertilizante, irrigación, tierra arable) y ambos modelos
    entrenados superan claramente al baseline de media por país -- no un
    empate honesto, una mejora real y grande."""
    metrics = json.loads(_require(REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    baseline_r2 = metrics["results"]["baseline_media_pais"]["r2"]
    mlp_r2 = metrics["results"]["mlp_pytorch"]["r2"]
    xgb_r2 = metrics["results"]["xgboost"]["r2"]

    assert mlp_r2 > baseline_r2 + 0.5
    assert xgb_r2 > baseline_r2 + 0.5
    # Resultado meaningfully positivo (no solo "mejor que un baseline débil").
    assert mlp_r2 > 0.5
    assert xgb_r2 > 0.5


def test_mlp_did_not_collapse_to_dying_relu_failure_mode():
    """Sanity check explícito contra el bug documentado en model.py: un R²
    catastróficamente negativo (el síntoma medido del colapso dying-ReLU en
    el dominio financiero de referencia fue -8746) señalaría que el fix
    (LeakyReLU) no se aplicó correctamente acá."""
    metrics = json.loads(_require(REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["results"]["mlp_pytorch"]["r2"] > -1.0
