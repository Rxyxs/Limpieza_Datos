"""Pruebas de humo sobre los artefactos REALES ya generados por el pipeline
(`fetch` -> `clean` -> `features` -> `model`) -- no mockeadas. Se saltan si el
pipeline todavía no corrió (el Excel de 80MB y el warehouse no se comitean)."""
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "consulting"
PROCESSED_DIR = ROOT / "data" / "processed" / "consulting"
REPORTS_DIR = ROOT / "outputs" / "consulting"
WAREHOUSE_PATH = PROCESSED_DIR / "wdi_warehouse.duckdb"

pytestmark = pytest.mark.skipif(
    not WAREHOUSE_PATH.exists(), reason="corre fetch.py + clean.py + features.py + model.py primero"
)


def test_curated_raw_extract_has_real_indicators():
    from src.domains.consulting_excel_dwh.fetch import CURATED_INDICATORS

    wide = pd.read_csv(RAW_DIR / "wdi_curated_wide.csv")
    assert set(wide["Indicator Code"].unique()) == set(CURATED_INDICATORS.keys())
    assert len(wide) > 1000


def test_warehouse_has_the_three_star_schema_tables():
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    con.close()
    assert {"dim_country", "dim_indicator", "fact_indicator_value"} <= tables


def test_dim_country_excludes_regional_aggregates():
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    countries = con.execute("SELECT country_code, region FROM dim_country").fetchdf()
    con.close()
    assert countries["region"].notna().all()
    assert "WLD" not in countries["country_code"].values  # "World" es un agregado, no un pais


def test_fact_table_life_expectancy_is_in_a_plausible_real_range():
    # El piso real es mucho mas bajo de lo intuitivo: Camboya 1976-78 (Jemeres
    # Rojos) y Ruanda 1994 (genocidio) registran esperanza de vida ~11-12 anios
    # en esta misma tabla -- eventos historicos reales documentados, no un
    # error de datos, asi que el limite inferior del test tiene que dejarlos
    # pasar en vez de asumir un piso "razonable" que en realidad no lo es.
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    life_exp = con.execute(
        "SELECT valor FROM fact_indicator_value WHERE indicator_code = 'SP.DYN.LE00.IN'"
    ).fetchdf()["valor"]
    con.close()
    assert life_exp.between(8, 90).all()


def test_features_have_no_nulls_in_modeling_columns():
    from src.domains.consulting_excel_dwh.features import FEATURE_COLUMNS, TARGET_COLUMN

    features_df = pd.read_csv(PROCESSED_DIR / "consulting_features.csv")
    assert features_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum() == 0
    assert len(features_df) > 1000


def test_best_model_beats_baseline_by_a_real_margin():
    import json

    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    baseline_r2 = metrics["results"]["baseline_media"]["r2"]
    best_r2 = max(metrics["results"]["mlp_pytorch"]["r2"], metrics["results"]["xgboost"]["r2"])
    assert best_r2 > baseline_r2 + 0.5  # margen real y grande, no un empate casual


def test_mlp_trained_at_least_100_epochs():
    import json

    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["results"]["mlp_pytorch"]["epochs_run"] >= 100
