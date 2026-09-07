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
    best_r2 = max(
        m["r2"] for name, m in metrics["results"].items() if not name.startswith("baseline")
    )
    assert best_r2 > baseline_r2 + 0.5  # margen real y grande, no un empate casual


def test_mlp_trained_at_least_100_epochs():
    import json

    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["results"]["mlp_pytorch"]["epochs_run"] >= 100


# ---------------------------------------------------------------------------
# Calidad del dato: integridad del esquema estrella, drift y fuga de target
# ---------------------------------------------------------------------------


def test_candidate_key_search_recovers_the_declared_fact_grain():
    # El grano de la tabla de hechos se redescubre DESDE EL DATO, sin mirar el
    # DDL: si alguna carga futura duplicara una fila, esta busqueda dejaria de
    # encontrar la clave y el test caeria antes de que el duplicado inflara
    # cualquier agregado del warehouse.
    from src.toolkit.keys import find_candidate_keys

    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    hechos = con.execute("SELECT * FROM fact_indicator_value").fetchdf()
    dim = con.execute("SELECT * FROM dim_country").fetchdf()
    con.close()

    assert ("country_code", "indicator_code", "anio") in find_candidate_keys(hechos, max_columns=3)
    assert ("country_code",) in find_candidate_keys(dim, max_columns=1)


def test_star_schema_join_has_real_referential_integrity():
    from src.toolkit.keys import describe_join

    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    hechos = con.execute("SELECT * FROM fact_indicator_value").fetchdf()
    dim = con.execute("SELECT * FROM dim_country").fetchdf()
    con.close()

    diagnostico = describe_join(hechos, dim, on="country_code")
    assert diagnostico["cardinalidad"] == "N:1"
    assert diagnostico["huerfanas_izquierda"] == 0  # ningun hecho sin pais
    assert diagnostico["huerfanas_derecha"] == 0  # ninguna dimension sin uso
    assert diagnostico["factor_multiplicacion"] == 1.0  # el join no multiplica filas


def test_orphan_detection_recovers_the_aggregates_filtered_out_of_the_dimension():
    # Contra la dimension CRUDA del Excel (265 filas, agregados incluidos), las
    # filas huerfanas del lado de la dimension son exactamente los agregados
    # regionales y por ingreso que clean.py descarto -- un numero que sale del
    # join, sin volver a aplicar el filtro.
    from src.toolkit.keys import find_orphans

    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    hechos = con.execute("SELECT * FROM fact_indicator_value").fetchdf()
    n_paises_reales = con.execute("SELECT COUNT(*) FROM dim_country").fetchone()[0]
    con.close()

    crudo = pd.read_csv(RAW_DIR / "wdi_country_dim.csv").rename(columns={"Country Code": "country_code"})
    crudo = crudo[["country_code"]].drop_duplicates()

    _hechos_huerfanos, dim_huerfanas = find_orphans(hechos, crudo, on="country_code")
    assert len(dim_huerfanas) == len(crudo) - n_paises_reales
    assert len(dim_huerfanas) > 0


def test_target_drift_quantitatively_explains_the_negative_baseline():
    """El reclamo mas fuerte de este dominio sobre calidad de dato: el R2
    negativo del baseline NO es un bug, es drift del target medido. El R2 que
    predice `target_shift` a partir solo del desplazamiento de la media
    coincide con el R2 real del baseline publicado en metrics.json."""
    import json

    from src.domains.consulting_excel_dwh.features import TARGET_COLUMN
    from src.domains.consulting_excel_dwh.model import chronological_split
    from src.toolkit.drift import target_shift

    features_df = pd.read_csv(PROCESSED_DIR / "consulting_features.csv")
    train_df, _val_df, test_df = chronological_split(features_df)
    desplazamiento = target_shift(train_df[TARGET_COLUMN], test_df[TARGET_COLUMN])

    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    r2_baseline = metrics["results"]["baseline_media"]["r2"]

    assert desplazamiento["r2_de_predecir_la_media_vieja"] == pytest.approx(r2_baseline, abs=1e-4)
    assert desplazamiento["desplazamiento_en_desvios"] > 0.5  # la esperanza de vida subio
    assert r2_baseline < 0


def test_leakage_check_flags_the_dominant_feature_without_calling_it_a_leak():
    """El Random Forest de este dominio pone 97,2% de su importancia en una sola
    feature. Ese patron tiene la firma numerica de una fuga, y el chequeo lo
    marca -- pero como `revisar` y no como `fuga`, porque no hay ninguna
    relacion deterministica con el target: la esperanza de vida del año en curso
    esta genuinamente disponible al predecir la del siguiente."""
    from src.domains.consulting_excel_dwh.features import FEATURE_COLUMNS, TARGET_COLUMN
    from src.domains.consulting_excel_dwh.model import chronological_split
    from src.toolkit.leakage import leakage_report

    features_df = pd.read_csv(PROCESSED_DIR / "consulting_features.csv")
    train_df, _val_df, test_df = chronological_split(features_df)

    reporte = leakage_report(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], X_test=test_df[FEATURE_COLUMNS],
    )

    assert reporte["veredicto"] == "revisar"
    assert "esperanza_vida" in reporte["features_sospechosas"]
    assert reporte["relaciones_exactas"] == []  # ninguna feature ES el target
    # El split cronologico no comparte ni una fila entre train y test.
    assert reporte["solapamiento_train_test"]["n_solapadas"] == 0
