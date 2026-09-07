"""Pruebas unitarias para src/toolkit/keys.py."""
import numpy as np
import pandas as pd
import pytest

from src.toolkit.keys import (
    describe_join,
    duplicate_key_rows,
    find_candidate_keys,
    find_orphans,
    safe_merge,
)


@pytest.fixture
def hechos():
    return pd.DataFrame({
        "pais": ["CHL", "CHL", "ARG", "PER"],
        "anio": [2023, 2024, 2023, 2023],
        "produccion": [100.0, 110.0, 20.0, 30.0],
    })


@pytest.fixture
def dimension():
    return pd.DataFrame({
        "pais": ["CHL", "ARG", "BRA"],
        "region": ["Sudamerica", "Sudamerica", "Sudamerica"],
    })


def test_find_candidate_keys_returns_only_minimal_combinations(hechos):
    claves = find_candidate_keys(hechos, max_columns=2)

    # (pais, anio) identifica cada fila; "produccion" sola tambien, porque no
    # tiene repetidos en este ejemplo. Ninguna columna sola de las otras dos.
    assert ("pais", "anio") in claves
    assert ("produccion",) in claves
    # Minimalidad: no se reporta un superconjunto de una clave ya encontrada.
    assert not any(set(("produccion",)).issubset(k) and len(k) > 1 for k in claves)


def test_find_candidate_keys_rejects_columns_with_nulls():
    df = pd.DataFrame({"id": [1, 2, np.nan], "otro": ["a", "b", "c"]})
    claves = find_candidate_keys(df, max_columns=1)

    # `id` seria unica ignorando el nulo, pero dos NULL no son iguales en SQL,
    # asi que no sirve como clave.
    assert ("id",) not in claves
    assert ("otro",) in claves


def test_duplicate_key_rows_returns_the_rows_not_just_the_count():
    df = pd.DataFrame({
        "pais": ["CHL", "CHL", "ARG"],
        "anio": [2023, 2023, 2023],
        "valor": [100.0, 999.0, 20.0],  # las dos filas de CHL NO son identicas
    })
    duplicadas = duplicate_key_rows(df, ["pais", "anio"])

    assert len(duplicadas) == 2
    assert set(duplicadas["valor"]) == {100.0, 999.0}
    assert duplicadas["n_repeticiones"].max() == 2


def test_duplicate_key_rows_is_empty_when_the_key_is_unique(hechos):
    assert duplicate_key_rows(hechos, ["pais", "anio"]).empty


def test_find_orphans_separates_each_side(hechos, dimension):
    huerfanas_izq, huerfanas_der = find_orphans(hechos, dimension, on="pais")

    assert list(huerfanas_izq["pais"]) == ["PER"]  # hecho sin dimension
    assert list(huerfanas_der["pais"]) == ["BRA"]  # dimension sin uso


def test_describe_join_reports_observed_cardinality(hechos, dimension):
    diagnostico = describe_join(hechos, dimension, on="pais")

    assert diagnostico["cardinalidad"] == "N:1"
    assert diagnostico["huerfanas_izquierda"] == 1
    assert diagnostico["huerfanas_derecha"] == 1
    assert diagnostico["factor_multiplicacion"] == 1.0
    assert diagnostico["filas_tras_left_join"] == 4
    assert diagnostico["filas_tras_inner_join"] == 3


def test_describe_join_detects_the_silent_row_multiplication(hechos):
    # La dimension trae CHL dos veces: pandas multiplica sin decir nada.
    dimension_sucia = pd.DataFrame({
        "pais": ["CHL", "CHL", "ARG"],
        "region": ["Sudamerica", "Cono Sur", "Sudamerica"],
    })
    diagnostico = describe_join(hechos, dimension_sucia, on="pais")

    assert diagnostico["cardinalidad"] == "M:N"
    assert diagnostico["factor_multiplicacion"] > 1
    assert diagnostico["filas_tras_left_join"] == 6  # 4 filas -> 6
    assert len(hechos.merge(dimension_sucia, on="pais", how="left")) == 6  # pandas no avisa


def test_safe_merge_refuses_to_multiply_rows(hechos):
    dimension_sucia = pd.DataFrame({"pais": ["CHL", "CHL"], "region": ["A", "B"]})

    with pytest.raises(ValueError, match="multiplicaria filas"):
        safe_merge(hechos, dimension_sucia, on="pais")


def test_safe_merge_allows_multiplication_when_it_is_explicit(hechos):
    dimension_sucia = pd.DataFrame({"pais": ["CHL", "CHL"], "region": ["A", "B"]})
    resultado = safe_merge(hechos, dimension_sucia, on="pais", allow_multiplication=True)

    assert len(resultado) == 6


def test_safe_merge_refuses_to_silently_drop_too_many_rows(hechos, dimension):
    # 1 de 4 filas (25%) queda huerfana, sobre un maximo tolerado de 10%.
    with pytest.raises(ValueError, match="no\nencuentran|encuentran clave"):
        safe_merge(hechos, dimension, on="pais", max_orphan_rate=0.10)


def test_safe_merge_passes_through_when_everything_is_clean(hechos, dimension):
    resultado = safe_merge(hechos, dimension, on="pais", max_orphan_rate=0.30)

    assert len(resultado) == len(hechos)
    assert "region" in resultado.columns
