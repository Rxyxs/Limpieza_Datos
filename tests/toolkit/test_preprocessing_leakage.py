"""Día 22 -- auditoría de fuga estadística (distinta de la fuga de target que
ya cubre `test_leakage.py`): ¿los parámetros con los que se limpia/imputa/
escala un dato dependen de información que, en un split cronológico real, no
estaría disponible todavía?

Hallazgo del Día 22, probado acá con números, no solo leído en el código:
`impute_numeric_by_category`, `interpolate_within_group` y `winsorize_column*`
son funciones puras y sin estado -- recalculan sus estadísticos (media,
mediana condicional, cuantiles IQR) directamente sobre el DataFrame que se les
pasa, cada vez. Eso las vuelve perfectamente reusables en streaming, pero no
tienen ningún mecanismo de "ajustar sobre train, aplicar sobre test" -- a
diferencia de `zscore_scale`, que sí devuelve sus estadísticos aprendidos para
poder reutilizarlos.

En este repo, `agriculture_worldbank/clean.py` y `consulting_excel_dwh/
clean.py` llaman `interpolate_within_group`/`impute_numeric_by_category`
sobre el panel completo, ANTES del split cronológico que corre después en
`model.py`. El resultado: un hueco en el período de entrenamiento puede
quedar interpolado usando valores del período de test. Es una fuga real,
aunque mucho más chica que una fuga de target -- no se está filtrando la
respuesta, se está filtrando "cuánto va a valer esta columna más adelante"
hacia una fila de entrenamiento que, en producción real, nunca lo habría
sabido.

No se toca el pipeline hoy (restricción del Día 22: sin reescrituras
masivas). Estos tests documentan el comportamiento actual -- tanto el que
filtra como el que no -- para que un cambio futuro se note como un test que
cambia, no como una sorpresa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.toolkit.encoding import zscore_scale
from src.toolkit.missing_data import impute_numeric_by_category, interpolate_within_group
from src.toolkit.outliers import winsorize_column


# --------------------------------------------------- interpolate_within_group

def test_interpolate_within_group_filtra_informacion_del_periodo_de_test():
    """Un hueco en el período de train se interpola distinto según si el
    período de test (futuro) está o no presente en el DataFrame -- la
    definición operativa de fuga temporal."""
    panel = pd.DataFrame({
        "pais": ["CL"] * 10,
        "anio": range(2010, 2020),
        # Hueco en 2013 (índice 3, período train). Sin el futuro: solo hay
        # vecinos train interpolables. Con el futuro (2018=90 en vez de
        # continuar la tendencia lineal), el vecino "de la derecha" cambia.
        "valor": [10, 20, 30, np.nan, 50, 60, 70, 80, 90, 999],
    })
    train_periodo = panel[panel["anio"] < 2015].reset_index(drop=True)

    solo_train = interpolate_within_group(train_periodo, column="valor", group_column="pais", sort_by="anio")
    con_test_futuro = interpolate_within_group(panel, column="valor", group_column="pais", sort_by="anio")

    valor_solo_train = solo_train.loc[solo_train["anio"] == 2013, "valor"].iloc[0]
    valor_con_futuro = con_test_futuro.loc[con_test_futuro["anio"] == 2013, "valor"].iloc[0]

    # En este caso concreto la interpolacion lineal 2010-2020 da el mismo
    # resultado en 2013 (la serie es lineal salvo el ultimo punto, que no
    # participa en la interpolacion de 2013 -- son vecinos INMEDIATOS los que
    # importan). Confirma la propiedad util: la fuga existe cuando el vecino
    # inmediato del hueco cae en el futuro, no en cualquier fila futura.
    hueco_pegado_al_futuro = pd.DataFrame({
        "pais": ["CL"] * 6,
        "anio": range(2010, 2016),
        "valor": [10, 20, np.nan, np.nan, np.nan, 999],  # los 3 huecos tocan el borde train/test
    })
    train_periodo_2 = hueco_pegado_al_futuro[hueco_pegado_al_futuro["anio"] < 2014].reset_index(drop=True)

    solo_train_2 = interpolate_within_group(train_periodo_2, column="valor", group_column="pais", sort_by="anio")
    con_futuro_2 = interpolate_within_group(hueco_pegado_al_futuro, column="valor", group_column="pais", sort_by="anio")

    # Sin el futuro: no hay vecino a la derecha, `limit_direction="both"`
    # arrastra el ultimo valor conocido (20) hacia adelante.
    assert solo_train_2.loc[solo_train_2["anio"] == 2013, "valor"].iloc[0] == pytest.approx(20.0)
    # Con el futuro: el 999 de 2015 SI participa como vecino derecho y arrastra
    # el valor interpolado de 2013 hacia arriba -- una fila de 2013 (train)
    # "sabe" que en 2015 (test) el valor salta a 999.
    valor_2013_con_fuga = con_futuro_2.loc[con_futuro_2["anio"] == 2013, "valor"].iloc[0]
    assert valor_2013_con_fuga != pytest.approx(20.0)
    assert valor_2013_con_fuga > 20.0


# ------------------------------------------------ impute_numeric_by_category

def test_impute_numeric_by_category_filtra_la_media_de_test_hacia_train():
    """La media condicional por categoria se calcula sobre TODO el df que se
    le pase -- si ese df mezcla train y test, la media usada para imputar una
    fila de train incluye observaciones de test."""
    train = pd.DataFrame({
        "pais": ["CL", "CL", "CL"],
        "valor": [10.0, np.nan, 30.0],
    })
    completo = pd.concat([
        train,
        pd.DataFrame({"pais": ["CL"], "valor": [1000.0]}),  # una sola fila de "test", muy alejada
    ], ignore_index=True)

    imputado_solo_train = impute_numeric_by_category(train, value_column="valor", category_column="pais")
    imputado_con_test = impute_numeric_by_category(completo, value_column="valor", category_column="pais")

    media_solo_train = imputado_solo_train.loc[1, "valor"]
    media_con_test = imputado_con_test.loc[1, "valor"]

    assert media_solo_train == pytest.approx(20.0)  # media de [10, 30]
    assert media_con_test == pytest.approx((10 + 30 + 1000) / 3)  # la fila de test arrastra la media
    assert media_con_test != pytest.approx(media_solo_train)


# --------------------------------------------------------- winsorize_column

def test_winsorize_column_los_limites_iqr_dependen_de_si_el_futuro_esta_incluido():
    """Los cuantiles Q1/Q3 se calculan sobre el df completo que se pase --
    agregar filas de test antes de winsorizar cambia los limites que se
    aplican sobre las filas de train."""
    train = pd.Series(list(range(1, 21)), dtype=float)  # 1..20, sin outliers
    con_test_extremo = pd.concat([train, pd.Series([500.0, 600.0])], ignore_index=True)

    _, n_recortados_solo_train = winsorize_column(pd.DataFrame({"x": train}), "x")
    _, n_recortados_con_test = winsorize_column(pd.DataFrame({"x": con_test_extremo}), "x")

    # Sin las filas de test, 1..20 no tiene outliers bajo IQR de Tukey.
    assert n_recortados_solo_train == 0
    # Con las dos filas de test extremas agregadas ANTES de winsorizar, el
    # IQR se ensancha y ademas esas dos filas se cuentan como outliers --
    # el limite superior que ve la fila 20 de train ya no es el mismo.
    assert n_recortados_con_test == 2


# ----------------------------------------------- zscore_scale: el patron correcto

def test_zscore_scale_permite_ajustar_en_train_y_aplicar_los_mismos_stats_en_test():
    """A diferencia de las funciones de arriba, zscore_scale devuelve sus
    estadisticos aprendidos -- el contrato que permite usarla sin fuga: se
    ajusta una vez sobre train, y esos mismos (media, std) se aplican a mano
    sobre test, sin volver a llamar a la funcion sobre datos que incluyan test.
    Este es el patron que los 4 dominios de este repo efectivamente usan en
    su `model.py`."""
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"x": [100.0, 200.0]})  # deliberadamente fuera de la escala de train

    train_scaled, stats = zscore_scale(train, ["x"])
    mean, std = stats["x"]

    # El patron correcto: aplicar los stats de train a mano sobre test.
    test_scaled_correcto = test.copy()
    test_scaled_correcto["x"] = (test_scaled_correcto["x"] - mean) / std

    # El patron incorrecto (el que NO deberia usarse): volver a llamar
    # zscore_scale sobre test, que recalcula media/std solo con esos 2 valores.
    test_scaled_incorrecto, stats_incorrectos = zscore_scale(test, ["x"])

    assert stats_incorrectos["x"] != pytest.approx(stats["x"])
    assert not np.allclose(test_scaled_correcto["x"].to_numpy(), test_scaled_incorrecto["x"].to_numpy())

    # Los stats de train no se mutan por aplicarlos afuera ni por la llamada
    # incorrecta sobre test -- son un dict de floats, inmutable de hecho.
    assert stats["x"] == pytest.approx((mean, std))


def test_zscore_scale_stats_no_cambian_al_reaplicarse_sobre_otro_df():
    train = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
    _, stats_1 = zscore_scale(train, ["x"])

    otro_df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
    mean, std = stats_1["x"]
    otro_df["x"] = (otro_df["x"] - mean) / std

    assert stats_1["x"] == pytest.approx((mean, std))  # el dict original no se toco
