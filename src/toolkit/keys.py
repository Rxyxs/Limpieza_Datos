"""Integridad de claves y diagnóstico de joins: qué identifica unívocamente una
fila, qué cardinalidad tiene realmente un `merge` antes de ejecutarlo, y qué
filas se van a perder o a multiplicar al hacerlo.

`pd.merge` es la operación que más silenciosamente corrompe un dataset. No
lanza ninguna excepción cuando la clave derecha está duplicada: multiplica las
filas de la izquierda y devuelve un DataFrame más grande que el original, con
cada fila repetida tantas veces como duplicados haya. Todo agregado posterior
--una suma de producción, un conteo de clientes-- queda inflado por un factor
que no aparece en ningún lado, y el resultado sigue siendo un DataFrame de
aspecto perfectamente normal.

Tampoco avisa de lo contrario: un `how="inner"` que descarta el 40% de las
filas porque las claves no matchean (un espacio duro de Excel, un código con
ceros a la izquierda perdidos al leer como número) devuelve un resultado más
chico, correcto en apariencia y sin una sola advertencia.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd


def duplicate_key_rows(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Devuelve las filas cuya combinación de `keys` aparece más de una vez, con
    la cuenta de repeticiones, ordenadas de la más repetida a la menos.

    Se devuelven las filas y no solo el conteo a propósito: para decidir qué
    hacer con un duplicado hay que ver si las dos filas son idénticas (sobra
    una) o si difieren en alguna columna (hay un conflicto real de datos, y
    quedarse con cualquiera de las dos es una decisión de negocio, no técnica).
    """
    marcadas = df[df.duplicated(subset=keys, keep=False)].copy()
    if marcadas.empty:
        return marcadas.assign(n_repeticiones=pd.Series(dtype=int))

    marcadas["n_repeticiones"] = marcadas.groupby(keys)[keys[0]].transform("size")
    return marcadas.sort_values("n_repeticiones", ascending=False)


def find_candidate_keys(
    df: pd.DataFrame, max_columns: int = 2, columns: list[str] | None = None,
) -> list[tuple[str, ...]]:
    """Busca combinaciones de hasta `max_columns` columnas que identifiquen cada
    fila de forma única, de la más chica a la más grande.

    Solo se reportan las combinaciones **mínimas**: si `id` ya es clave, no se
    reporta `(id, fecha)`, que también lo sería trivialmente y solo agregaría
    ruido a la salida.

    Una columna con nulos nunca se acepta como clave: en SQL dos `NULL` no son
    iguales entre sí, así que una "clave" con nulos no identifica nada y además
    se comporta distinto en pandas (que sí los agrupa) que en la base de datos
    a la que después se carga el dato.
    """
    candidatas = [
        c for c in (columns if columns is not None else df.columns) if df[c].notna().all()
    ]
    n = len(df)
    claves: list[tuple[str, ...]] = []

    for tamano in range(1, max_columns + 1):
        for combinacion in combinations(candidatas, tamano):
            if any(set(clave).issubset(combinacion) for clave in claves):
                continue  # no es mínima: ya contiene una clave conocida
            if len(df.drop_duplicates(subset=list(combinacion))) == n:
                claves.append(combinacion)
    return claves


def find_orphans(
    left: pd.DataFrame, right: pd.DataFrame, on: str | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filas de cada lado cuya clave no existe en el otro: `(huerfanas_izq, huerfanas_der)`.

    Son exactamente las filas que un `inner join` descarta sin decir nada. En un
    esquema estrella, huérfanas del lado del hecho significan hechos sin
    dimensión (un código de país que no está en la tabla de países); del lado de
    la dimensión, significan dimensiones sin uso, que es normal y no un error.
    Distinguir los dos casos es la razón de devolver los dos lados por separado.
    """
    keys = [on] if isinstance(on, str) else list(on)
    izq = left.merge(right[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    der = right.merge(left[keys].drop_duplicates(), on=keys, how="left", indicator=True)

    return (
        left.loc[(izq["_merge"] == "left_only").to_numpy()],
        right.loc[(der["_merge"] == "left_only").to_numpy()],
    )


def describe_join(
    left: pd.DataFrame, right: pd.DataFrame, on: str | list[str],
) -> dict:
    """Diagnostica un join **antes** de ejecutarlo: cardinalidad real, huérfanas de
    cada lado, y cuántas filas devolvería un `inner` y un `left`.

    `factor_multiplicacion` es el número que hay que mirar: cuántas filas
    devuelve el `left join` por cada fila de la izquierda. Cualquier valor
    mayor a 1.0 significa que la clave derecha está duplicada y que el join
    está multiplicando filas -- el modo de falla que no lanza ninguna
    excepción y que infla todo agregado calculado después.

    La cardinalidad se reporta como se observa en LOS DATOS (`1:1`, `1:N`,
    `N:1`, `M:N`), no como dice el esquema. Una tabla declarada 1:1 que hoy
    tiene un duplicado se comporta como 1:N, y es el comportamiento el que
    rompe el pipeline.
    """
    keys = [on] if isinstance(on, str) else list(on)

    izq_unica = not left.duplicated(subset=keys).any()
    der_unica = not right.duplicated(subset=keys).any()
    cardinalidad = {(True, True): "1:1", (True, False): "1:N", (False, True): "N:1"}.get(
        (izq_unica, der_unica), "M:N"
    )

    huerfanas_izq, huerfanas_der = find_orphans(left, right, keys)
    claves_der = right.groupby(keys, dropna=False).size()
    filas_left_join = int(
        left.merge(claves_der.rename("n").reset_index(), on=keys, how="left")["n"].fillna(1).sum()
    )

    return {
        "cardinalidad": cardinalidad,
        "clave_izquierda_unica": izq_unica,
        "clave_derecha_unica": der_unica,
        "filas_izquierda": len(left),
        "filas_derecha": len(right),
        "huerfanas_izquierda": len(huerfanas_izq),
        "huerfanas_derecha": len(huerfanas_der),
        "filas_tras_left_join": filas_left_join,
        "filas_tras_inner_join": filas_left_join - len(huerfanas_izq),
        "factor_multiplicacion": round(filas_left_join / len(left), 4) if len(left) else float("nan"),
    }


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "left",
    allow_multiplication: bool = False,
    max_orphan_rate: float | None = None,
) -> pd.DataFrame:
    """`pd.merge` que falla en vez de corromper: aborta si el join multiplicaría
    filas, o si descartaría más filas de las toleradas.

    Es `pd.merge(..., validate=...)` llevado al caso real: `validate` solo
    chequea cardinalidad, mientras que acá el segundo modo de falla -- perder
    filas silenciosamente porque las claves no matchean -- también corta, con
    `max_orphan_rate` (ej. `0.05` = "no acepto perder más del 5%"). Los dos
    chequeos son opt-out explícito, así que multiplicar filas queda como una
    decisión escrita en el código y no como algo que pasó sin que nadie lo viera.
    """
    diagnostico = describe_join(left, right, on)

    if not allow_multiplication and diagnostico["factor_multiplicacion"] > 1:
        raise ValueError(
            f"el join multiplicaria filas x{diagnostico['factor_multiplicacion']} "
            f"({len(left)} -> {diagnostico['filas_tras_left_join']}): la clave derecha "
            f"esta duplicada (cardinalidad {diagnostico['cardinalidad']}). "
            "Deduplicar la derecha, o pasar allow_multiplication=True si la multiplicacion es intencional."
        )

    if max_orphan_rate is not None and len(left):
        tasa = diagnostico["huerfanas_izquierda"] / len(left)
        if tasa > max_orphan_rate:
            raise ValueError(
                f"{diagnostico['huerfanas_izquierda']} de {len(left)} filas ({tasa:.1%}) no "
                f"encuentran clave en la derecha, sobre el maximo tolerado de {max_orphan_rate:.1%}. "
                "Suele ser un problema de formato de la clave (espacios, ceros a la izquierda, "
                "mayusculas), no de datos genuinamente ausentes."
            )

    return left.merge(right, on=on, how=how)
