"""Lectura y escritura de *dumps* SQL (`.sql`): el formato en que un cliente
entrega una base de datos completa cuando no da acceso al motor.

Un dump no es un CSV con otra extensión: es un script -- `CREATE TABLE`,
miles de `INSERT INTO ... VALUES (...), (...)`, o bloques `COPY ... FROM stdin`
de `pg_dump` -- con comentarios, comillas escapadas de tres formas distintas
según el motor, `NULL` como palabra reservada y no como celda vacía, y varias
tablas dentro del mismo archivo. Restaurarlo requiere levantar el motor
correspondiente (MySQL para un dump de MySQL, Postgres para uno de Postgres),
que es exactamente lo que no hay a mano cuando el archivo llega por correo.

Este módulo lee el dump directamente a DataFrames -- en streaming, línea por
línea, porque un dump real pesa más que la RAM disponible -- y escribe el
resultado limpio de vuelta como dump portable, que es el formato en que un
equipo de ingeniería lo quiere recibir para cargarlo a su propia base.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# Palabras con que arranca una definición de restricción dentro de un
# `CREATE TABLE`: no son columnas y hay que saltarlas al leer el esquema.
CONSTRAINT_KEYWORDS = (
    "PRIMARY", "UNIQUE", "KEY", "CONSTRAINT", "FOREIGN", "CHECK",
    "INDEX", "FULLTEXT", "SPATIAL", "EXCLUDE", "PERIOD",
)

# Secuencias de escape con backslash que usa MySQL dentro de literales de texto
# (Postgres, con `standard_conforming_strings` activo, no las interpreta).
BACKSLASH_ESCAPES = {
    "0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t",
    "Z": "\x1a", "\\": "\\", "'": "'", '"': '"', "%": "\\%", "_": "\\_",
}

CREATE_TABLE_RE = re.compile(
    r"^\s*CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?", re.IGNORECASE
)
INSERT_RE = re.compile(
    r"^\s*INSERT\s+(?:LOW_PRIORITY\s+|DELAYED\s+|HIGH_PRIORITY\s+|IGNORE\s+)*INTO\s+", re.IGNORECASE
)
COPY_RE = re.compile(r"^\s*COPY\s+", re.IGNORECASE)
COPY_FROM_STDIN_RE = re.compile(r"^\s*COPY\s+.*\bFROM\s+stdin", re.IGNORECASE | re.DOTALL)
VALUES_RE = re.compile(r"\s*VALUES\s*", re.IGNORECASE)

# Comillas con que los motores delimitan un identificador, y su cierre.
IDENTIFIER_DELIMITERS = {'"': '"', "`": "`", "[": "]"}

SQL_TYPES = {
    "postgres": {"int": "BIGINT", "float": "DOUBLE PRECISION", "bool": "BOOLEAN",
                 "datetime": "TIMESTAMP", "text": "TEXT"},
    "mysql": {"int": "BIGINT", "float": "DOUBLE", "bool": "TINYINT(1)",
              "datetime": "DATETIME", "text": "TEXT"},
    "sqlite": {"int": "INTEGER", "float": "REAL", "bool": "INTEGER",
               "datetime": "TEXT", "text": "TEXT"},
}
IDENTIFIER_QUOTES = {"postgres": '"', "mysql": "`", "sqlite": '"'}


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

@contextmanager
def _open_lines(source: str | Path | Iterable[str]):
    """Acepta una ruta, el texto SQL completo, o cualquier iterable de líneas.

    Un `str` se interpreta como texto SQL si contiene un salto de línea, y como
    ruta en caso contrario -- la ambigüedad se resuelve así y no por
    `os.path.exists` para que el comportamiento no dependa del disco. Solo se
    cierra el archivo si lo abrió esta función; un handle recibido de afuera se
    devuelve intacto a quien lo abrió.
    """
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source):
        handle = Path(source).open(encoding="utf-8", errors="replace")
        try:
            yield handle
        finally:
            handle.close()
    elif isinstance(source, str):
        yield source.splitlines(keepends=True)
    else:
        yield source


def iter_sql_statements(source: str | Path | Iterable[str]) -> Iterator[str]:
    """Itera las sentencias de un dump, una a una, sin cargar el archivo en memoria.

    Partir por `;` con `split` es la solución obvia y está mal: un `;` dentro de
    un literal de texto (`'Codelco; División Norte'`) parte la sentencia por la
    mitad y todo lo que sigue queda corrido. Acá el corte se hace con un
    autómata que conoce los tres tipos de comilla que usan los motores
    (`'texto'`, `"identificador"`, `` `identificador` ``), los escapes con
    backslash de MySQL, y los comentarios `--`, `#` y `/* */`.

    Los bloques `COPY ... FROM stdin` de `pg_dump` se emiten completos (la
    sentencia más sus líneas de datos hasta el `\\.` final), porque sus datos no
    son SQL y no terminan en `;`.
    """
    with _open_lines(source) as lines:
        yield from _parse_statements(lines)


def _parse_statements(lines: Iterable[str]) -> Iterator[str]:
    """Autómata que corta las sentencias de un iterable de líneas ya abierto."""
    current: list[str] = []
    in_single = in_double = in_backtick = in_block_comment = False
    escaped = False
    copy_mode = False

    for raw_line in lines:
        if copy_mode:
            current.append(raw_line)
            if raw_line.strip() == "\\.":
                copy_mode = False
                yield "".join(current)
                current = []
            continue

        i, length = 0, len(raw_line)
        while i < length:
            char = raw_line[i]

            if in_block_comment:
                if char == "*" and raw_line[i + 1 : i + 2] == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if in_single:
                current.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "'":
                    if raw_line[i + 1 : i + 2] == "'":  # comilla escapada al estilo SQL estándar
                        current.append("'")
                        i += 2
                        continue
                    in_single = False
                i += 1
                continue

            if in_double or in_backtick:
                current.append(char)
                if char == '"' and in_double:
                    in_double = False
                elif char == "`" and in_backtick:
                    in_backtick = False
                i += 1
                continue

            two = raw_line[i : i + 2]
            if two == "--" or char == "#":
                break  # comentario de línea: se descarta el resto de la línea
            if two == "/*":
                in_block_comment = True
                i += 2
                continue

            if char == "'":
                in_single = True
            elif char == '"':
                in_double = True
            elif char == "`":
                in_backtick = True
            elif char == ";":
                statement = "".join(current).strip()
                current = []
                if statement:
                    if COPY_FROM_STDIN_RE.match(statement):
                        current = [statement + ";\n"]
                        copy_mode = True
                        break
                    yield statement
                i += 1
                continue

            current.append(char)
            i += 1

        if not (in_single or in_double or in_backtick or copy_mode):
            current.append("\n")

    tail = "".join(current).strip()
    if tail:
        yield tail


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    """Parte `text` por `separator` ignorando los que caen dentro de comillas o
    de un paréntesis anidado -- el separador de una lista de valores SQL."""
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    in_single = in_double = in_backtick = False
    escaped = False

    for char in text:
        if in_single:
            buffer.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                in_single = False
            continue
        if in_double or in_backtick:
            buffer.append(char)
            if char == '"' and in_double:
                in_double = False
            elif char == "`" and in_backtick:
                in_backtick = False
            continue

        if char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "`":
            in_backtick = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append("".join(buffer))
            buffer = []
            continue
        buffer.append(char)

    if "".join(buffer).strip():
        parts.append("".join(buffer))
    return parts


def _unquote(token: str) -> str:
    """Quita comillas de identificador (`` ` ``, `"`, `[]`) y el prefijo de esquema.

    El punto se trata como separador de esquema solo fuera de comillas: hay
    nombres reales de columna que lo contienen (`"Chuqui y R.Tomic"`, en el
    archivo de COCHILCO de este repo), y cortarlos por el punto los dejaría
    como `Tomic`, es decir, con un nombre que no existe en ninguna otra parte
    del pipeline.
    """
    token = token.strip().strip(",;").strip()
    segments = _split_top_level(token, ".")
    token = segments[-1].strip() if segments else token

    if len(token) >= 2 and token[0] in IDENTIFIER_DELIMITERS and token[-1] == IDENTIFIER_DELIMITERS[token[0]]:
        token = token[1:-1].replace('""', '"').replace("``", "`")
    return token.strip()


def _read_identifier(text: str, start: int = 0) -> tuple[str, int]:
    """Lee el identificador que empieza en `start` y devuelve `(identificador, fin)`.

    Un identificador entre comillas puede contener espacios y paréntesis (el
    archivo real de COCHILCO trae columnas como `Centinela (sulfuros)`), así que
    cortar por el primer espacio o el primer paréntesis -- lo que haría una
    expresión regular simple -- parte el nombre en dos.
    """
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return "", i

    if text[i] in IDENTIFIER_DELIMITERS:
        closing = IDENTIFIER_DELIMITERS[text[i]]
        j = i + 1
        while j < len(text):
            if text[j] == closing:
                if text[j + 1 : j + 2] == closing:  # comilla escapada duplicándola
                    j += 2
                    continue
                j += 1
                break
            j += 1
        # Un nombre calificado (`"esquema"."tabla"`) sigue con un punto: vale el
        # último segmento, que es la tabla.
        if text[j : j + 1] == ".":
            qualified, end = _read_identifier(text, j + 1)
            return (qualified or _unquote(text[i:j])), end
        return _unquote(text[i:j]), j

    j = i
    while j < len(text) and not text[j].isspace() and text[j] not in "(,;":
        j += 1
    return _unquote(text[i:j]), j


def _matching_paren(text: str, open_index: int) -> int:
    """Índice del `)` que cierra el `(` de `open_index`, ignorando paréntesis que
    estén dentro de un literal o de un identificador entre comillas."""
    depth = 0
    in_single = in_double = in_backtick = False
    escaped = False

    for i in range(open_index, len(text)):
        char = text[i]
        if in_single:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                in_single = False
            continue
        if in_double or in_backtick:
            if char == '"' and in_double:
                in_double = False
            elif char == "`" and in_backtick:
                in_backtick = False
            continue

        if char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "`":
            in_backtick = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _unescape_string(literal: str) -> str:
    """Convierte un literal `'...'` de SQL al texto que representa."""
    body = literal[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            out.append(BACKSLASH_ESCAPES.get(body[i + 1], body[i + 1]))
            i += 2
        elif char == "'" and body[i + 1 : i + 2] == "'":
            out.append("'")
            i += 2
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _read_optional_column_list(text: str, cursor: int) -> tuple[list[str] | None, int]:
    """Lee la lista `(col1, col2, ...)` que puede seguir al nombre de la tabla.

    Devuelve `(None, cursor)` si no hay lista: en un `INSERT` de `mysqldump` las
    columnas no se declaran y hay que tomarlas del `CREATE TABLE`.
    """
    i = cursor
    while i < len(text) and text[i].isspace():
        i += 1
    if text[i : i + 1] != "(":
        return None, cursor

    close = _matching_paren(text, i)
    if close == -1:
        return None, cursor
    return [_unquote(c) for c in _split_top_level(text[i + 1 : close])], close + 1


def parse_sql_literal(token: str):
    """Convierte un literal SQL al valor Python que corresponde.

    `NULL` es la distinción que importa: en un dump es una palabra reservada sin
    comillas, mientras que `'NULL'` con comillas es el texto "NULL". Un split
    ingenuo por comas colapsa ambos al mismo string y convierte un dato faltante
    real en un valor presente (o al revés).
    """
    token = token.strip()
    if not token:
        return None
    upper = token.upper()
    if upper in {"NULL", "\\N"}:
        return None
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False

    if token.startswith("_binary "):
        token = token[len("_binary ") :].strip()
    if len(token) >= 2 and token[0] == "'" and token[-1] == "'":
        return _unescape_string(token)

    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_create_table(statement: str) -> tuple[str, list[str]] | None:
    """Extrae `(tabla, columnas)` de un `CREATE TABLE`, o `None` si la sentencia
    no lo es. Ignora las definiciones de índice/clave, que comparten sintaxis de
    lista con las columnas pero no son columnas.
    """
    match = CREATE_TABLE_RE.match(statement)
    if not match:
        return None

    table, cursor = _read_identifier(statement, match.end())
    open_paren = statement.find("(", cursor)
    close_paren = _matching_paren(statement, open_paren) if open_paren != -1 else -1
    if open_paren == -1 or close_paren == -1:
        return table, []

    columns = []
    for definition in _split_top_level(statement[open_paren + 1 : close_paren]):
        definition = definition.strip()
        if not definition:
            continue
        # Una restricción (`PRIMARY KEY (...)`, `KEY idx (...)`) comparte la
        # sintaxis de lista con las columnas pero no define ninguna. Solo se
        # descarta si la palabra viene SIN comillas: `"key"` entre comillas es
        # un nombre de columna legítimo.
        if definition[0] not in IDENTIFIER_DELIMITERS and definition.split()[0].upper() in CONSTRAINT_KEYWORDS:
            continue
        name, _end = _read_identifier(definition)
        if name:
            columns.append(name)
    return table, columns


def parse_insert(statement: str) -> tuple[str, list[str] | None, list[list]] | None:
    """Extrae `(tabla, columnas_o_None, filas)` de un `INSERT`, o `None`.

    Soporta el `INSERT` multi-fila que genera `mysqldump` (una sola sentencia
    con miles de tuplas `(...),(...)`), que es la razón por la que un dump de
    MySQL puede tener millones de filas en unas pocas líneas de texto.
    """
    match = INSERT_RE.match(statement)
    if not match:
        return None

    table, cursor = _read_identifier(statement, match.end())
    columns, cursor = _read_optional_column_list(statement, cursor)

    values_match = VALUES_RE.match(statement, cursor)
    if not values_match:
        return None

    rows = []
    for chunk in _split_top_level(statement[values_match.end() :].strip().rstrip(";")):
        chunk = chunk.strip()
        if not (chunk.startswith("(") and chunk.endswith(")")):
            continue
        rows.append([parse_sql_literal(v) for v in _split_top_level(chunk[1:-1])])
    return table, columns, rows


def parse_copy_block(chunk: str) -> tuple[str, list[str] | None, list[list]] | None:
    """Extrae `(tabla, columnas, filas)` de un bloque `COPY ... FROM stdin` de
    `pg_dump`, o `None`.

    El cuerpo del bloque es TSV, no SQL: `\\N` marca nulo (distinto del texto
    vacío, que es un campo de largo cero) y el tabulador, el salto de línea y el
    backslash van escapados. Es el formato por defecto de `pg_dump` sin
    `--inserts`, así que aparece en la mayoría de los dumps de Postgres reales.
    """
    lines = chunk.splitlines()
    if not lines or not COPY_FROM_STDIN_RE.match(lines[0]):
        return None

    match = COPY_RE.match(lines[0])
    table, cursor = _read_identifier(lines[0], match.end())
    columns, _cursor = _read_optional_column_list(lines[0], cursor)

    rows = []
    for line in lines[1:]:
        if line.strip() == "\\.":
            break
        if line == "":
            continue
        rows.append([
            None if field == "\\N" else field.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\")
            for field in line.rstrip("\n").split("\t")
        ])
    return table, columns, rows


def _record(columns: list[str] | None, row: list) -> dict:
    names = columns if columns else [f"col_{i}" for i in range(len(row))]
    return dict(zip(names, row))


def read_sql_dump(
    source: str | Path | Iterable[str], tables: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Lee un dump completo y devuelve `{nombre_tabla: DataFrame}`.

    Los nombres de columna salen del `CREATE TABLE` cuando existe (un dump de
    `pg_dump` con formato `COPY` no repite los nombres en cada fila), del propio
    `INSERT` si trae lista explícita, y en última instancia son posicionales
    (`col_0`, `col_1`...). `tables` filtra qué tablas materializar, que sobre un
    dump grande es la diferencia entre cargar una tabla y cargar la base entera.
    """
    wanted = {t.lower() for t in tables} if tables else None
    schemas: dict[str, list[str]] = {}
    records: dict[str, list[dict]] = {}

    for statement in iter_sql_statements(source):
        created = parse_create_table(statement)
        if created:
            table, columns = created
            if columns:
                schemas[table] = columns
            continue

        parsed = parse_insert(statement) or parse_copy_block(statement)
        if not parsed:
            continue

        table, columns, rows = parsed
        if wanted is not None and table.lower() not in wanted:
            continue
        columns = columns or schemas.get(table)
        records.setdefault(table, []).extend(_record(columns, row) for row in rows)

    frames = {}
    for table, rows in records.items():
        df = pd.DataFrame(rows)
        ordered = [c for c in schemas.get(table, []) if c in df.columns]
        frames[table] = df[ordered + [c for c in df.columns if c not in ordered]]
    return frames


def inventory_sql_dump(source: str | Path | Iterable[str]) -> pd.DataFrame:
    """Inventario de un dump sin materializarlo: tabla, filas, columnas y de dónde
    salieron los datos (`INSERT` o `COPY`).

    Sirve para decidir qué leer antes de leerlo. Un dump de producción trae
    decenas de tablas de las que interesan dos, y el inventario cuesta una
    pasada de texto en vez de la memoria de la base completa.
    """
    schemas: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    origins: dict[str, str] = {}

    for statement in iter_sql_statements(source):
        created = parse_create_table(statement)
        if created:
            table, columns = created
            schemas[table] = columns
            counts.setdefault(table, 0)
            continue

        insert = parse_insert(statement)
        parsed, origin = (insert, "INSERT") if insert else (parse_copy_block(statement), "COPY")
        if not parsed:
            continue

        table, columns, rows = parsed
        counts[table] = counts.get(table, 0) + len(rows)
        origins[table] = origin
        if columns and table not in schemas:
            schemas[table] = columns

    return pd.DataFrame([
        {
            "tabla": table,
            "n_filas": n_rows,
            "n_columnas": len(schemas.get(table, [])),
            "origen": origins.get(table, "solo DDL"),
        }
        for table, n_rows in counts.items()
    ])


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def _sql_type(series: pd.Series, dialect: str) -> str:
    types = SQL_TYPES[dialect]
    if pd.api.types.is_bool_dtype(series):
        return types["bool"]
    if pd.api.types.is_integer_dtype(series):
        return types["int"]
    if pd.api.types.is_float_dtype(series):
        return types["float"]
    if pd.api.types.is_datetime64_any_dtype(series):
        return types["datetime"]
    return types["text"]


def _quote_identifier(name: str, dialect: str) -> str:
    quote = IDENTIFIER_QUOTES[dialect]
    return f"{quote}{str(name).replace(quote, quote * 2)}{quote}"


def format_sql_value(value, dialect: str = "postgres") -> str:
    """Formatea un valor Python como literal SQL del dialecto pedido.

    El escape de comillas cambia según el motor: el `''` del estándar SQL lo
    entienden los tres, pero MySQL además interpreta el backslash como escape,
    así que un texto que contenga `\\` hay que duplicarlo para MySQL y dejarlo
    intacto para Postgres. Escribir el dump con el escape del motor equivocado
    produce un archivo que carga sin error y con el texto corrompido.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        return "NULL"
    if isinstance(value, bool):
        return ("1" if value else "0") if dialect == "sqlite" else ("TRUE" if value else "FALSE")
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, pd.Timestamp):
        return "'" + value.strftime("%Y-%m-%d %H:%M:%S") + "'"

    text = str(value).replace("'", "''")
    if dialect == "mysql":
        text = text.replace("\\", "\\\\")
    return f"'{text}'"


def dataframe_to_sql_dump(
    df: pd.DataFrame,
    table: str,
    path: str | Path,
    dialect: str = "postgres",
    batch_size: int = 500,
    include_create: bool = True,
    drop_if_exists: bool = True,
) -> Path:
    """Escribe `df` como dump SQL cargable (`CREATE TABLE` + `INSERT` por lotes) y
    devuelve la ruta.

    Los `INSERT` se agrupan de a `batch_size` filas en una sola sentencia
    multi-fila en vez de una sentencia por fila: es la misma decisión que toma
    `mysqldump` y la diferencia es de órdenes de magnitud al cargar, porque cada
    sentencia individual paga su propio round-trip y su propia transacción.

    El archivo se escribe en streaming, un lote a la vez, así que exportar un
    DataFrame grande no construye antes el dump completo en memoria.
    """
    if dialect not in SQL_TYPES:
        raise ValueError(f"dialecto '{dialect}' no soportado; usar uno de {sorted(SQL_TYPES)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted_table = _quote_identifier(table, dialect)
    quoted_columns = ", ".join(_quote_identifier(c, dialect) for c in df.columns)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"-- Tabla: {table} | {len(df)} filas | dialecto: {dialect}\n")

        if include_create:
            if drop_if_exists:
                handle.write(f"DROP TABLE IF EXISTS {quoted_table};\n")
            definitions = ",\n".join(
                f"  {_quote_identifier(c, dialect)} {_sql_type(df[c], dialect)}" for c in df.columns
            )
            handle.write(f"CREATE TABLE {quoted_table} (\n{definitions}\n);\n\n")

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            tuples = ",\n".join(
                "  (" + ", ".join(format_sql_value(v, dialect) for v in row) + ")"
                for row in batch.itertuples(index=False, name=None)
            )
            handle.write(f"INSERT INTO {quoted_table} ({quoted_columns}) VALUES\n{tuples};\n")

    return path
