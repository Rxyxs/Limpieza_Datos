"""Pruebas unitarias para src/toolkit/sql_dump.py.

Los dumps de prueba son fragmentos con la sintaxis exacta que emiten `mysqldump`
(INSERT multi-fila, backticks, escapes con backslash) y `pg_dump` (bloques
`COPY ... FROM stdin` con `\\N` como nulo).
"""
import pandas as pd
import pytest

from src.toolkit.sql_dump import (
    dataframe_to_sql_dump,
    format_sql_value,
    inventory_sql_dump,
    iter_sql_statements,
    parse_copy_block,
    parse_create_table,
    parse_insert,
    parse_sql_literal,
    read_sql_dump,
)

MYSQL_DUMP = """
-- Volcado generado por mysqldump
/*!40101 SET NAMES utf8 */;

DROP TABLE IF EXISTS `faena`;
CREATE TABLE `faena` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `empresa` varchar(120) DEFAULT NULL,
  `produccion_ton` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_empresa` (`empresa`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `faena` VALUES (1,'Codelco; Division Norte',140.5),(2,'O\\'Higgins',98.2),(3,NULL,NULL);
INSERT INTO `faena` VALUES (4,'NULL',0);
"""

PG_DUMP = """
--
-- PostgreSQL database dump
--

CREATE TABLE public.faena (
    id integer NOT NULL,
    empresa text,
    produccion_ton double precision
);

COPY public.faena (id, empresa, produccion_ton) FROM stdin;
1\tCodelco\t140.5
2\tEscondida\t\\N
\\.
"""


def test_iter_sql_statements_ignores_semicolons_inside_string_literals():
    statements = list(iter_sql_statements(MYSQL_DUMP))
    inserts = [s for s in statements if s.upper().startswith("INSERT")]

    assert len(inserts) == 2
    assert "Codelco; Division Norte" in inserts[0]


def test_iter_sql_statements_strips_comments():
    joined = " ".join(iter_sql_statements(MYSQL_DUMP))

    assert "mysqldump" not in joined
    assert "SET NAMES" not in joined  # comentario condicional /*! ... */


def test_iter_sql_statements_emits_copy_block_with_its_data():
    copy_chunks = [s for s in iter_sql_statements(PG_DUMP) if s.upper().startswith("COPY")]

    assert len(copy_chunks) == 1
    assert "Escondida" in copy_chunks[0]
    assert copy_chunks[0].rstrip().endswith("\\.")


def test_parse_create_table_keeps_columns_and_drops_constraints():
    statement = next(s for s in iter_sql_statements(MYSQL_DUMP) if s.upper().startswith("CREATE"))
    table, columns = parse_create_table(statement)

    assert table == "faena"
    assert columns == ["id", "empresa", "produccion_ton"]


def test_parse_insert_reads_every_tuple_of_a_multi_row_insert():
    statement = next(s for s in iter_sql_statements(MYSQL_DUMP) if s.upper().startswith("INSERT"))
    table, columns, rows = parse_insert(statement)

    assert table == "faena"
    assert columns is None  # este INSERT no declara columnas
    assert len(rows) == 3
    assert rows[0] == [1, "Codelco; Division Norte", 140.5]
    assert rows[1][1] == "O'Higgins"  # comilla escapada con backslash (MySQL)
    assert rows[2] == [3, None, None]


def test_parse_sql_literal_separates_null_from_the_text_null():
    assert parse_sql_literal("NULL") is None
    assert parse_sql_literal("'NULL'") == "NULL"
    assert parse_sql_literal("TRUE") is True
    assert parse_sql_literal("42") == 42
    assert parse_sql_literal("3.5") == 3.5
    assert parse_sql_literal("'O''Higgins'") == "O'Higgins"


def test_parse_copy_block_reads_tsv_body_and_backslash_n_as_null():
    chunk = next(s for s in iter_sql_statements(PG_DUMP) if s.upper().startswith("COPY"))
    table, columns, rows = parse_copy_block(chunk)

    assert table == "faena"
    assert columns == ["id", "empresa", "produccion_ton"]
    assert rows[0] == ["1", "Codelco", "140.5"]
    assert rows[1][2] is None


def test_read_sql_dump_builds_a_dataframe_with_schema_column_names():
    tables = read_sql_dump(MYSQL_DUMP)

    assert set(tables) == {"faena"}
    faena = tables["faena"]
    assert list(faena.columns) == ["id", "empresa", "produccion_ton"]
    assert len(faena) == 4
    assert faena["produccion_ton"].isna().sum() == 1
    assert faena.loc[faena["id"] == 4, "empresa"].item() == "NULL"


def test_read_sql_dump_reads_a_postgres_copy_dump():
    faena = read_sql_dump(PG_DUMP)["faena"]

    assert list(faena.columns) == ["id", "empresa", "produccion_ton"]
    assert len(faena) == 2
    assert faena["produccion_ton"].isna().sum() == 1


def test_read_sql_dump_can_filter_tables():
    dump = MYSQL_DUMP + "\nINSERT INTO `otra` VALUES (1,'x');\n"

    assert set(read_sql_dump(dump)) == {"faena", "otra"}
    assert set(read_sql_dump(dump, tables=["faena"])) == {"faena"}


def test_inventory_sql_dump_counts_rows_per_table():
    inventory = inventory_sql_dump(MYSQL_DUMP).set_index("tabla")

    assert inventory.loc["faena", "n_filas"] == 4
    assert inventory.loc["faena", "n_columnas"] == 3
    assert inventory.loc["faena", "origen"] == "INSERT"


def test_format_sql_value_escapes_per_dialect():
    assert format_sql_value(None) == "NULL"
    assert format_sql_value(float("nan")) == "NULL"
    assert format_sql_value("O'Higgins") == "'O''Higgins'"
    assert format_sql_value("C:\\ruta", dialect="mysql") == "'C:\\\\ruta'"
    assert format_sql_value("C:\\ruta", dialect="postgres") == "'C:\\ruta'"
    assert format_sql_value(True) == "TRUE"
    assert format_sql_value(True, dialect="sqlite") == "1"
    assert format_sql_value(pd.Timestamp("2024-01-01")) == "'2024-01-01 00:00:00'"


@pytest.mark.parametrize("dialect", ["postgres", "mysql", "sqlite"])
def test_dataframe_to_sql_dump_roundtrips_through_read_sql_dump(tmp_path, dialect):
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "empresa": ["Codelco; Division Norte", "O'Higgins", None],
        "produccion_ton": [140.5, 98.2, float("nan")],
    })
    path = dataframe_to_sql_dump(df, "faena", tmp_path / f"{dialect}.sql", dialect=dialect)

    back = read_sql_dump(path)["faena"]
    assert list(back.columns) == ["id", "empresa", "produccion_ton"]
    pd.testing.assert_frame_equal(back, df, check_dtype=False)


def test_roundtrip_survives_column_names_with_parentheses_and_dots(tmp_path):
    # Nombres reales de columna del .xlsx de COCHILCO que usa este repo. Ambos
    # rompen el parseo ingenuo: el parentesis cierra antes de tiempo la lista de
    # columnas del INSERT, y el punto se confunde con el separador de esquema
    # (`esquema.tabla`), dejando la columna como "Tomic".
    df = pd.DataFrame({
        "Chuqui y R.Tomic": [54.4, 48.5],
        "Centinela (sulfuros)": [12.1, 11.8],
        "Enami (Plantas)": [1.0, 1.2],
    })
    path = dataframe_to_sql_dump(df, "faena", tmp_path / "cochilco.sql")

    back = read_sql_dump(path)["faena"]
    assert list(back.columns) == ["Chuqui y R.Tomic", "Centinela (sulfuros)", "Enami (Plantas)"]
    pd.testing.assert_frame_equal(back, df, check_dtype=False)


def test_parse_create_table_keeps_a_quoted_column_named_like_a_keyword():
    statement = 'CREATE TABLE t ("key" text, "index" int, PRIMARY KEY ("key"))'
    table, columns = parse_create_table(statement)

    assert table == "t"
    assert columns == ["key", "index"]  # la restriccion PRIMARY KEY no es columna


def test_dataframe_to_sql_dump_batches_inserts(tmp_path):
    df = pd.DataFrame({"id": range(10)})
    path = dataframe_to_sql_dump(df, "t", tmp_path / "batched.sql", batch_size=4)

    text = path.read_text(encoding="utf-8")
    assert text.upper().count("INSERT INTO") == 3  # 4 + 4 + 2 filas
    assert read_sql_dump(path)["t"].shape == (10, 1)


def test_dataframe_to_sql_dump_rejects_unknown_dialect(tmp_path):
    with pytest.raises(ValueError, match="dialecto"):
        dataframe_to_sql_dump(pd.DataFrame({"a": [1]}), "t", tmp_path / "x.sql", dialect="oracle")
