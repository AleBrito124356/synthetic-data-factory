"""Tabular export: a usable SQLite database and well-formed flat files."""
import csv
import os
import sqlite3

import pytest

from factory import Dataset, export_dataset, generate
from factory.export import ExportError, sqlite_column_types
from factory.schema import Schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED = ["ecommerce.yaml", "healthcare.yaml", "saas-users.yaml"]


def _schema_path(name):
    return os.path.join(ROOT, "schemas", name)


@pytest.mark.parametrize("name", SHIPPED)
def test_sqlite_passes_foreign_key_check(tmp_path, name):
    export_dataset(generate(_schema_path(name)), str(tmp_path), formats=["sqlite"])
    conn = sqlite3.connect(tmp_path / "dataset.db")
    conn.execute("PRAGMA foreign_keys=ON")
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_sqlite_numeric_columns_compare_as_numbers(tmp_path):
    dataset = generate(_schema_path("ecommerce.yaml"))
    export_dataset(dataset, str(tmp_path), formats=["sqlite"])
    conn = sqlite3.connect(tmp_path / "dataset.db")

    expected = sum(1 for r in dataset["order_items"] if r["line_total"] > 1000)
    got = conn.execute("SELECT count(*) FROM order_items WHERE line_total > 1000").fetchone()[0]
    assert got == expected  # was 4982 vs 1003 with TEXT affinity

    assert conn.execute("SELECT max(customer_id) FROM customers").fetchone()[0] == 500
    first = [r[0] for r in conn.execute("SELECT customer_id FROM customers ORDER BY customer_id LIMIT 3")]
    assert first == [1, 2, 3]
    assert conn.execute("SELECT typeof(customer_id) FROM orders LIMIT 1").fetchone()[0] == "integer"
    assert conn.execute("SELECT typeof(cost) FROM products LIMIT 1").fetchone()[0] == "real"
    assert conn.execute("SELECT typeof(line_total) FROM order_items LIMIT 1").fetchone()[0] == "real"


def test_sqlite_fk_to_uuid_parent_is_enforceable(tmp_path):
    dataset = generate(_schema_path("healthcare.yaml"))
    export_dataset(dataset, str(tmp_path), formats=["sqlite"])
    conn = sqlite3.connect(tmp_path / "dataset.db")
    conn.execute("PRAGMA foreign_keys=ON")
    parent = dataset["encounters"][0]["patient_uuid"]
    # Used to raise "foreign key mismatch"; now SQLite enforces the constraint.
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute("DELETE FROM patients WHERE patient_uuid = ?", (parent,))


def test_sqlite_types_are_inherited_through_foreign_keys():
    types = sqlite_column_types(generate(_schema_path("ecommerce.yaml")))
    assert types["customers"]["customer_id"] == "INTEGER"
    assert types["orders"]["customer_id"] == "INTEGER"   # FK inherits parent type
    assert types["orders"]["order_id"] == "TEXT"         # prefixed id
    assert types["order_items"]["order_id"] == "TEXT"
    assert types["products"]["cost"] == "REAL"           # formula typed from values


def test_sqlite_tables_created_parents_first(tmp_path):
    export_dataset(generate(_schema_path("ecommerce.yaml")), str(tmp_path), formats=["sqlite"])
    conn = sqlite3.connect(tmp_path / "dataset.db")
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid")]
    assert names.index("customers") < names.index("orders") < names.index("order_items")
    assert names.index("products") < names.index("order_items")
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_order_items_order_id" in indexes


def test_sqlite_export_refuses_orphans(tmp_path):
    schema = Schema.from_dict(
        {
            "tables": [
                {"name": "p", "rows": 2, "fields": [{"name": "id", "type": "id"}]},
                {"name": "c", "rows": 1, "fields": [{"name": "pid", "type": "foreign_key", "references": "p.id"}]},
            ]
        }
    )
    broken = Dataset(schema=schema, tables={"p": [{"id": 1}, {"id": 2}], "c": [{"pid": 99}]})
    with pytest.raises(ExportError, match="foreign-key violation"):
        export_dataset(broken, str(tmp_path), formats=["sqlite"])


def test_sqlite_fk_to_non_unique_column_is_skipped_with_warning(tmp_path):
    schema = {
        "seed": 2,
        "tables": [
            {
                "name": "p",
                "rows": 20,
                "fields": [
                    {"name": "id", "type": "id"},
                    {"name": "grp", "type": "category", "categories": ["a", "b"]},
                ],
            },
            {"name": "c", "rows": 5, "fields": [{"name": "g", "type": "foreign_key", "references": "p.grp"}]},
        ],
    }
    with pytest.warns(UserWarning, match="duplicate values"):
        export_dataset(generate(schema), str(tmp_path), formats=["sqlite"])
    conn = sqlite3.connect(tmp_path / "dataset.db")
    conn.execute("PRAGMA foreign_keys=ON")
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_empty_table_gets_csv_header_and_typed_parquet(tmp_path):
    schema = {
        "tables": [
            {
                "name": "empty",
                "rows": 0,
                "fields": [{"name": "id", "type": "id"}, {"name": "x", "type": "int"}],
            }
        ]
    }
    export_dataset(generate(schema), str(tmp_path), formats=["csv", "parquet", "sqlite"])
    with open(tmp_path / "empty.csv", newline="", encoding="utf-8") as fh:
        assert fh.read().strip() == "id,x"  # was just "\n"
    import pandas as pd

    assert list(pd.read_parquet(tmp_path / "empty.parquet").columns) == ["id", "x"]


def test_csv_columns_follow_schema_order(tmp_path):
    dataset = generate(_schema_path("ecommerce.yaml"))
    export_dataset(dataset, str(tmp_path), formats=["csv", "jsonl"])
    with open(tmp_path / "order_items.csv", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == dataset.schema.get_table("order_items").field_names()
