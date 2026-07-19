"""Schema parsing and validation. No LLM, no network."""
import glob
import os

import pytest

from factory.schema import Schema, SchemaError


def test_parses_minimal_schema():
    schema = Schema.from_dict(
        {
            "seed": 1,
            "tables": [
                {
                    "name": "t",
                    "rows": 3,
                    "fields": [
                        {"name": "id", "type": "id"},
                        {"name": "score", "type": "int", "min": 0, "max": 10},
                    ],
                }
            ],
        }
    )
    assert schema.seed == 1
    table = schema.get_table("t")
    assert table is not None
    assert table.rows == 3
    assert table.field_names() == ["id", "score"]


def test_type_aliases_normalized():
    schema = Schema.from_dict(
        {
            "tables": [
                {
                    "name": "t",
                    "rows": 1,
                    "fields": [
                        {"name": "a", "type": "integer"},
                        {"name": "b", "type": "categorical", "categories": ["x", "y"]},
                        {"name": "c", "type": "fk", "references": "t.a"},
                    ],
                }
            ]
        }
    )
    fields = {f.name: f.type for f in schema.get_table("t").fields}
    assert fields["a"] == "int"
    assert fields["b"] == "category"
    assert fields["c"] == "foreign_key"


def test_category_list_becomes_weighted_dict():
    schema = Schema.from_dict(
        {
            "tables": [
                {
                    "name": "t",
                    "rows": 1,
                    "fields": [{"name": "c", "type": "category", "categories": ["a", "b", "c"]}],
                }
            ]
        }
    )
    cats = schema.get_table("t").get_field("c").get("categories")
    assert cats == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_unknown_type_raises():
    with pytest.raises(SchemaError):
        Schema.from_dict(
            {"tables": [{"name": "t", "rows": 1, "fields": [{"name": "x", "type": "wat"}]}]}
        )


def test_missing_rows_raises():
    with pytest.raises(SchemaError):
        Schema.from_dict({"tables": [{"name": "t", "fields": [{"name": "x", "type": "id"}]}]})


def test_dangling_foreign_key_raises():
    with pytest.raises(SchemaError):
        Schema.from_dict(
            {
                "tables": [
                    {
                        "name": "t",
                        "rows": 1,
                        "fields": [{"name": "fk", "type": "foreign_key", "references": "ghost.id"}],
                    }
                ]
            }
        )


def test_duplicate_field_raises():
    with pytest.raises(SchemaError):
        Schema.from_dict(
            {
                "tables": [
                    {
                        "name": "t",
                        "rows": 1,
                        "fields": [
                            {"name": "x", "type": "id"},
                            {"name": "x", "type": "int"},
                        ],
                    }
                ]
            }
        )


def test_cyclic_foreign_keys_raise():
    with pytest.raises(SchemaError):
        Schema.from_dict(
            {
                "tables": [
                    {
                        "name": "a",
                        "rows": 1,
                        "fields": [{"name": "fk", "type": "foreign_key", "references": "b.id"}],
                    },
                    {
                        "name": "b",
                        "rows": 1,
                        "fields": [{"name": "fk", "type": "foreign_key", "references": "a.id"}],
                    },
                ]
            }
        )


def test_negative_category_weight_raises():
    with pytest.raises(SchemaError):
        Schema.from_dict(
            {
                "tables": [
                    {
                        "name": "t",
                        "rows": 1,
                        "fields": [{"name": "c", "type": "category", "categories": {"a": -1, "b": 1}}],
                    }
                ]
            }
        )


def test_shipped_schemas_parse():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tabular = ["ecommerce.yaml", "saas-users.yaml", "healthcare.yaml"]
    for name in tabular:
        path = os.path.join(here, "schemas", name)
        schema = Schema.from_yaml(path)
        assert schema.tables, f"{name} produced no tables"
