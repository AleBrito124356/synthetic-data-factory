"""Seed determinism and stability."""
from factory import generate


def _schema(rows=200):
    return {
        "seed": 123,
        "tables": [
            {
                "name": "people",
                "rows": rows,
                "fields": [
                    {"name": "id", "type": "id"},
                    {"name": "name", "type": "name"},
                    {"name": "email", "type": "email", "depends_on": "name", "unique": True},
                    {"name": "age", "type": "int", "min": 18, "max": 90},
                    {"name": "tier", "type": "category", "categories": {"a": 0.5, "b": 0.5}},
                    {"name": "joined", "type": "date", "start": "2020-01-01", "end": "2024-12-31"},
                ],
            }
        ],
    }


def test_same_seed_is_identical():
    d1 = generate(_schema())
    d2 = generate(_schema())
    assert d1["people"] == d2["people"]


def test_explicit_seed_override_is_deterministic():
    d1 = generate(_schema(), seed=999)
    d2 = generate(_schema(), seed=999)
    assert d1["people"] == d2["people"]


def test_different_seed_changes_data():
    d1 = generate(_schema(), seed=1)
    d2 = generate(_schema(), seed=2)
    assert d1["people"] != d2["people"]


def test_adding_a_table_does_not_shift_unrelated_table():
    """Per-table seeding: adding an unrelated table must not change the values
    of an existing one. This is a design guarantee worth locking down."""
    base = {
        "seed": 55,
        "tables": [
            {
                "name": "customers",
                "rows": 100,
                "fields": [
                    {"name": "customer_id", "type": "id", "start": 1},
                    {"name": "name", "type": "name"},
                    {"name": "country", "type": "category", "categories": {"PA": 0.5, "MX": 0.5}},
                ],
            }
        ],
    }
    extended = {
        "seed": 55,
        "tables": [
            base["tables"][0],
            {
                "name": "orders",
                "rows": 300,
                "fields": [
                    {"name": "order_id", "type": "id"},
                    {"name": "customer_id", "type": "foreign_key", "references": "customers.customer_id"},
                ],
            },
        ],
    }
    d_base = generate(base)
    d_ext = generate(extended)
    assert d_base["customers"] == d_ext["customers"]


def test_uniqueness_holds_under_generation():
    dataset = generate(_schema(rows=500))
    emails = [r["email"] for r in dataset["people"]]
    assert len(emails) == len(set(emails))
