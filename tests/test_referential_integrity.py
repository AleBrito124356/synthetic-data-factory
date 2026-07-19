"""Foreign keys must resolve; validation must confirm it."""
import os

from factory import generate, validate_dataset
from factory.schema import Schema


def _related_schema():
    return {
        "seed": 5,
        "tables": [
            {
                "name": "customers",
                "rows": 50,
                "fields": [
                    {"name": "customer_id", "type": "id", "start": 1},
                    {"name": "name", "type": "name"},
                ],
            },
            {
                "name": "orders",
                "rows": 300,
                "fields": [
                    {"name": "order_id", "type": "id", "start": 1000},
                    {"name": "customer_id", "type": "foreign_key", "references": "customers.customer_id"},
                    {"name": "amount", "type": "float", "min": 1, "max": 100, "round": 2},
                ],
            },
        ],
    }


def test_every_fk_value_exists_in_parent():
    dataset = generate(_related_schema())
    parent_ids = {r["customer_id"] for r in dataset["customers"]}
    child_ids = {r["customer_id"] for r in dataset["orders"]}
    assert child_ids.issubset(parent_ids)
    # No orphan rows at all.
    assert all(r["customer_id"] in parent_ids for r in dataset["orders"])


def test_validation_passes_on_related_dataset():
    dataset = generate(_related_schema())
    report = validate_dataset(dataset)
    assert report.ok, report.render()


def test_unique_foreign_key_is_one_to_one():
    schema = {
        "seed": 9,
        "tables": [
            {"name": "users", "rows": 40, "fields": [{"name": "user_id", "type": "id"}]},
            {
                "name": "profiles",
                "rows": 40,
                "fields": [
                    {"name": "profile_id", "type": "id"},
                    {
                        "name": "user_id",
                        "type": "foreign_key",
                        "references": "users.user_id",
                        "unique": True,
                    },
                ],
            },
        ],
    }
    dataset = generate(schema)
    fk_values = [r["user_id"] for r in dataset["profiles"]]
    assert len(fk_values) == len(set(fk_values))  # no repeats -> 1:1


def test_three_level_fk_chain_from_shipped_schema():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema = Schema.from_yaml(os.path.join(here, "schemas", "ecommerce.yaml"))
    dataset = generate(schema)
    order_ids = {r["order_id"] for r in dataset["orders"]}
    product_ids = {r["product_id"] for r in dataset["products"]}
    for item in dataset["order_items"]:
        assert item["order_id"] in order_ids
        assert item["product_id"] in product_ids
    assert validate_dataset(dataset).ok
