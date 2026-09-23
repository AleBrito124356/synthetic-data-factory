"""`sdf infer` + `sdf validate --data`: synthetic stand-ins from real files."""
import csv
import os
import sqlite3

import pytest
import yaml

from factory import export_dataset, generate, validate_dataset
from factory.cli import main
from factory.infer import fidelity_report, infer_schema, strength_from_rho
from factory.load import DataLoadError, load_raw, validate_data
from factory.schema import Schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECOMMERCE = os.path.join(ROOT, "schemas", "ecommerce.yaml")


@pytest.fixture(scope="module")
def ecommerce_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("eco")
    original = Schema.from_yaml(ECOMMERCE)
    export_dataset(generate(original), str(out), formats=["csv", "sqlite"])
    return out, original


def _fields(schema: Schema, table: str):
    return {f.name: f for f in schema.get_table(table).fields}


@pytest.mark.parametrize("source", ["csv", "sqlite"])
def test_round_trip_recovers_structure(ecommerce_export, source):
    out, original = ecommerce_export
    raw = load_raw(str(out) if source == "csv" else str(out / "dataset.db"))
    inferred = infer_schema(raw).schema

    # Foreign-key graph.
    refs = {(t.name, f.name): f.get("references") for t in inferred.tables for f in t.fields
            if f.type == "foreign_key"}
    assert refs == {
        ("orders", "customer_id"): "customers.customer_id",
        ("order_items", "order_id"): "orders.order_id",
        ("order_items", "product_id"): "products.product_id",
    }
    # Types.
    customers = _fields(inferred, "customers")
    assert customers["customer_id"].type == "id" and customers["customer_id"].get("start") == 1
    assert customers["email"].type == "email" and customers["email"].unique
    assert customers["email"].depends_on == "full_name"
    assert customers["full_name"].type == "name"
    assert customers["signup_date"].type == "date"
    orders = _fields(inferred, "orders")
    assert orders["order_id"].type == "id" and orders["order_id"].get("prefix") == "ORD-"
    assert orders["status"].type == "category"
    # Relations beyond FKs.
    items = _fields(inferred, "order_items")
    assert items["unit_price"].type == "lookup" and items["unit_price"].get("column") == "price"
    assert items["line_total"].type == "formula"
    assert items["line_total"].get("expr") == "round(quantity * unit_price, 2)"
    assert items["order_id"].get("min_per_parent") == 1
    assert items["product_id"].get("skew") == "zipf"
    assert orders["order_date"].get("after") == "customer_id.signup_date"
    assert _fields(inferred, "products")["cost"].get("expr") == "round(price * 0.6, 2)"

    # Category weights within 0.02 of the originals.
    for table in ("customers", "orders", "products"):
        for name, fs in _fields(original, table).items():
            if fs.type != "category":
                continue
            want = fs.get("categories")
            total = sum(want.values())
            got = _fields(inferred, table)[name].get("categories")
            for label, weight in want.items():
                assert abs(got[str(label)] - weight / total) <= 0.02, (table, name, label)

    # The regenerated data passes validation against the inferred schema.
    report = validate_dataset(generate(inferred))
    assert report.ok, report.render()


def test_inferred_schema_yaml_round_trips_through_cli(ecommerce_export, tmp_path, capsys):
    out, _ = ecommerce_export
    schema_path = tmp_path / "inferred.yaml"
    assert main(["infer", "--data", str(out), "--out", str(schema_path)]) == 0
    text = capsys.readouterr().out
    assert "foreign key: orders.customer_id -> customers.customer_id" in text
    assert "# Fidelity: real vs synthetic" in text
    synth = tmp_path / "synthetic"
    assert main(["generate", "tabular", "--schema", str(schema_path), "--out", str(synth),
                 "--format", "csv,sqlite", "--validate"]) == 0
    assert main(["validate", "--schema", str(schema_path), "--data", str(synth)]) == 0
    assert main(["validate", "--schema", str(schema_path), "--data", str(synth / "dataset.db")]) == 0
    loaded = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert [t["name"] for t in loaded["tables"]].index("customers") < \
        [t["name"] for t in loaded["tables"]].index("orders")


def test_rows_scale(ecommerce_export):
    out, _ = ecommerce_export
    schema = infer_schema(load_raw(str(out)), rows_scale=0.1).schema
    assert schema.get_table("orders").rows == 200
    assert schema.get_table("order_items").rows == 500
    assert validate_dataset(generate(schema)).ok


def test_fidelity_report(ecommerce_export):
    out, _ = ecommerce_export
    raw = load_raw(str(out))
    schema = infer_schema(raw).schema
    entries = {(e["table"], e["field"]): e for e in fidelity_report(raw, schema, generate(schema))}
    assert entries[("orders", "status")]["tvd"] <= 0.01
    price = entries[("products", "price")]
    assert abs(price["real_mean"] - price["synthetic_mean"]) / price["real_mean"] < 0.1


def test_validate_data_on_original_export(ecommerce_export):
    out, original = ecommerce_export
    report = validate_data(original, str(out))
    assert report.ok, report.render()
    assert validate_data(original, str(out / "dataset.db")).ok


def test_validate_data_catches_a_broken_foreign_key(ecommerce_export, tmp_path, capsys):
    out, _ = ecommerce_export
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in os.listdir(out):
        if name.endswith(".csv"):
            (broken / name).write_bytes((out / name).read_bytes())
    path = broken / "order_items.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["order_id"] = "ORD-999999"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert main(["validate", "--schema", ECOMMERCE, "--data", str(broken)]) == 1
    output = capsys.readouterr().out
    assert "[FAIL] foreign_key :: order_items.order_id" in output
    assert "'ORD-999999'" in output


def test_validate_data_reports_missing_columns_and_tables(ecommerce_export, tmp_path):
    out, original = ecommerce_export
    partial = tmp_path / "partial"
    partial.mkdir()
    with open(out / "customers.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(partial / "customers.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[c for c in rows[0] if c != "email"] + ["extra"])
        writer.writeheader()
        for r in rows:
            r.pop("email")
            r["extra"] = "x"
            writer.writerow(r)
    report = validate_data(original, str(partial))
    columns = {c.table: c for c in report.checks if c.name == "columns"}
    assert not columns["customers"].passed and "missing: email" in columns["customers"].detail
    assert "extra (ignored): extra" in columns["customers"].detail
    assert not columns["orders"].passed and "table not found" in columns["orders"].detail


def test_rare_categories_never_reach_the_schema(tmp_path):
    rows = ([{"id": i, "city_zone": "north"} for i in range(50)]
            + [{"id": 50 + i, "city_zone": "south"} for i in range(30)]
            + [{"id": 80, "city_zone": "Rare-Village-A"}, {"id": 81, "city_zone": "Rare-Village-A"},
               {"id": 82, "city_zone": "Rare-Village-B"}])
    with open(tmp_path / "people.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "city_zone"])
        writer.writeheader()
        writer.writerows(rows)
    result = infer_schema(load_raw(str(tmp_path)), min_category_count=5)
    cats = _fields(result.schema, "people")["city_zone"].get("categories")
    assert set(cats) == {"north", "south", "other"}
    assert abs(cats["other"] - 3 / 83) < 0.001
    assert "Rare-Village" not in result.to_yaml()
    assert result.suppressed == {"people.city_zone": 2}
    # With a threshold of 1 nothing is merged.
    result = infer_schema(load_raw(str(tmp_path)), min_category_count=1)
    assert "Rare-Village-B" in _fields(result.schema, "people")["city_zone"].get("categories")


def test_self_reference_and_type_detection_from_sqlite(tmp_path):
    db = tmp_path / "hr.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE employees (emp_id INTEGER PRIMARY KEY, manager_id INTEGER, "
                 "is_remote INTEGER, badge TEXT, hired TEXT, updated_at TEXT, salary REAL, notes TEXT)")
    for i in range(1, 61):
        conn.execute(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (i, None if i <= 2 else (i * 7) % (i - 1) + 1, i % 3 == 0,
             f"3f2504e0-4f89-11d3-9a0c-0305e82c{i:04d}", f"2020-{1 + i % 12:02d}-{1 + i % 28:02d}",
             f"2024-01-{1 + i % 28:02d} 10:{i % 60:02d}:00", 1000.0 + i * 12.5,
             None if i % 2 else "likes tea."),
        )
    conn.commit()
    conn.close()
    fields = _fields(infer_schema(load_raw(str(db))).schema, "employees")
    assert fields["emp_id"].type == "id"
    assert fields["manager_id"].type == "foreign_key"
    assert fields["manager_id"].get("references") == "employees.emp_id"
    assert fields["is_remote"].type == "bool"
    assert fields["badge"].type == "uuid"
    assert fields["hired"].type == "date"
    assert fields["updated_at"].type == "datetime" and fields["updated_at"].get("after") == "hired"
    assert fields["salary"].type == "float" and fields["salary"].get("round") == 1
    assert fields["notes"].get("null_rate") == 0.5


def test_numeric_distribution_fit(tmp_path):
    import random

    rng = random.Random(1)
    with open(tmp_path / "m.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["u", "g", "e"])
        for _ in range(2000):
            writer.writerow([round(rng.uniform(0, 100), 2), round(rng.gauss(50, 5), 2),
                             round(rng.expovariate(1 / 10), 2)])
    fields = _fields(infer_schema(load_raw(str(tmp_path))).schema, "m")
    assert fields["u"].get("distribution") is None          # uniform
    assert fields["g"].get("distribution") == "normal"
    assert abs(fields["g"].get("mean") - 50) < 1
    assert fields["e"].get("distribution") == "exponential"


def test_strength_from_rho_inverts_the_generator_blend():
    for s in (0.2, 0.5, 0.75, 0.9):
        rho = s / ((s * s + (1 - s) ** 2) ** 0.5)
        assert abs(strength_from_rho(rho) - s) < 0.01


def test_load_errors_are_clear(tmp_path):
    with pytest.raises(DataLoadError, match="no .csv or .jsonl"):
        load_raw(str(tmp_path))
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(DataLoadError, match="expected a folder"):
        load_raw(str(tmp_path / "x.txt"))


def test_infer_from_jsonl_export(tmp_path):
    export_dataset(generate(os.path.join(ROOT, "schemas", "saas-users.yaml")), str(tmp_path), formats=["jsonl"])
    result = infer_schema(load_raw(str(tmp_path)))
    relations = "\n".join(result.relations)
    assert "users.account_id -> accounts.account_id" in relations
    assert "events.user_id -> users.user_id" in relations
    assert "events.ts before user_id.last_login" in relations
    assert validate_dataset(generate(result.schema)).ok
