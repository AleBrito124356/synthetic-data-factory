"""Relational realism: lookup, after/before, when, min_per_parent, skew."""
import copy
import datetime as dt
import os
import sqlite3
from collections import Counter

import pytest

from factory import Dataset, export_dataset, generate, validate_dataset
from factory.schema import Schema, SchemaError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shop(**order_opts):
    """customers -> orders, products -> items; options patched per test."""
    orders_date = {"name": "order_date", "type": "date", "start": "2023-01-01", "end": "2024-12-31",
                   "after": "customer_id.signup_date"}
    orders_date.update(order_opts)
    return {
        "seed": 11,
        "tables": [
            {"name": "customers", "rows": 60, "fields": [
                {"name": "customer_id", "type": "id"},
                {"name": "signup_date", "type": "date", "start": "2021-01-01", "end": "2024-06-30"},
            ]},
            {"name": "products", "rows": 20, "fields": [
                {"name": "sku", "type": "id", "prefix": "SKU-"},
                {"name": "price", "type": "float", "min": 5, "max": 100, "round": 2},
            ]},
            {"name": "orders", "rows": 300, "fields": [
                {"name": "order_id", "type": "id"},
                {"name": "customer_id", "type": "foreign_key", "references": "customers.customer_id"},
                orders_date,
                {"name": "shipped_at", "type": "datetime", "after": "order_date",
                 "min_days": 1, "max_days": 5},
            ]},
            {"name": "items", "rows": 900, "fields": [
                {"name": "item_id", "type": "id"},
                {"name": "order_id", "type": "foreign_key", "references": "orders.order_id",
                 "min_per_parent": 2},
                {"name": "sku", "type": "foreign_key", "references": "products.sku", "skew": "zipf"},
                {"name": "unit_price", "type": "lookup", "via": "sku", "column": "price"},
                {"name": "discount", "type": "float", "min": 0, "max": 0.3, "round": 2},
                {"name": "net", "type": "formula", "expr": "round(unit_price * (1 - discount), 2)"},
            ]},
        ],
    }


def _index(rows, key):
    return {r[key]: r for r in rows}


# --------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------
def test_lookup_copies_the_parent_rows_column():
    ds = generate(_shop())
    products = _index(ds["products"], "sku")
    for item in ds["items"]:
        assert item["unit_price"] == products[item["sku"]]["price"]
        assert item["net"] == round(item["unit_price"] * (1 - item["discount"]), 2)
    report = validate_dataset(ds)
    assert report.ok, report.render()
    assert any(c.name == "lookup_consistency" for c in report.checks)


def test_lookup_consistency_flags_a_tampered_value():
    ds = generate(_shop())
    tables = copy.deepcopy(ds.tables)
    tables["items"][0]["unit_price"] += 1
    report = validate_dataset(Dataset(schema=ds.schema, tables=tables))
    assert [c.name for c in report.failures()] == ["lookup_consistency"]


def test_lookup_through_self_reference():
    schema = {"tables": [{"name": "emp", "rows": 50, "fields": [
        {"name": "id", "type": "id"},
        {"name": "dept", "type": "category", "categories": ["ops", "eng", "sales"]},
        {"name": "boss", "type": "foreign_key", "references": "emp.id"},
        {"name": "boss_dept", "type": "lookup", "via": "boss", "column": "dept"},
    ]}]}
    ds = generate(schema)
    by_id = _index(ds["emp"], "id")
    for r in ds["emp"]:
        expected = by_id[r["boss"]]["dept"] if r["boss"] is not None else None
        assert r["boss_dept"] == expected
    assert validate_dataset(ds).ok


# --------------------------------------------------------------------------
# after / before
# --------------------------------------------------------------------------
def test_after_parent_column_and_sibling_with_gap():
    ds = generate(_shop())
    customers = _index(ds["customers"], "customer_id")
    for o in ds["orders"]:
        signup = dt.date.fromisoformat(customers[o["customer_id"]]["signup_date"])
        ordered = dt.date.fromisoformat(o["order_date"])
        assert signup <= ordered <= dt.date(2024, 12, 31)
        assert ordered >= dt.date(2023, 1, 1)
        shipped = dt.datetime.fromisoformat(o["shipped_at"])
        gap = shipped - dt.datetime.combine(ordered, dt.time.min)
        assert dt.timedelta(days=1) <= gap <= dt.timedelta(days=5)


def test_temporal_order_flags_a_violation():
    ds = generate(_shop())
    tables = copy.deepcopy(ds.tables)
    tables["orders"][0]["shipped_at"] = tables["orders"][0]["order_date"] + " 00:00:01"
    failed = validate_dataset(Dataset(schema=ds.schema, tables=tables)).failures()
    assert [(c.name, c.field) for c in failed] == [("temporal_order", "shipped_at")]


def test_before_anchor():
    schema = {"seed": 3, "tables": [
        {"name": "users", "rows": 40, "fields": [
            {"name": "id", "type": "id"},
            {"name": "last_seen", "type": "datetime", "start": "2024-01-01", "end": "2024-12-31"},
        ]},
        {"name": "events", "rows": 400, "fields": [
            {"name": "id", "type": "id"},
            {"name": "user", "type": "foreign_key", "references": "users.id"},
            {"name": "ts", "type": "datetime", "start": "2024-01-01", "before": "user.last_seen"},
        ]},
    ]}
    ds = generate(schema)
    users = _index(ds["users"], "id")
    assert all(e["ts"] <= users[e["user"]]["last_seen"] for e in ds["events"])
    assert all(e["ts"] >= "2024-01-01" for e in ds["events"])
    assert validate_dataset(ds).ok


def test_after_without_end_uses_a_default_window():
    schema = {"tables": [{"name": "t", "rows": 100, "fields": [
        {"name": "a", "type": "date", "start": "2024-01-01", "end": "2024-01-31"},
        {"name": "b", "type": "date", "after": "a"},
    ]}]}
    rows = generate(schema)["t"]
    for r in rows:
        gap = (dt.date.fromisoformat(r["b"]) - dt.date.fromisoformat(r["a"])).days
        assert 0 <= gap <= 365
    assert validate_dataset(generate(schema)).ok


@pytest.mark.parametrize(
    "opts, message",
    [
        ({"end": "2024-03-01"}, "can be as late as 2024-06-30"),
        ({"after": "customer_id.nope"}, "neither a sibling field"),
        ({"after": "order_id"}, "not a date"),
        ({"max_days": 10, "start": "2024-01-01"}, "can never reach start"),
        ({"min_days": 5, "max_days": 2}, "greater than max_days"),
    ],
)
def test_infeasible_or_invalid_anchors_fail_at_parse_time(opts, message):
    with pytest.raises(SchemaError, match=message):
        Schema.from_dict(_shop(**opts))


# --------------------------------------------------------------------------
# when
# --------------------------------------------------------------------------
def _plans():
    return {"seed": 5, "tables": [{"name": "accounts", "rows": 600, "fields": [
        {"name": "id", "type": "id"},
        {"name": "plan", "type": "category", "categories": {"free": 0.5, "pro": 0.3, "enterprise": 0.2}},
        {"name": "seats", "type": "int", "min": 1, "max": 500,
         "when": {"field": "plan", "cases": {"free": {"min": 1, "max": 3},
                                             "enterprise": {"min": 50, "max": 500}}}},
        {"name": "mrr", "type": "float", "min": 0, "max": 20000, "round": 2,
         "correlate": {"field": "seats", "strength": 0.9},
         "when": {"field": "plan", "cases": {"free": {"min": 0, "max": 0},
                                             "pro": {"min": 50, "max": 3000}}}},
        {"name": "support", "type": "category", "categories": {"email": 1, "phone": 1},
         "when": {"field": "plan", "cases": {"enterprise": {"categories": {"phone": 0.9, "dedicated": 0.1}}}}},
        {"name": "sso", "type": "bool", "true_rate": 0.1,
         "when": {"field": "plan", "cases": {"enterprise": {"true_rate": 1.0}}}},
    ]}]}


def test_when_overrides_apply_per_case():
    ds = generate(_plans())
    rows = ds["accounts"]
    by_plan = {p: [r for r in rows if r["plan"] == p] for p in ("free", "pro", "enterprise")}
    assert all(1 <= r["seats"] <= 3 and r["mrr"] == 0 for r in by_plan["free"])
    assert all(50 <= r["mrr"] <= 3000 for r in by_plan["pro"])
    assert all(50 <= r["seats"] <= 500 and r["sso"] for r in by_plan["enterprise"])
    ent_support = Counter(r["support"] for r in by_plan["enterprise"])
    assert ent_support == {"phone": 108, "dedicated": 12}  # exact per-group quotas
    assert set(r["support"] for r in by_plan["free"]) <= {"email", "phone"}
    report = validate_dataset(ds)
    assert report.ok, report.render()
    assert {c.field for c in report.checks if c.name == "when_bounds"} == {"seats", "mrr", "support", "sso"}


def test_correlation_holds_within_each_case():
    rows = [r for r in generate(_plans())["accounts"] if r["plan"] == "pro"]
    xs, ys = [r["seats"] for r in rows], [r["mrr"] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    corr = cov / ((sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5)
    assert corr > 0.6


def test_when_bounds_flags_a_free_account_that_pays():
    ds = generate(_plans())
    tables = copy.deepcopy(ds.tables)
    free = next(r for r in tables["accounts"] if r["plan"] == "free")
    free["mrr"] = 99.0
    failed = validate_dataset(Dataset(schema=ds.schema, tables=tables)).failures()
    assert [(c.name, c.field) for c in failed] == [("when_bounds", "mrr")]
    assert "free: " in failed[0].detail


@pytest.mark.parametrize(
    "when, message",
    [
        ({"field": "plan", "cases": {"gold": {"min": 1}}}, "not a value of 'plan'"),
        ({"field": "plan", "cases": {"free": {"categories": ["x"]}}}, "cannot override categories"),
        ({"field": "nope", "cases": {"free": {"min": 1}}}, "not another field"),
        ({"field": "id", "cases": {"1": {"min": 1}}}, "must be a category or bool"),
        ({"field": "plan", "cases": {"free": {"min": 9, "max": 1}}}, "greater than max"),
    ],
)
def test_invalid_when_fails_at_parse_time(when, message):
    schema = _plans()
    schema["tables"][0]["fields"][2]["when"] = when
    with pytest.raises(SchemaError, match=message):
        Schema.from_dict(schema)


# --------------------------------------------------------------------------
# min_per_parent and skew
# --------------------------------------------------------------------------
def test_min_per_parent_covers_every_parent():
    ds = generate(_shop())
    counts = Counter(i["order_id"] for i in ds["items"])
    assert len(ds["items"]) == 900
    assert all(counts[o["order_id"]] >= 2 for o in ds["orders"])


def test_min_per_parent_survives_null_rate():
    schema = _shop()
    schema["tables"][3]["fields"][1]["null_rate"] = 0.3
    ds = generate(schema)
    counts = Counter(i["order_id"] for i in ds["items"] if i["order_id"] is not None)
    assert all(counts[o["order_id"]] >= 2 for o in ds["orders"])
    assert any(i["order_id"] is None for i in ds["items"])
    assert validate_dataset(ds).ok


def test_parent_coverage_flags_a_childless_parent():
    ds = generate(_shop())
    tables = copy.deepcopy(ds.tables)
    victim = tables["orders"][0]["order_id"]
    other = tables["orders"][1]["order_id"]
    for item in tables["items"]:
        if item["order_id"] == victim:
            item["order_id"] = other
    failed = validate_dataset(Dataset(schema=ds.schema, tables=tables)).failures()
    assert [c.name for c in failed] == ["parent_coverage"]


def test_zipf_skew_concentrates_popularity():
    ds = generate(_shop())
    counts = sorted(Counter(i["sku"] for i in ds["items"]).values(), reverse=True)
    assert counts[0] > 3 * (900 / 20)          # uniform would give ~45 each
    assert counts[0] > 5 * counts[len(counts) // 2]


def test_uniform_fk_without_options_is_unchanged():
    """No options -> the exact same draws as before these features existed."""
    import random

    schema = {"seed": 2, "tables": [
        {"name": "p", "rows": 10, "fields": [{"name": "id", "type": "id"}]},
        {"name": "c", "rows": 50, "fields": [{"name": "pid", "type": "foreign_key", "references": "p.id"}]},
    ]}
    rng = random.Random("2:c")
    expected = [rng.choice(list(range(1, 11))) for _ in range(50)]
    assert [r["pid"] for r in generate(schema)["c"]] == expected


@pytest.mark.parametrize(
    "fk_opts, message",
    [
        ({"min_per_parent": 4}, "needs at least 1200 rows"),
        ({"skew": "pareto"}, "skew must be one of"),
        ({"min_per_parent": 1, "unique": True}, "cannot be combined with unique"),
        ({"min_per_parent": -1}, "non-negative integer"),
    ],
)
def test_invalid_fk_options(fk_opts, message):
    schema = _shop()
    fk = schema["tables"][3]["fields"][1]
    fk.pop("min_per_parent")
    fk.update(fk_opts)
    with pytest.raises(SchemaError, match=message):
        Schema.from_dict(schema)


def test_lookup_needs_a_unique_parent_key():
    schema = {"tables": [
        {"name": "p", "rows": 5, "fields": [
            {"name": "grp", "type": "category", "categories": ["a", "b"]},
            {"name": "v", "type": "int"},
        ]},
        {"name": "c", "rows": 5, "fields": [
            {"name": "g", "type": "foreign_key", "references": "p.grp"},
            {"name": "v", "type": "lookup", "via": "g", "column": "v"},
        ]},
    ]}
    with pytest.raises(SchemaError, match="not unique"):
        Schema.from_dict(schema)
    schema["tables"][1]["fields"][1]["via"] = "v"
    with pytest.raises(SchemaError, match="must name a foreign_key"):
        Schema.from_dict(schema)


# --------------------------------------------------------------------------
# determinism and the shipped schemas
# --------------------------------------------------------------------------
def test_relational_options_are_deterministic_and_keep_per_table_seeding():
    assert generate(_shop()).tables == generate(_shop()).tables
    extended = _shop()
    extended["tables"].append({"name": "unrelated", "rows": 10, "fields": [{"name": "id", "type": "id"}]})
    base, ext = generate(_shop()), generate(extended)
    for name in ("customers", "products", "orders", "items"):
        assert base[name] == ext[name]


def test_shipped_schemas_are_relationally_coherent(tmp_path):
    for name in ("ecommerce", "saas-users"):
        export_dataset(generate(os.path.join(ROOT, "schemas", f"{name}.yaml")), str(tmp_path / name),
                       formats=["sqlite"])
    eco = sqlite3.connect(tmp_path / "ecommerce" / "dataset.db")
    q = lambda c, sql: c.execute(sql).fetchone()[0]  # noqa: E731
    assert q(eco, "SELECT count(*) FROM orders o JOIN customers c USING (customer_id) "
                  "WHERE o.order_date < c.signup_date") == 0          # was 579
    assert q(eco, "SELECT count(*) FROM order_items i JOIN products p USING (product_id) "
                  "WHERE i.unit_price != p.price") == 0               # was 5000
    assert q(eco, "SELECT count(*) FROM orders o WHERE NOT EXISTS "
                  "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)") == 0  # was 159

    saas = sqlite3.connect(tmp_path / "saas-users" / "dataset.db")
    assert q(saas, "SELECT count(*) FROM accounts WHERE plan = 'free' AND mrr > 0") == 0  # was 165
    lo, hi = saas.execute("SELECT min(seats), max(seats) FROM accounts WHERE plan = 'enterprise'").fetchone()
    assert 50 <= lo and hi <= 250
    assert q(saas, "SELECT count(*) FROM users u JOIN accounts a USING (account_id) "
                   "WHERE u.last_login < a.created_at") == 0
    assert q(saas, "SELECT count(*) FROM events e JOIN users u USING (user_id) "
                   "WHERE e.ts > u.last_login") == 0
