"""Regression tests for bugs found in the 0.1.0 audit.

Every test here failed (or hung, or crashed the interpreter) on the original
code. They lock in the fixes to the tabular core, formulas and validator.
"""
import os
import re
import time
from collections import Counter

import pytest

from factory import Dataset, generate, validate_dataset
from factory.formula import Formula, FormulaError
from factory.providers import is_reserved_email
from factory.schema import Schema, SchemaError


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _one_table(fields, rows=50, seed=1, name="t"):
    return {"seed": seed, "tables": [{"name": name, "rows": rows, "fields": fields}]}


# --------------------------------------------------------------------------
# uniqueness keeps values well-formed and typed
# --------------------------------------------------------------------------
def test_unique_emails_stay_valid_addresses_at_scale():
    schema = _one_table(
        [
            {"name": "id", "type": "id"},
            {"name": "full_name", "type": "name"},
            {"name": "email", "type": "email", "depends_on": "full_name", "unique": True},
        ],
        rows=200_000,
    )
    emails = [r["email"] for r in generate(schema)["t"]]
    assert len(set(emails)) == len(emails)
    malformed = [e for e in emails if not is_reserved_email(e)]
    assert malformed == []  # was 185,300 like 'x@inbox.example.com-1'
    # Deduplication happens inside the local part.
    assert any(re.match(r"^[a-z.]+\d+@", e) for e in emails)


def test_shipped_ecommerce_emails_have_no_suffix_after_tld():
    emails = [r["email"] for r in generate(os.path.join(ROOT, "schemas", "ecommerce.yaml"))["customers"]]
    assert all(is_reserved_email(e) for e in emails)
    assert not any(re.search(r"\.(com|org|net)-\d+$", e) for e in emails)


def test_unique_int_is_resampled_inside_range():
    schema = _one_table([{"name": "x", "type": "int", "min": 1, "max": 60, "unique": True}], rows=50)
    dataset = generate(schema)
    values = [r["x"] for r in dataset["t"]]
    assert {type(v) for v in values} == {int}
    assert len(set(values)) == 50
    assert all(1 <= v <= 60 for v in values)
    assert validate_dataset(dataset).ok


def test_unique_int_filling_the_whole_range():
    schema = _one_table([{"name": "x", "type": "int", "min": 1, "max": 60, "unique": True}], rows=60)
    values = sorted(r["x"] for r in generate(schema)["t"])
    assert values == list(range(1, 61))


def test_unique_rounded_float_stays_numeric():
    schema = _one_table(
        [{"name": "x", "type": "float", "min": 0, "max": 1, "round": 2, "unique": True}], rows=90
    )
    dataset = generate(schema)
    values = [r["x"] for r in dataset["t"]]
    assert all(isinstance(v, float) and 0 <= v <= 1 for v in values)
    assert len(set(values)) == 90
    assert validate_dataset(dataset).ok


def test_unique_date_and_small_pool_types():
    schema = _one_table(
        [
            {"name": "d", "type": "date", "start": "2024-01-01", "end": "2024-01-31", "unique": True},
            {"name": "c", "type": "country", "unique": True},
            {"name": "ip", "type": "ipv4", "unique": True},
        ],
        rows=19,
    )
    dataset = generate(schema)
    for col in ("d", "c", "ip"):
        values = [r[col] for r in dataset["t"]]
        assert len(set(values)) == 19, col
    assert validate_dataset(dataset).ok


@pytest.mark.parametrize(
    "field, rows, message",
    [
        ({"name": "x", "type": "int", "min": 1, "max": 5, "unique": True}, 20, "only take 5 distinct"),
        ({"name": "c", "type": "category", "categories": ["a", "b"], "unique": True}, 10, "2 categories"),
        ({"name": "c", "type": "country", "unique": True}, 500, "distinct values"),
        ({"name": "b", "type": "bool", "unique": True}, 3, "2 distinct"),
        (
            {"name": "d", "type": "date", "start": "2024-01-01", "end": "2024-01-05", "unique": True},
            6,
            "5 distinct",
        ),
    ],
)
def test_impossible_unique_is_a_parse_error(field, rows, message):
    with pytest.raises(SchemaError, match=message):
        Schema.from_dict(_one_table([field], rows=rows))


def test_unique_category_with_enough_labels_uses_each_once():
    schema = _one_table(
        [{"name": "c", "type": "category", "categories": ["a", "b", "c", "d"], "unique": True}], rows=3
    )
    dataset = generate(schema)
    values = [r["c"] for r in dataset["t"]]
    assert len(set(values)) == 3 and set(values) <= {"a", "b", "c", "d"}
    assert validate_dataset(dataset).ok


# --------------------------------------------------------------------------
# self-referencing foreign keys
# --------------------------------------------------------------------------
def test_self_referencing_fk_builds_a_hierarchy():
    schema = {
        "seed": 4,
        "tables": [
            {
                "name": "employees",
                "rows": 200,
                "fields": [
                    {"name": "emp_id", "type": "id"},
                    {"name": "manager_id", "type": "foreign_key", "references": "employees.emp_id"},
                ],
            }
        ],
    }
    dataset = generate(schema)  # used to raise "the parent table is empty"
    rows = dataset["employees"]
    index = {r["emp_id"]: i for i, r in enumerate(rows)}
    assert rows[0]["manager_id"] is None
    for i, r in enumerate(rows[1:], start=1):
        assert index[r["manager_id"]] < i  # parents always come first -> acyclic
    report = validate_dataset(dataset)
    assert report.ok, report.render()
    assert any(c.name == "hierarchy_acyclic" and c.passed for c in report.checks)


def test_self_reference_null_rate_adds_roots():
    schema = {
        "seed": 4,
        "tables": [
            {
                "name": "nodes",
                "rows": 500,
                "fields": [
                    {"name": "node_id", "type": "id"},
                    {"name": "parent_id", "type": "foreign_key", "references": "nodes.node_id", "null_rate": 0.2},
                ],
            }
        ],
    }
    roots = sum(1 for r in generate(schema)["nodes"] if r["parent_id"] is None)
    assert 60 < roots < 150


def test_validator_detects_cycle_in_hierarchy():
    schema = Schema.from_dict(
        {
            "tables": [
                {
                    "name": "e",
                    "rows": 3,
                    "fields": [
                        {"name": "id", "type": "id"},
                        {"name": "boss", "type": "foreign_key", "references": "e.id"},
                    ],
                }
            ]
        }
    )
    rows = [{"id": 1, "boss": None}, {"id": 2, "boss": 3}, {"id": 3, "boss": 2}]
    report = validate_dataset(Dataset(schema=schema, tables={"e": rows}))
    failed = {c.name for c in report.failures()}
    assert "hierarchy_acyclic" in failed


def test_unique_self_reference_is_rejected():
    with pytest.raises(SchemaError, match="cannot be unique"):
        Schema.from_dict(
            _one_table(
                [
                    {"name": "id", "type": "id"},
                    {"name": "p", "type": "foreign_key", "references": "t.id", "unique": True},
                ]
            )
        )


# --------------------------------------------------------------------------
# formulas: null safety and resource limits
# --------------------------------------------------------------------------
def test_formula_over_nullable_column_does_not_crash():
    schema = _one_table(
        [
            {"name": "qty", "type": "int", "min": 1, "max": 5, "null_rate": 0.3},
            {"name": "price", "type": "float", "min": 1, "max": 10, "round": 2},
            {"name": "total", "type": "formula", "expr": "round(qty * price, 2)"},
            {"name": "total_or_zero", "type": "formula", "expr": "round(coalesce(qty, 0) * price, 2)"},
        ],
        rows=300,
    )
    rows = generate(schema)["t"]  # used to raise TypeError: NoneType * float
    null_rows = [r for r in rows if r["qty"] is None]
    assert null_rows, "fixture should produce some nulls"
    assert all(r["total"] is None for r in null_rows)
    assert all(r["total_or_zero"] == 0 for r in null_rows)
    assert all(r["total"] == round(r["qty"] * r["price"], 2) for r in rows if r["qty"] is not None)


def test_formula_null_semantics():
    assert Formula("a + 1").eval({"a": None}) is None
    assert Formula("a > 1").eval({"a": None}) is None
    assert Formula("1 if a > 1 else 0").eval({"a": None}) is None
    assert Formula("coalesce(a, b, 7)").eval({"a": None, "b": None}) == 7
    assert Formula("is_null(a)").eval({"a": None}) is True
    assert Formula("a and b").eval({"a": False, "b": None}) is False
    assert Formula("a or b").eval({"a": True, "b": None}) is True
    assert Formula("a and b").eval({"a": True, "b": None}) is None


@pytest.mark.parametrize(
    "expr",
    [
        "9 ** 9 ** 9",
        '"x" * 10**9',
        '10**9 * "x"',
        "pow(10, 10**6)",
        "exp(1000)",
        "10.0 ** 400",
        "(-8) ** 0.5",
        "(2 ** 4000) * (2 ** 4000)",
    ],
)
def test_formula_resource_limits_raise_fast(expr):
    started = time.perf_counter()
    with pytest.raises(FormulaError):
        Formula(expr).eval({})
    assert time.perf_counter() - started < 1.0


def test_formula_rejects_huge_source():
    with pytest.raises(FormulaError, match="limit"):
        Formula("+".join(["a"] * 2000))


def test_formula_runtime_error_names_the_field():
    schema = _one_table(
        [
            {"name": "a", "type": "int", "min": 0, "max": 0},
            {"name": "ratio", "type": "formula", "expr": "10 / a"},
        ],
        rows=3,
    )
    with pytest.raises(SchemaError, match="t.ratio"):
        generate(schema)


# --------------------------------------------------------------------------
# parse-time errors that used to be accepted or fail late
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field, message",
    [
        ({"name": "d", "type": "date", "start": "2025-01-01", "end": "2020-01-01"}, "before start"),
        ({"name": "d", "type": "datetime", "start": "2025-02-30"}, "not an ISO date"),
        ({"name": "x", "type": "int", "correlate": {"field": "nope"}}, "unknown field 'nope'"),
        ({"name": "x", "type": "int", "distribution": "poisson"}, "unknown distribution"),
        ({"name": "x", "type": "int", "null_rate": 1.5}, "null_rate"),
        ({"name": "b", "type": "bool", "true_rate": 2}, "true_rate"),
        ({"name": "f", "type": "formula", "expr": "ghost * 2"}, "unknown field 'ghost'"),
        ({"name": "f", "type": "formula", "expr": "__import__('os')"}, "not allowed"),
        ({"name": "e", "type": "email", "depends_on": "nobody"}, "unknown field 'nobody'"),
        ({"name": "i", "type": "id", "strategy": "random"}, "strategy"),
    ],
)
def test_invalid_specs_fail_at_parse_time(field, message):
    with pytest.raises(SchemaError, match=message):
        Schema.from_dict(_one_table([field], rows=5))


def test_fk_to_empty_parent_is_a_parse_error():
    with pytest.raises(SchemaError, match="rows: 0"):
        Schema.from_dict(
            {
                "tables": [
                    {"name": "p", "rows": 0, "fields": [{"name": "id", "type": "id"}]},
                    {
                        "name": "c",
                        "rows": 5,
                        "fields": [{"name": "pid", "type": "foreign_key", "references": "p.id"}],
                    },
                ]
            }
        )


# --------------------------------------------------------------------------
# validator: strict email, date bounds, fictional phones, non-null keys
# --------------------------------------------------------------------------
def _email_dataset(values):
    schema = Schema.from_dict(_one_table([{"name": "email", "type": "email"}], rows=len(values)))
    return Dataset(schema=schema, tables={"t": [{"email": v} for v in values]})


def test_validator_rejects_malformed_or_real_domain_email():
    report = validate_dataset(
        _email_dataset(["ok@example.com", "daniela.reyes@inbox.example.com-1", "someone@gmail.com"])
    )
    type_check = next(c for c in report.checks if c.name == "type")
    assert not type_check.passed
    assert "2 value(s)" in type_check.detail


def test_validator_checks_date_bounds():
    schema = Schema.from_dict(
        _one_table([{"name": "d", "type": "date", "start": "2024-01-01", "end": "2024-12-31"}], rows=2)
    )
    ds = Dataset(schema=schema, tables={"t": [{"d": "2024-05-05"}, {"d": "2019-01-01"}]})
    failures = validate_dataset(ds).failures()
    assert [c.name for c in failures] == ["type"]


def test_validator_flags_null_ids_and_fks():
    schema = Schema.from_dict(
        {
            "tables": [
                {"name": "p", "rows": 2, "fields": [{"name": "id", "type": "id"}]},
                {
                    "name": "c",
                    "rows": 2,
                    "fields": [{"name": "pid", "type": "foreign_key", "references": "p.id"}],
                },
            ]
        }
    )
    ds = Dataset(
        schema=schema,
        tables={"p": [{"id": 1}, {"id": None}], "c": [{"pid": 1}, {"pid": None}]},
    )
    failed = Counter((c.name, c.table) for c in validate_dataset(ds).failures())
    assert failed[("not_null", "p")] == 1
    assert failed[("not_null", "c")] == 1


def test_phones_are_in_the_fictional_nanp_block():
    rows = generate(_one_table([{"name": "phone", "type": "phone"}], rows=500))["t"]
    assert all(re.fullmatch(r"\+1-\d{3}-555-01\d{2}", r["phone"]) for r in rows)
