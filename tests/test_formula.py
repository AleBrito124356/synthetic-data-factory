"""The safe formula evaluator: correctness and refusal of unsafe input."""
import pytest

from factory.formula import Formula, FormulaError


def test_arithmetic():
    f = Formula("round(quantity * unit_price, 2)")
    assert f.eval({"quantity": 3, "unit_price": 4.005}) == 12.02 or f.eval({"quantity": 3, "unit_price": 4.0}) == 12.0


def test_referenced_fields_detected():
    f = Formula("price * (1 + tax_rate)")
    assert set(f.referenced_fields) == {"price", "tax_rate"}


def test_conditional_expression():
    f = Formula("100 if score >= 50 else 0")
    assert f.eval({"score": 70}) == 100
    assert f.eval({"score": 10}) == 0


def test_builtin_functions():
    assert Formula("max(a, b)").eval({"a": 2, "b": 9}) == 9
    assert Formula("min(a, b)").eval({"a": 2, "b": 9}) == 2
    assert Formula("abs(x)").eval({"x": -5}) == 5


def test_unknown_field_raises():
    with pytest.raises(FormulaError):
        Formula("a + b").eval({"a": 1})


def test_attribute_access_is_rejected():
    with pytest.raises(FormulaError):
        Formula("a.__class__")


def test_disallowed_function_is_rejected():
    with pytest.raises(FormulaError):
        Formula("__import__('os')")


def test_subscript_is_rejected():
    with pytest.raises(FormulaError):
        Formula("a[0]")


def test_date_helpers():
    row = {"a": "2024-01-31", "b": "2024-03-01 18:30:00"}
    assert Formula("year(a)").eval(row) == 2024
    assert Formula("month(b)").eval(row) == 3
    assert Formula("day(a)").eval(row) == 31
    assert Formula("weekday(a)").eval(row) == 2  # a Wednesday
    assert Formula("days_between(a, b)").eval(row) == 30
    assert Formula("hours_between(a, b)").eval(row) == 30 * 24 + 18.5
    assert Formula("year(x)").eval({"x": None}) is None


def test_date_helper_rejects_non_dates():
    with pytest.raises(FormulaError):
        Formula("year(x)").eval({"x": "not a date"})
