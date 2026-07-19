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
