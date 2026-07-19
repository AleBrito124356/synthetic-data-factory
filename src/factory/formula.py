"""A small, safe arithmetic expression evaluator for ``formula`` fields.

Formula fields let a schema compute one column from others in the same row,
e.g. ``total = round(quantity * unit_price * (1 + tax_rate), 2)``.

We deliberately do NOT use ``eval``. Instead we parse the expression to an AST
once and walk a whitelist of node types. There is no attribute access, no
subscripting, no comprehensions, no imports, and only a fixed set of math
functions — so a schema author (or a schema pulled from elsewhere) cannot make
the generator read files, spawn processes, or leak globals.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, List

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_FUNCS = {
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
    "int": int,
    "float": float,
    "len": len,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "exp": math.exp,
    "pow": pow,
}


class FormulaError(ValueError):
    """Raised for an invalid or disallowed formula expression."""


class Formula:
    """A compiled formula. Parse once, evaluate per row."""

    __slots__ = ("source", "_tree", "_names")

    def __init__(self, source: str):
        self.source = source
        try:
            self._tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - message passthrough
            raise FormulaError(f"Cannot parse formula {source!r}: {exc}") from exc
        self._names = _collect_names(self._tree)
        # Fail fast on disallowed constructs by validating the tree up front.
        _validate(self._tree.body)

    @property
    def referenced_fields(self) -> List[str]:
        """Names used in the expression that are not known functions.

        These are the sibling columns the formula depends on, which the
        generator uses to order column generation.
        """
        return [n for n in self._names if n not in _FUNCS]

    def eval(self, row: Dict[str, Any]) -> Any:
        return _eval_node(self._tree.body, row)


# --------------------------------------------------------------------------
def _collect_names(tree: ast.AST) -> List[str]:
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in names:
                names.append(node.id)
    return names


def _validate(node: ast.AST) -> None:
    """Reject anything outside the supported grammar before evaluation."""
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise FormulaError(f"Operator {type(node.op).__name__} is not allowed.")
        _validate(node.left)
        _validate(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise FormulaError(f"Unary operator {type(node.op).__name__} is not allowed.")
        _validate(node.operand)
    elif isinstance(node, ast.BoolOp):
        for v in node.values:
            _validate(v)
    elif isinstance(node, ast.Compare):
        _validate(node.left)
        for op, comp in zip(node.ops, node.comparators):
            if type(op) not in _CMPOPS:
                raise FormulaError(f"Comparison {type(op).__name__} is not allowed.")
            _validate(comp)
    elif isinstance(node, ast.IfExp):
        _validate(node.test)
        _validate(node.body)
        _validate(node.orelse)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            fname = getattr(node.func, "id", type(node.func).__name__)
            raise FormulaError(f"Function {fname!r} is not allowed.")
        if node.keywords:
            raise FormulaError("Keyword arguments are not allowed in formulas.")
        for arg in node.args:
            _validate(arg)
    elif isinstance(node, ast.Name):
        return
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool, str)):
            raise FormulaError(f"Constant of type {type(node.value).__name__} is not allowed.")
    else:
        raise FormulaError(f"Expression element {type(node).__name__} is not allowed.")


def _eval_node(node: ast.AST, row: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in row:
            return row[node.id]
        raise FormulaError(f"Formula references unknown field {node.id!r}.")
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval_node(node.left, row), _eval_node(node.right, row))
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, row))
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, row) for v in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for v in values:
                result = result and v
            return result
        result = False
        for v in values:
            result = result or v
        return result
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, row)
        for op, comp_node in zip(node.ops, node.comparators):
            right = _eval_node(comp_node, row)
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _eval_node(node.body, row) if _eval_node(node.test, row) else _eval_node(node.orelse, row)
    if isinstance(node, ast.Call):
        func = _FUNCS[node.func.id]
        args = [_eval_node(a, row) for a in node.args]
        return func(*args)
    raise FormulaError(f"Unsupported expression element: {type(node).__name__}")
