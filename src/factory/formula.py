"""A small, safe arithmetic expression evaluator for ``formula`` fields.

Formula fields let a schema compute one column from others in the same row,
e.g. ``total = round(quantity * unit_price * (1 + tax_rate), 2)``.

We deliberately do NOT use ``eval``. Instead we parse the expression to an AST
once and walk a whitelist of node types. There is no attribute access, no
subscripting, no comprehensions, no imports, and only a fixed set of math
functions — so a schema author (or a schema pulled from elsewhere) cannot make
the generator read files, spawn processes, or leak globals.

Two more guarantees make the evaluator safe to run on untrusted schemas:

* **Resource limits.** Exponentiation, integer growth, and string repetition
  are bounded *before* the work is done, so ``9 ** 9 ** 9`` or
  ``"x" * 10**9`` raise :class:`FormulaError` immediately instead of hanging
  the process or allocating gigabytes.
* **Null safety.** A ``None`` operand (a column with ``null_rate``) makes the
  result ``None`` — SQL-style — instead of crashing with a ``TypeError``.
  ``coalesce(a, b, ...)`` returns the first non-null argument, so
  ``coalesce(qty, 0) * price`` supplies a default.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict, List

# ---- resource limits ------------------------------------------------------
MAX_SOURCE_LENGTH = 2000      # characters in one expression
MAX_NODES = 400               # AST nodes in one expression
MAX_INT_BITS = 4096           # largest integer a formula may produce (~1233 digits)
MAX_STRING_LENGTH = 10_000    # largest string a formula may produce


class FormulaError(ValueError):
    """Raised for an invalid or disallowed formula expression."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _guard_result(value: Any) -> Any:
    if isinstance(value, complex):
        raise FormulaError("Formula produced a complex number (e.g. a fractional power of a negative).")
    if _is_int(value) and value.bit_length() > MAX_INT_BITS:
        raise FormulaError(f"Formula result exceeds the {MAX_INT_BITS}-bit integer limit.")
    if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
        raise FormulaError(f"Formula result exceeds the {MAX_STRING_LENGTH}-character string limit.")
    return value


def _safe_pow(base: Any, exponent: Any, mod: Any = None) -> Any:
    if mod is not None:
        return pow(base, exponent, mod)  # modular pow stays small and fast
    if _is_int(base) and _is_int(exponent) and abs(base) > 1 and exponent > 0:
        # Size of the exact integer result, estimated BEFORE computing it.
        if exponent * math.log2(abs(base)) > MAX_INT_BITS:
            raise FormulaError(
                f"Exponentiation {base} ** {exponent} would exceed the "
                f"{MAX_INT_BITS}-bit integer limit."
            )
    try:
        return pow(base, exponent)
    except OverflowError as exc:
        raise FormulaError(f"Exponentiation overflowed: {exc}") from exc


def _safe_mul(left: Any, right: Any) -> Any:
    # Sequence repetition: "x" * n or n * "x".
    for seq, count in ((left, right), (right, left)):
        if isinstance(seq, str) and _is_int(count) and len(seq) * max(count, 0) > MAX_STRING_LENGTH:
            raise FormulaError(
                f"String repetition would exceed the {MAX_STRING_LENGTH}-character limit."
            )
    if _is_int(left) and _is_int(right) and left.bit_length() + right.bit_length() > MAX_INT_BITS + 1:
        raise FormulaError(f"Multiplication would exceed the {MAX_INT_BITS}-bit integer limit.")
    return left * right


def _safe_add(left: Any, right: Any) -> Any:
    if isinstance(left, str) and isinstance(right, str) and len(left) + len(right) > MAX_STRING_LENGTH:
        raise FormulaError(
            f"String concatenation would exceed the {MAX_STRING_LENGTH}-character limit."
        )
    return left + right


_BINOPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: _safe_add,
    ast.Sub: operator.sub,
    ast.Mult: _safe_mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
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


def _coalesce(*args: Any) -> Any:
    for a in args:
        if a is not None:
            return a
    return None


def _is_null(value: Any) -> bool:
    return value is None


def _safe_int(value: Any) -> int:
    if isinstance(value, str) and len(value) > 1000:
        raise FormulaError("int() of a string longer than 1000 characters is not allowed.")
    return int(value)


_FUNCS: Dict[str, Callable[..., Any]] = {
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
    "int": _safe_int,
    "float": float,
    "str": str,
    "len": len,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "exp": math.exp,
    "pow": _safe_pow,
    "coalesce": _coalesce,
    "is_null": _is_null,
}

# Functions that receive None arguments as-is. Every other function returns
# None when any argument is None (null propagation).
_NULL_AWARE = {"coalesce", "is_null"}

FUNCTION_NAMES = tuple(sorted(_FUNCS))


class Formula:
    """A compiled formula. Parse once, evaluate per row."""

    __slots__ = ("source", "_tree", "_names")

    def __init__(self, source: str):
        self.source = source
        if len(source) > MAX_SOURCE_LENGTH:
            raise FormulaError(
                f"Formula is {len(source)} characters long; the limit is {MAX_SOURCE_LENGTH}."
            )
        try:
            self._tree = ast.parse(source, mode="eval")
        except (SyntaxError, RecursionError, MemoryError) as exc:
            raise FormulaError(f"Cannot parse formula {source!r}: {exc}") from exc
        node_count = sum(1 for _ in ast.walk(self._tree))
        if node_count > MAX_NODES:
            raise FormulaError(f"Formula has {node_count} elements; the limit is {MAX_NODES}.")
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
        try:
            return _eval_node(self._tree.body, row)
        except FormulaError:
            raise
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise FormulaError(
                f"Formula {self.source!r} failed: {type(exc).__name__}: {exc}"
            ) from exc


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
            if isinstance(arg, ast.Starred):
                raise FormulaError("Star-arguments are not allowed in formulas.")
            _validate(arg)
    elif isinstance(node, ast.Name):
        return
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool, str)):
            raise FormulaError(f"Constant of type {type(node.value).__name__} is not allowed.")
        _guard_result(node.value)
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
        left = _eval_node(node.left, row)
        right = _eval_node(node.right, row)
        if left is None or right is None:
            return None
        return _guard_result(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, row)
        if operand is None:
            return None
        return _UNARYOPS[type(node.op)](operand)
    if isinstance(node, ast.BoolOp):
        # Kleene three-valued logic: a definite False (and) / True (or) wins;
        # otherwise any unknown (None) operand makes the result unknown.
        values = [_eval_node(v, row) for v in node.values]
        is_and = isinstance(node.op, ast.And)
        for v in values:
            if v is not None and ((not v) if is_and else v):
                return v
        if any(v is None for v in values):
            return None
        return values[-1]
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, row)
        for op, comp_node in zip(node.ops, node.comparators):
            right = _eval_node(comp_node, row)
            if left is None or right is None:
                return None
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        test = _eval_node(node.test, row)
        if test is None:
            return None
        return _eval_node(node.body, row) if test else _eval_node(node.orelse, row)
    if isinstance(node, ast.Call):
        name = node.func.id
        args = [_eval_node(a, row) for a in node.args]
        if name not in _NULL_AWARE and any(a is None for a in args):
            return None
        return _guard_result(_FUNCS[name](*args))
    raise FormulaError(f"Unsupported expression element: {type(node).__name__}")
