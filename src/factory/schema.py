"""Schema model and parser for the tabular generator.

A schema is a plain dict (usually loaded from YAML) describing one or more
related tables. This module turns that dict into validated dataclasses and
raises clear, actionable errors when the schema is malformed — so a typo in a
weight, a dangling foreign key, an impossible ``unique`` request, or an
inverted date range fails loudly at parse time, not with a confusing
traceback (or silently wrong data) deep inside generation.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .formula import Formula, FormulaError
from .providers import domain_size

# Field types the generator understands. Aliases map to a canonical name.
_TYPE_ALIASES = {
    "id": "id",
    "uuid": "uuid",
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
    "bool": "bool",
    "boolean": "bool",
    "category": "category",
    "categorical": "category",
    "enum": "category",
    "name": "name",
    "first_name": "first_name",
    "last_name": "last_name",
    "email": "email",
    "phone": "phone",
    "date": "date",
    "datetime": "datetime",
    "text": "text",
    "sentence": "text",
    "city": "city",
    "country": "country",
    "address": "address",
    "company": "company",
    "job": "job",
    "url": "url",
    "ipv4": "ipv4",
    "foreign_key": "foreign_key",
    "fk": "foreign_key",
    "formula": "formula",
    "computed": "formula",
}

CANONICAL_TYPES = sorted(set(_TYPE_ALIASES.values()))

NUMERIC_DISTRIBUTIONS = ("uniform", "normal", "exponential", "exp")

DEFAULT_DATE_START = "2020-01-01"
DEFAULT_DATE_END = "2025-12-31"


class SchemaError(ValueError):
    """Raised when a schema is structurally invalid."""


@dataclass
class FieldSpec:
    """One column definition. Unknown keys are preserved in ``params`` so
    type-specific options (min, max, categories, references, expr, ...) stay
    accessible without a giant fixed attribute list."""

    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience accessors ------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @property
    def unique(self) -> bool:
        return bool(self.params.get("unique", False))

    @property
    def null_rate(self) -> float:
        return float(self.params.get("null_rate", 0.0) or 0.0)

    @property
    def depends_on(self) -> Optional[str]:
        """Explicit intra-row dependency (e.g. email derived from a name field)."""
        return self.params.get("depends_on")

    @property
    def references(self) -> Optional[Tuple[str, str]]:
        """``(table, column)`` for a foreign key, else None."""
        ref = self.params.get("references")
        if self.type != "foreign_key" or not isinstance(ref, str) or "." not in ref:
            return None
        table, col = ref.split(".", 1)
        return table, col

    @property
    def is_sequential_int_id(self) -> bool:
        """True for an ``id`` that yields plain integers (no prefix, no uuid)."""
        return (
            self.type == "id"
            and self.params.get("strategy", "sequential") == "sequential"
            and not self.params.get("prefix")
        )


@dataclass
class TableSpec:
    name: str
    rows: int
    fields: List[FieldSpec]

    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def get_field(self, name: str) -> Optional[FieldSpec]:
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass
class Schema:
    tables: List[TableSpec]
    seed: int = 1234
    meta: Dict[str, Any] = field(default_factory=dict)

    def get_table(self, name: str) -> Optional[TableSpec]:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    # ---- parsing ---------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        if not isinstance(data, dict):
            raise SchemaError("Top-level schema must be a mapping.")

        raw_tables = data.get("tables")
        if not raw_tables or not isinstance(raw_tables, list):
            raise SchemaError("Schema must contain a non-empty 'tables' list.")

        seed = data.get("seed", 1234)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SchemaError(f"'seed' must be an integer, got {type(seed).__name__}.")

        tables: List[TableSpec] = []
        seen_tables: set = set()
        for i, rt in enumerate(raw_tables):
            table = _parse_table(rt, index=i)
            if table.name in seen_tables:
                raise SchemaError(f"Duplicate table name: '{table.name}'.")
            seen_tables.add(table.name)
            tables.append(table)

        meta = {k: v for k, v in data.items() if k not in {"tables", "seed"}}
        schema = cls(tables=tables, seed=seed, meta=meta)
        schema.validate()
        return schema

    @classmethod
    def from_yaml(cls, path: str) -> "Schema":
        if not os.path.exists(path):
            raise SchemaError(f"Schema file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise SchemaError(f"Schema file {path} is not valid YAML: {exc}") from exc
        return cls.from_dict(data)

    # ---- validation ------------------------------------------------------
    def validate(self) -> None:
        """Cross-field and cross-table checks. Raises :class:`SchemaError`."""
        for table in self.tables:
            for fs in table.fields:
                _validate_common(table, fs)
                if fs.type == "category":
                    _validate_categories(table.name, fs)
                elif fs.type == "foreign_key":
                    _validate_foreign_key(self, table, fs)
                elif fs.type in ("int", "float"):
                    _validate_numeric(table, fs)
                elif fs.type in ("date", "datetime"):
                    _validate_temporal(table.name, fs)
                elif fs.type == "formula":
                    _validate_formula(table, fs)
                elif fs.type == "id":
                    _validate_id(table.name, fs)
                _validate_unique_capacity(table, fs)

        # Foreign keys must not form a cycle across tables (a table cannot be
        # generated before the parent it references).
        _check_table_cycle(self)


# --------------------------------------------------------------------------
# public helpers
# --------------------------------------------------------------------------
def parse_date(value: Any, where: str = "") -> _dt.date:
    """Parse an ISO date (YAML may already have produced a ``date``)."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise SchemaError(
            f"{where}: {value!r} is not an ISO date (expected YYYY-MM-DD)."
        ) from exc


def field_capacity(table: TableSpec, fs: FieldSpec) -> Optional[int]:
    """How many distinct non-null values ``fs`` can take, or None if the
    space is effectively unbounded. Used for the ``unique`` pigeonhole check."""
    t = fs.type
    if t in ("int", "float"):
        is_int = t == "int"
        lo = fs.get("min", 0)
        hi = fs.get("max", 100 if is_int else 1.0)
        if is_int:
            return max(0, int(round(hi)) - int(round(lo)) + 1)
        nd = fs.get("round")
        if nd is None:
            return None
        scale = 10 ** int(nd)
        return max(0, math.floor(hi * scale + 1e-9) - math.ceil(lo * scale - 1e-9) + 1)
    if t == "category":
        cats = fs.get("categories") or {}
        return len(cats)
    if t == "date":
        start = parse_date(fs.get("start", DEFAULT_DATE_START))
        end = parse_date(fs.get("end", DEFAULT_DATE_END))
        return max(0, (end - start).days + 1)
    if t == "datetime":
        start = parse_date(fs.get("start", DEFAULT_DATE_START))
        end = parse_date(fs.get("end", DEFAULT_DATE_END))
        return max(0, ((end - start).days + 1) * 86400)
    return domain_size(t)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _parse_table(rt: Any, index: int) -> TableSpec:
    if not isinstance(rt, dict):
        raise SchemaError(f"tables[{index}] must be a mapping.")
    name = rt.get("name")
    if not name or not isinstance(name, str):
        raise SchemaError(f"tables[{index}] is missing a string 'name'.")

    rows = rt.get("rows", rt.get("count"))
    if rows is None:
        raise SchemaError(f"Table '{name}' must specify 'rows'.")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise SchemaError(f"Table '{name}': 'rows' must be a non-negative integer.")

    raw_fields = rt.get("fields")
    if not raw_fields or not isinstance(raw_fields, list):
        raise SchemaError(f"Table '{name}' must contain a non-empty 'fields' list.")

    fields_: List[FieldSpec] = []
    seen: set = set()
    for j, rf in enumerate(raw_fields):
        fs = _parse_field(name, rf, j)
        if fs.name in seen:
            raise SchemaError(f"Table '{name}' has a duplicate field: '{fs.name}'.")
        seen.add(fs.name)
        fields_.append(fs)

    return TableSpec(name=name, rows=rows, fields=fields_)


def _parse_field(table_name: str, rf: Any, index: int) -> FieldSpec:
    if not isinstance(rf, dict):
        raise SchemaError(f"Table '{table_name}': fields[{index}] must be a mapping.")
    fname = rf.get("name")
    if not fname or not isinstance(fname, str):
        raise SchemaError(f"Table '{table_name}': fields[{index}] needs a string 'name'.")

    ftype_raw = rf.get("type")
    if not ftype_raw or not isinstance(ftype_raw, str):
        raise SchemaError(f"Table '{table_name}.{fname}' needs a string 'type'.")
    ftype = _TYPE_ALIASES.get(ftype_raw.lower())
    if ftype is None:
        raise SchemaError(
            f"Table '{table_name}.{fname}': unknown type '{ftype_raw}'. "
            f"Valid types: {', '.join(CANONICAL_TYPES)}."
        )

    params = {k: v for k, v in rf.items() if k not in {"name", "type"}}
    return FieldSpec(name=fname, type=ftype, params=params)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_common(table: TableSpec, fs: FieldSpec) -> None:
    where = f"'{table.name}.{fs.name}'"
    nr = fs.params.get("null_rate")
    if nr is not None and (not _is_number(nr) or not (0.0 <= nr <= 1.0)):
        raise SchemaError(f"{where}: null_rate must be a number between 0 and 1.")
    dep = fs.depends_on
    if dep is not None:
        if dep == fs.name:
            raise SchemaError(f"{where}: depends_on cannot point at the field itself.")
        if table.get_field(dep) is None:
            raise SchemaError(
                f"{where}: depends_on refers to unknown field '{dep}' "
                f"(fields: {', '.join(table.field_names())})."
            )
    if fs.type == "bool":
        tr = fs.params.get("true_rate", 0.5)
        if not _is_number(tr) or not (0.0 <= tr <= 1.0):
            raise SchemaError(f"{where}: true_rate must be a number between 0 and 1.")


def _validate_categories(table: str, fs: FieldSpec) -> None:
    cats = fs.params.get("categories")
    if cats is None:
        raise SchemaError(f"Category field '{table}.{fs.name}' needs 'categories'.")
    if isinstance(cats, list):
        # Unweighted list — assign equal weight. Normalize into a dict here so
        # the generator only ever deals with weighted dicts.
        if not cats:
            raise SchemaError(f"'{table}.{fs.name}': 'categories' list is empty.")
        fs.params["categories"] = {str(c): 1.0 for c in cats}
        return
    if not isinstance(cats, dict) or not cats:
        raise SchemaError(
            f"'{table}.{fs.name}': 'categories' must be a non-empty list or "
            f"a mapping of value -> weight."
        )
    for value, weight in cats.items():
        if not _is_number(weight) or weight < 0:
            raise SchemaError(
                f"'{table}.{fs.name}': weight for '{value}' must be a non-negative number."
            )
    if sum(cats.values()) <= 0:
        raise SchemaError(f"'{table}.{fs.name}': category weights sum to zero.")


def _validate_numeric(table: TableSpec, fs: FieldSpec) -> None:
    where = f"'{table.name}.{fs.name}'"
    lo = fs.params.get("min")
    hi = fs.params.get("max")
    for key, val in (("min", lo), ("max", hi)):
        if val is not None and not _is_number(val):
            raise SchemaError(f"{where}: '{key}' must be a number, got {val!r}.")
    if lo is not None and hi is not None and lo > hi:
        raise SchemaError(f"{where}: min ({lo}) is greater than max ({hi}).")
    dist = fs.params.get("distribution", "uniform")
    if dist not in NUMERIC_DISTRIBUTIONS:
        raise SchemaError(
            f"{where}: unknown distribution {dist!r}. "
            f"Valid: {', '.join(NUMERIC_DISTRIBUTIONS)}."
        )
    corr = fs.params.get("correlate")
    if corr is not None:
        if not isinstance(corr, dict) or "field" not in corr:
            raise SchemaError(f"{where}: 'correlate' must be a mapping with a 'field' key.")
        strength = corr.get("strength", 0.6)
        if not _is_number(strength) or not (0.0 <= strength <= 1.0):
            raise SchemaError(f"{where}: correlate.strength must be between 0 and 1.")
        driver = corr["field"]
        if driver == fs.name:
            raise SchemaError(f"{where}: a field cannot correlate with itself.")
        if table.get_field(driver) is None:
            raise SchemaError(
                f"{where}: correlate.field refers to unknown field '{driver}' "
                f"(fields: {', '.join(table.field_names())})."
            )
        direction = str(corr.get("direction", "positive")).lower()
        if direction not in ("positive", "negative"):
            raise SchemaError(f"{where}: correlate.direction must be 'positive' or 'negative'.")


def _validate_temporal(table: str, fs: FieldSpec) -> None:
    where = f"'{table}.{fs.name}'"
    start = parse_date(fs.params.get("start", DEFAULT_DATE_START), f"{where} start")
    end = parse_date(fs.params.get("end", DEFAULT_DATE_END), f"{where} end")
    if end < start:
        raise SchemaError(f"{where}: end ({end}) is before start ({start}).")


def _validate_formula(table: TableSpec, fs: FieldSpec) -> None:
    where = f"'{table.name}.{fs.name}'"
    expr = fs.params.get("expr") or fs.params.get("formula")
    if not expr:
        raise SchemaError(f"Formula field {where} needs an 'expr'.")
    try:
        formula = Formula(str(expr))
    except FormulaError as exc:
        raise SchemaError(f"{where}: {exc}") from exc
    for ref in formula.referenced_fields:
        if ref == fs.name:
            raise SchemaError(f"{where}: a formula cannot reference its own field.")
        if table.get_field(ref) is None:
            raise SchemaError(
                f"{where}: formula references unknown field '{ref}' "
                f"(fields: {', '.join(table.field_names())})."
            )


def _validate_id(table: str, fs: FieldSpec) -> None:
    strategy = fs.params.get("strategy", "sequential")
    if strategy not in ("sequential", "uuid"):
        raise SchemaError(
            f"'{table}.{fs.name}': id strategy must be 'sequential' or 'uuid', got {strategy!r}."
        )
    start = fs.params.get("start", 1)
    if not isinstance(start, int) or isinstance(start, bool):
        raise SchemaError(f"'{table}.{fs.name}': id 'start' must be an integer.")


def _validate_foreign_key(schema: Schema, table: TableSpec, fs: FieldSpec) -> None:
    ref = fs.params.get("references")
    where = f"Foreign key '{table.name}.{fs.name}'"
    if not ref or not isinstance(ref, str) or "." not in ref:
        raise SchemaError(f"{where} needs 'references' as 'table.column'.")
    ref_table_name, ref_col = ref.split(".", 1)
    ref_table = schema.get_table(ref_table_name)
    if ref_table is None:
        raise SchemaError(f"{where} references unknown table '{ref_table_name}'.")
    if ref_table.get_field(ref_col) is None:
        raise SchemaError(
            f"{where} references unknown column '{ref_table_name}.{ref_col}'."
        )
    self_ref = ref_table_name == table.name
    if self_ref:
        if ref_col == fs.name:
            raise SchemaError(f"{where} cannot reference itself.")
        if fs.unique:
            raise SchemaError(
                f"{where}: a self-referencing foreign key builds a hierarchy "
                f"(many children per parent) and cannot be unique."
            )
        return
    if table.rows > 0 and ref_table.rows == 0:
        raise SchemaError(
            f"{where} references '{ref}' but table '{ref_table_name}' has rows: 0. "
            f"Give the parent table rows > 0."
        )
    if fs.unique and table.rows > ref_table.rows:
        raise SchemaError(
            f"Unique {where[0].lower() + where[1:]} needs at least {table.rows} parent rows "
            f"in '{ref_table_name}', which only has {ref_table.rows}."
        )


def _validate_unique_capacity(table: TableSpec, fs: FieldSpec) -> None:
    """Pigeonhole check: ``unique: true`` must be satisfiable."""
    if not fs.unique or fs.type in ("id", "uuid", "foreign_key", "formula"):
        return
    capacity = field_capacity(table, fs)
    if capacity is not None and table.rows > capacity:
        kind = "categories" if fs.type == "category" else "distinct values"
        raise SchemaError(
            f"'{table.name}.{fs.name}' is unique but can only take {capacity} {kind}, "
            f"and the table has {table.rows} rows. Widen the range or drop 'unique'."
        )


def _check_table_cycle(schema: Schema) -> None:
    """Detect cyclic foreign-key dependencies between tables via DFS."""
    deps: Dict[str, set] = {t.name: set() for t in schema.tables}
    for table in schema.tables:
        for fs in table.fields:
            if fs.type == "foreign_key":
                ref_table = fs.params["references"].split(".", 1)[0]
                if ref_table != table.name:  # self-reference builds a hierarchy
                    deps[table.name].add(ref_table)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in deps}

    def visit(node: str, stack: List[str]) -> None:
        color[node] = GRAY
        for nxt in sorted(deps[node]):
            if color[nxt] == GRAY:
                cycle = " -> ".join(stack + [nxt])
                raise SchemaError(f"Foreign-key cycle between tables: {cycle}")
            if color[nxt] == WHITE:
                visit(nxt, stack + [nxt])
        color[node] = BLACK

    for name in deps:
        if color[name] == WHITE:
            visit(name, [name])


def topological_table_order(schema: Schema) -> List[str]:
    """Table names ordered so every FK parent precedes its children."""
    names = [t.name for t in schema.tables]
    deps: Dict[str, set] = {name: set() for name in names}
    for table in schema.tables:
        for fs in table.fields:
            ref = fs.references
            if ref and ref[0] != table.name and ref[0] in deps:
                deps[table.name].add(ref[0])

    ordered: List[str] = []
    visited: set = set()

    def visit(node: str, stack: List[str]) -> None:
        if node in visited:
            return
        for parent in sorted(deps[node]):
            if parent in stack:
                raise SchemaError(f"Foreign-key cycle: {' -> '.join(stack + [parent])}")
            visit(parent, stack + [node])
        visited.add(node)
        ordered.append(node)

    for name in names:
        visit(name, [])
    return ordered


def load_schema(path_or_dict: Any) -> Schema:
    """Convenience loader accepting a path string or an already-loaded dict."""
    if isinstance(path_or_dict, Schema):
        return path_or_dict
    if isinstance(path_or_dict, dict):
        return Schema.from_dict(path_or_dict)
    return Schema.from_yaml(str(path_or_dict))
