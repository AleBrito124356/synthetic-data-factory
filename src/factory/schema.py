"""Schema model and parser for the tabular generator.

A schema is a plain dict (usually loaded from YAML) describing one or more
related tables. This module turns that dict into validated dataclasses and
raises clear, actionable errors when the schema is malformed — so a typo in a
weight or a dangling foreign key fails loudly at parse time, not with a
confusing traceback deep inside generation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

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
        if not isinstance(seed, int):
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
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    # ---- validation ------------------------------------------------------
    def validate(self) -> None:
        """Cross-table checks: foreign keys resolve, weights are sane."""
        for table in self.tables:
            for fs in table.fields:
                if fs.type == "category":
                    _validate_categories(table.name, fs)
                elif fs.type == "foreign_key":
                    _validate_foreign_key(self, table.name, fs)
                elif fs.type in ("int", "float"):
                    _validate_numeric_range(table.name, fs)

        # Foreign keys must not form a cycle across tables (a table cannot be
        # generated before the parent it references).
        _check_table_cycle(self)


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
    if not isinstance(rows, int) or rows < 0:
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
        if not isinstance(weight, (int, float)) or weight < 0:
            raise SchemaError(
                f"'{table}.{fs.name}': weight for '{value}' must be a non-negative number."
            )
    if sum(cats.values()) <= 0:
        raise SchemaError(f"'{table}.{fs.name}': category weights sum to zero.")


def _validate_numeric_range(table: str, fs: FieldSpec) -> None:
    lo = fs.params.get("min")
    hi = fs.params.get("max")
    if lo is not None and hi is not None and lo > hi:
        raise SchemaError(f"'{table}.{fs.name}': min ({lo}) is greater than max ({hi}).")
    corr = fs.params.get("correlate")
    if corr is not None:
        if not isinstance(corr, dict) or "field" not in corr:
            raise SchemaError(
                f"'{table}.{fs.name}': 'correlate' must be a mapping with a 'field' key."
            )
        strength = corr.get("strength", 0.6)
        if not isinstance(strength, (int, float)) or not (0.0 <= strength <= 1.0):
            raise SchemaError(
                f"'{table}.{fs.name}': correlate.strength must be between 0 and 1."
            )


def _validate_foreign_key(schema: Schema, table: str, fs: FieldSpec) -> None:
    ref = fs.params.get("references")
    if not ref or not isinstance(ref, str) or "." not in ref:
        raise SchemaError(
            f"Foreign key '{table}.{fs.name}' needs 'references' as 'table.column'."
        )
    ref_table_name, ref_col = ref.split(".", 1)
    ref_table = schema.get_table(ref_table_name)
    if ref_table is None:
        raise SchemaError(
            f"Foreign key '{table}.{fs.name}' references unknown table '{ref_table_name}'."
        )
    if ref_table.get_field(ref_col) is None:
        raise SchemaError(
            f"Foreign key '{table}.{fs.name}' references unknown column "
            f"'{ref_table_name}.{ref_col}'."
        )


def _check_table_cycle(schema: Schema) -> None:
    """Detect cyclic foreign-key dependencies between tables via DFS."""
    deps: Dict[str, set] = {t.name: set() for t in schema.tables}
    for table in schema.tables:
        for fs in table.fields:
            if fs.type == "foreign_key":
                ref_table = fs.params["references"].split(".", 1)[0]
                if ref_table != table.name:  # self-reference is allowed
                    deps[table.name].add(ref_table)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in deps}

    def visit(node: str, stack: List[str]) -> None:
        color[node] = GRAY
        for nxt in deps[node]:
            if color[nxt] == GRAY:
                cycle = " -> ".join(stack + [nxt])
                raise SchemaError(f"Foreign-key cycle between tables: {cycle}")
            if color[nxt] == WHITE:
                visit(nxt, stack + [nxt])
        color[node] = BLACK

    for name in deps:
        if color[name] == WHITE:
            visit(name, [name])


def load_schema(path_or_dict: Any) -> Schema:
    """Convenience loader accepting a path string or an already-loaded dict."""
    if isinstance(path_or_dict, Schema):
        return path_or_dict
    if isinstance(path_or_dict, dict):
        return Schema.from_dict(path_or_dict)
    return Schema.from_yaml(str(path_or_dict))
