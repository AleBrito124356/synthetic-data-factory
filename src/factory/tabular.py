"""Schema-driven tabular data generation — the zero-dollar core of the factory.

Given a :class:`~factory.schema.Schema`, produce related tables with:

* realistic per-type values (names, emails, dates, categoricals, ...),
* referential integrity across foreign keys,
* weighted categoricals that closely match requested proportions,
* simple rank-based correlations between numeric columns,
* formula/computed columns evaluated from sibling columns,
* full determinism for a given seed.

No LLM and no network are involved on this path.

Generation strategy
--------------------
Tables are ordered by their foreign-key dependencies (topological sort) so a
parent table always exists before a child references it. Within a table,
columns are generated in dependency order too — a formula column that reads
``unit_price`` is generated after ``unit_price``; a correlated column after its
driver; a derived email after the name it is built from. Each column is
generated in full before its dependents, which is what lets correlation and
weighted-quota categoricals work on the whole column at once.
"""
from __future__ import annotations

import datetime as _dt
import random
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .formula import Formula
from .providers import Providers
from .schema import FieldSpec, Schema, SchemaError, TableSpec, load_schema

# A generated table is a list of row dicts; a dataset maps table name -> rows.
Rows = List[Dict[str, Any]]


@dataclass
class Dataset:
    """The output of a generation run. Pure Python; convert to pandas on demand."""

    schema: Schema
    tables: Dict[str, Rows]

    def __getitem__(self, name: str) -> Rows:
        return self.tables[name]

    def table_names(self) -> List[str]:
        return list(self.tables.keys())

    def row_counts(self) -> Dict[str, int]:
        return {name: len(rows) for name, rows in self.tables.items()}

    def to_pandas(self) -> Dict[str, "Any"]:
        import pandas as pd  # local import keeps pandas optional for pure use

        return {name: pd.DataFrame(rows) for name, rows in self.tables.items()}


class TabularGenerator:
    def __init__(self, schema: Schema):
        self.schema = schema

    # ---- public API ------------------------------------------------------
    def generate(self, seed: Optional[int] = None) -> Dataset:
        seed = self.schema.seed if seed is None else seed
        order = _topo_sort_tables(self.schema)
        tables: Dict[str, Rows] = {}
        for table_name in order:
            table = self.schema.get_table(table_name)
            # Give every table its own RNG derived deterministically from the
            # master seed and the table name, so adding a table does not shift
            # the values of unrelated tables.
            rng = random.Random(f"{seed}:{table_name}")
            tables[table_name] = self._generate_table(table, rng, tables)
        return Dataset(schema=self.schema, tables=tables)

    # ---- table-level -----------------------------------------------------
    def _generate_table(self, table: TableSpec, rng: random.Random, done: Dict[str, Rows]) -> Rows:
        n = table.rows
        providers = Providers(rng)
        columns: Dict[str, List[Any]] = {}

        for fname in _field_order(table):
            fs = table.get_field(fname)
            columns[fname] = self._generate_column(fs, table, n, rng, providers, columns, done)

        # Transpose columns into rows.
        rows: Rows = [dict() for _ in range(n)]
        for fname, values in columns.items():
            for i in range(n):
                rows[i][fname] = values[i]
        return rows

    # ---- column-level ----------------------------------------------------
    def _generate_column(
        self,
        fs: FieldSpec,
        table: TableSpec,
        n: int,
        rng: random.Random,
        providers: Providers,
        columns: Dict[str, List[Any]],
        done: Dict[str, Rows],
    ) -> List[Any]:
        t = fs.type

        if t == "category":
            values = self._gen_category(fs, n, rng)
        elif t == "foreign_key":
            values = self._gen_foreign_key(fs, n, rng, done)
        elif t == "formula":
            values = self._gen_formula(fs, n, columns)
        elif t in ("int", "float"):
            values = self._gen_numeric(fs, n, rng, columns)
        else:
            values = [self._gen_scalar(fs, rng, providers, columns, i) for i in range(n)]

        # Uniqueness enforcement (ids/emails are unique by construction; this
        # catches user-declared unique on other types).
        if fs.unique and t not in ("id", "uuid", "foreign_key"):
            values = _enforce_unique(values, fs, rng)

        # Null injection last, so declared distributions describe non-null values.
        null_rate = fs.null_rate
        if null_rate > 0 and not fs.unique:
            values = [None if rng.random() < null_rate else v for v in values]

        return values

    # ---- per-type generators --------------------------------------------
    def _gen_scalar(
        self,
        fs: FieldSpec,
        rng: random.Random,
        providers: Providers,
        columns: Dict[str, List[Any]],
        i: int,
    ) -> Any:
        t = fs.type
        if t == "id":
            strategy = fs.get("strategy", "sequential")
            if strategy == "uuid":
                return str(_uuid.UUID(int=rng.getrandbits(128)))
            start = int(fs.get("start", 1))
            prefix = fs.get("prefix", "")
            return f"{prefix}{start + i}" if prefix else start + i
        if t == "uuid":
            return str(_uuid.UUID(int=rng.getrandbits(128)))
        if t == "bool":
            p = float(fs.get("true_rate", 0.5))
            return rng.random() < p
        if t == "name":
            return providers.full_name()
        if t == "first_name":
            return providers.first_name()
        if t == "last_name":
            return providers.last_name()
        if t == "email":
            dep = fs.depends_on
            source = columns[dep][i] if dep and dep in columns else None
            return providers.email(source)
        if t == "phone":
            return providers.phone()
        if t == "city":
            return providers.city()
        if t == "country":
            return providers.country()
        if t == "address":
            return providers.address()
        if t == "company":
            return providers.company()
        if t == "job":
            return providers.job()
        if t == "url":
            return providers.url()
        if t == "ipv4":
            return providers.ipv4()
        if t == "text":
            return providers.text(int(fs.get("sentences", 2)))
        if t in ("date", "datetime"):
            return self._gen_temporal(fs, providers, t)
        raise SchemaError(f"No generator for field type '{t}'.")

    def _gen_temporal(self, fs: FieldSpec, providers: Providers, t: str) -> str:
        start = _parse_date(fs.get("start", "2020-01-01"))
        end = _parse_date(fs.get("end", "2025-12-31"))
        if t == "date":
            d = providers.date_between(start, end)
            return d.isoformat()
        s_dt = _dt.datetime.combine(start, _dt.time.min)
        e_dt = _dt.datetime.combine(end, _dt.time.max)
        return providers.datetime_between(s_dt, e_dt).isoformat(sep=" ", timespec="seconds")

    def _gen_category(self, fs: FieldSpec, n: int, rng: random.Random) -> List[Any]:
        """Largest-remainder quota assignment so observed proportions match the
        requested weights very closely, then a seeded shuffle to remove order."""
        cats: Dict[str, float] = fs.get("categories")
        total_weight = sum(cats.values())
        labels = list(cats.keys())
        # Ideal (fractional) counts.
        ideal = {k: (v / total_weight) * n for k, v in cats.items()}
        counts = {k: int(v) for k, v in ideal.items()}
        remainder = n - sum(counts.values())
        # Distribute the leftover to the largest fractional parts.
        frac_order = sorted(labels, key=lambda k: ideal[k] - counts[k], reverse=True)
        for k in frac_order[:remainder]:
            counts[k] += 1

        values: List[Any] = []
        for label in labels:
            values.extend([label] * counts[label])
        rng.shuffle(values)
        return values

    def _gen_foreign_key(
        self, fs: FieldSpec, n: int, rng: random.Random, done: Dict[str, Rows]
    ) -> List[Any]:
        ref = fs.get("references")
        ref_table, ref_col = ref.split(".", 1)
        parent_rows = done.get(ref_table, [])
        parent_values = [r[ref_col] for r in parent_rows]
        if not parent_values:
            raise SchemaError(
                f"Foreign key '{fs.name}' references '{ref}' but the parent table "
                f"is empty. Ensure '{ref_table}' has rows > 0."
            )
        if fs.unique:
            if n > len(parent_values):
                raise SchemaError(
                    f"Unique foreign key '{fs.name}' needs at least {n} parent rows "
                    f"in '{ref_table}', found {len(parent_values)}."
                )
            pool = list(parent_values)
            rng.shuffle(pool)
            return pool[:n]
        return [rng.choice(parent_values) for _ in range(n)]

    def _gen_numeric(
        self, fs: FieldSpec, n: int, rng: random.Random, columns: Dict[str, List[Any]]
    ) -> List[Any]:
        is_int = fs.type == "int"
        lo = fs.get("min", 0)
        hi = fs.get("max", 100 if is_int else 1.0)
        correlate = fs.get("correlate")

        if correlate:
            values = self._gen_correlated(fs, n, rng, columns, lo, hi)
        else:
            dist = fs.get("distribution", "uniform")
            values = [self._draw_numeric(rng, lo, hi, dist, fs) for _ in range(n)]

        return [self._finalize_numeric(v, fs, is_int) for v in values]

    def _gen_correlated(
        self,
        fs: FieldSpec,
        n: int,
        rng: random.Random,
        columns: Dict[str, List[Any]],
        lo: float,
        hi: float,
    ) -> List[float]:
        corr = fs.get("correlate")
        driver_name = corr["field"]
        if driver_name not in columns:
            raise SchemaError(
                f"Field '{fs.name}' correlates with '{driver_name}', which is not "
                f"generated before it. Check for a dependency cycle."
            )
        driver = columns[driver_name]
        strength = float(corr.get("strength", 0.6))
        negative = str(corr.get("direction", "positive")).lower() == "negative"

        # Rank-normalize the driver into [0, 1]. Non-numeric drivers fall back
        # to a stable ordering by string.
        try:
            numeric_driver = [float(x) for x in driver]
        except (TypeError, ValueError):
            numeric_driver = [float(i) for i in range(n)]

        order = sorted(range(n), key=lambda i: numeric_driver[i])
        rank = [0.0] * n
        for pos, idx in enumerate(order):
            rank[idx] = pos / (n - 1) if n > 1 else 0.5
        if negative:
            rank = [1.0 - r for r in rank]

        out: List[float] = []
        for i in range(n):
            u = rng.random()
            blended = strength * rank[i] + (1.0 - strength) * u
            out.append(lo + blended * (hi - lo))
        return out

    @staticmethod
    def _draw_numeric(rng: random.Random, lo: float, hi: float, dist: str, fs: FieldSpec) -> float:
        if dist == "normal":
            mean = float(fs.get("mean", (lo + hi) / 2.0))
            std = float(fs.get("std", (hi - lo) / 6.0 if hi > lo else 1.0))
            val = rng.gauss(mean, std)
            return min(max(val, lo), hi)
        if dist in ("exponential", "exp"):
            scale = float(fs.get("scale", (hi - lo) / 3.0 if hi > lo else 1.0))
            val = lo + rng.expovariate(1.0 / scale) if scale > 0 else lo
            return min(val, hi)
        # uniform default
        return rng.uniform(lo, hi)

    @staticmethod
    def _finalize_numeric(v: float, fs: FieldSpec, is_int: bool) -> Any:
        if is_int:
            return int(round(v))
        ndigits = fs.get("round")
        if ndigits is not None:
            return round(float(v), int(ndigits))
        return float(v)

    def _gen_formula(self, fs: FieldSpec, n: int, columns: Dict[str, List[Any]]) -> List[Any]:
        expr = fs.get("expr") or fs.get("formula")
        if not expr:
            raise SchemaError(f"Formula field '{fs.name}' needs an 'expr'.")
        formula = Formula(str(expr))
        ndigits = fs.get("round")
        values: List[Any] = []
        for i in range(n):
            row = {name: columns[name][i] for name in columns}
            val = formula.eval(row)
            if ndigits is not None and isinstance(val, (int, float)):
                val = round(float(val), int(ndigits))
            values.append(val)
        return values


# --------------------------------------------------------------------------
# ordering helpers
# --------------------------------------------------------------------------
def _topo_sort_tables(schema: Schema) -> List[str]:
    """Order tables so every FK parent is generated before its children."""
    names = [t.name for t in schema.tables]
    deps: Dict[str, set] = {name: set() for name in names}
    for table in schema.tables:
        for fs in table.fields:
            if fs.type == "foreign_key":
                parent = fs.params["references"].split(".", 1)[0]
                if parent != table.name and parent in deps:
                    deps[table.name].add(parent)

    ordered: List[str] = []
    visited: set = set()

    def visit(node: str, stack: Sequence[str]) -> None:
        if node in visited:
            return
        for parent in sorted(deps[node]):
            if parent in stack:
                raise SchemaError(f"Foreign-key cycle: {' -> '.join(list(stack) + [parent])}")
            visit(parent, list(stack) + [node])
        visited.add(node)
        ordered.append(node)

    for name in names:
        visit(name, [])
    return ordered


def _field_order(table: TableSpec) -> List[str]:
    """Order fields within a table by intra-row dependency."""
    field_names = set(table.field_names())
    deps: Dict[str, set] = {name: set() for name in table.field_names()}

    for fs in table.fields:
        d: set = set()
        if fs.depends_on and fs.depends_on in field_names:
            d.add(fs.depends_on)
        if fs.type in ("int", "float"):
            corr = fs.get("correlate")
            if corr and corr.get("field") in field_names:
                d.add(corr["field"])
        if fs.type == "formula":
            expr = fs.get("expr") or fs.get("formula")
            if expr:
                for ref in Formula(str(expr)).referenced_fields:
                    if ref in field_names:
                        d.add(ref)
        deps[fs.name] = d

    ordered: List[str] = []
    visited: set = set()

    def visit(node: str, stack: Sequence[str]) -> None:
        if node in visited:
            return
        for dep in sorted(deps[node]):
            if dep in stack:
                raise SchemaError(
                    f"Field dependency cycle in table '{table.name}': "
                    f"{' -> '.join(list(stack) + [dep])}"
                )
            visit(dep, list(stack) + [node])
        visited.add(node)
        ordered.append(node)

    for name in table.field_names():
        visit(name, [])
    return ordered


def _enforce_unique(values: List[Any], fs: FieldSpec, rng: random.Random) -> List[Any]:
    seen: set = set()
    out: List[Any] = []
    for v in values:
        candidate = v
        attempts = 0
        while candidate in seen:
            attempts += 1
            candidate = f"{v}-{attempts}"
            if attempts > len(values) + 10:
                raise SchemaError(
                    f"Cannot satisfy uniqueness on field '{fs.name}': not enough "
                    f"distinct values available."
                )
        seen.add(candidate)
        out.append(candidate)
    return out


def _parse_date(value: Any) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value))


# --------------------------------------------------------------------------
# top-level convenience
# --------------------------------------------------------------------------
def generate(schema: Any, seed: Optional[int] = None) -> Dataset:
    """Load a schema (path, dict, or Schema) and generate a dataset."""
    schema_obj = load_schema(schema)
    return TabularGenerator(schema_obj).generate(seed=seed)
