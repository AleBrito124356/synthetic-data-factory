"""Schema-driven tabular data generation — the zero-dollar core of the factory.

Given a :class:`~factory.schema.Schema`, produce related tables with:

* realistic per-type values (names, emails, dates, categoricals, ...),
* referential integrity across foreign keys (including self-references,
  which become a proper hierarchy with null roots),
* weighted categoricals that closely match requested proportions,
* simple rank-based correlations between numeric columns,
* formula/computed columns evaluated from sibling columns (null-safe),
* ``unique`` that keeps every value well-formed and inside its declared range,
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
import math
import random
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .formula import Formula, FormulaError
from .providers import Providers, enumerate_domain
from .schema import (
    DEFAULT_DATE_END,
    DEFAULT_DATE_START,
    FieldSpec,
    Schema,
    SchemaError,
    TableSpec,
    load_schema,
    parse_date,
    topological_table_order,
)

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

        out = {}
        for table in self.schema.tables:
            rows = self.tables.get(table.name, [])
            out[table.name] = pd.DataFrame(rows, columns=table.field_names())
        return out


class TabularGenerator:
    def __init__(self, schema: Schema):
        self.schema = schema

    # ---- public API ------------------------------------------------------
    def generate(self, seed: Optional[int] = None) -> Dataset:
        seed = self.schema.seed if seed is None else seed
        tables: Dict[str, Rows] = {}
        for table_name in topological_table_order(self.schema):
            table = self.schema.get_table(table_name)
            # Give every table its own RNG derived deterministically from the
            # master seed and the table name, so adding a table does not shift
            # the values of unrelated tables.
            rng = random.Random(f"{seed}:{table_name}")
            tables[table_name] = self._generate_table(table, rng, tables)
        # Present tables in schema order regardless of generation order.
        ordered = {t.name: tables[t.name] for t in self.schema.tables}
        return Dataset(schema=self.schema, tables=ordered)

    # ---- table-level -----------------------------------------------------
    def _generate_table(self, table: TableSpec, rng: random.Random, done: Dict[str, Rows]) -> Rows:
        n = table.rows
        providers = Providers(rng)
        columns: Dict[str, List[Any]] = {}

        for fname in _field_order(table):
            fs = table.get_field(fname)
            columns[fname] = self._generate_column(fs, table, n, rng, providers, columns, done)

        # Transpose columns into rows, keeping the declared field order.
        names = table.field_names()
        return [{name: columns[name][i] for name in names} for i in range(n)]

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
        ref = fs.references
        nulls_handled = False

        if t == "category":
            values = self._gen_category(fs, n, rng)
        elif t == "foreign_key" and ref and ref[0] == table.name:
            values = self._gen_self_reference(fs, n, rng, columns)
            nulls_handled = True
        elif t == "foreign_key":
            values = self._gen_foreign_key(fs, n, rng, done)
        elif t == "formula":
            values = self._gen_formula(fs, table, n, columns)
        elif t in ("int", "float"):
            values = self._gen_numeric(fs, n, rng, columns)
        else:
            values = [self._gen_scalar(fs, rng, providers, columns, i) for i in range(n)]

        if fs.unique and t not in ("id", "uuid", "foreign_key", "category"):
            values = self._make_unique(fs, table, values, rng, providers, columns)

        # Null injection last, so declared distributions describe non-null values.
        null_rate = fs.null_rate
        if null_rate > 0 and not fs.unique and not nulls_handled:
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
        start = parse_date(fs.get("start", DEFAULT_DATE_START))
        end = parse_date(fs.get("end", DEFAULT_DATE_END))
        if t == "date":
            return providers.date_between(start, end).isoformat()
        s_dt = _dt.datetime.combine(start, _dt.time.min)
        e_dt = _dt.datetime.combine(end, _dt.time(23, 59, 59))
        return providers.datetime_between(s_dt, e_dt).isoformat(sep=" ", timespec="seconds")

    def _gen_category(self, fs: FieldSpec, n: int, rng: random.Random) -> List[Any]:
        """Largest-remainder quota assignment so observed proportions match the
        requested weights very closely, then a seeded shuffle to remove order.

        ``unique: true`` draws each category at most once (the parser has
        already checked there are enough categories for every row)."""
        cats: Dict[Any, float] = fs.get("categories")
        if fs.unique:
            return rng.sample(list(cats.keys()), n)
        values = _quota_values(cats, n)
        rng.shuffle(values)
        return values

    def _gen_foreign_key(
        self, fs: FieldSpec, n: int, rng: random.Random, done: Dict[str, Rows]
    ) -> List[Any]:
        ref_table, ref_col = fs.references
        parent_values = [r[ref_col] for r in done.get(ref_table, [])]
        if n == 0:
            return []
        if not parent_values:
            raise SchemaError(
                f"Foreign key '{fs.name}' references '{ref_table}.{ref_col}' but the "
                f"parent table is empty. Ensure '{ref_table}' has rows > 0."
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

    def _gen_self_reference(
        self, fs: FieldSpec, n: int, rng: random.Random, columns: Dict[str, List[Any]]
    ) -> List[Any]:
        """A self-referencing FK (``employees.manager_id -> employees.emp_id``)
        becomes a forest: row ``i`` points at a row with a smaller index, so the
        graph can never contain a cycle, and roots are null. Row 0 is always a
        root; ``null_rate`` adds extra roots."""
        _, ref_col = fs.references
        keys = columns[ref_col]
        extra_roots = fs.null_rate
        out: List[Any] = []
        for i in range(n):
            if i == 0 or (extra_roots > 0 and rng.random() < extra_roots):
                out.append(None)
            else:
                out.append(keys[rng.randrange(i)])
        return out

    def _gen_numeric(
        self, fs: FieldSpec, n: int, rng: random.Random, columns: Dict[str, List[Any]]
    ) -> List[Any]:
        is_int = fs.type == "int"
        lo, hi = _numeric_bounds(fs)
        correlate = fs.get("correlate")

        if correlate:
            values = self._gen_correlated(fs, n, rng, columns, lo, hi)
        else:
            dist = fs.get("distribution", "uniform")
            values = [_draw_numeric(rng, lo, hi, dist, fs) for _ in range(n)]

        return [_finalize_numeric(v, fs, is_int, lo, hi) for v in values]

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

        # Rank-normalize the driver into [0, 1]. Nulls and non-numeric drivers
        # sort by row position so the result stays deterministic.
        numeric_driver: List[float] = []
        for i, x in enumerate(driver):
            try:
                numeric_driver.append(float(x))
            except (TypeError, ValueError):
                numeric_driver.append(float(i))

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

    def _gen_formula(
        self, fs: FieldSpec, table: TableSpec, n: int, columns: Dict[str, List[Any]]
    ) -> List[Any]:
        expr = fs.get("expr") or fs.get("formula")
        formula = Formula(str(expr))
        refs = [r for r in formula.referenced_fields if r in columns]
        ndigits = fs.get("round")
        values: List[Any] = []
        for i in range(n):
            row = {name: columns[name][i] for name in refs}
            try:
                val = formula.eval(row)
            except FormulaError as exc:
                raise SchemaError(
                    f"Formula field '{table.name}.{fs.name}' failed on row {i}: {exc}"
                ) from exc
            if ndigits is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                val = round(float(val), int(ndigits))
            values.append(val)
        return values

    # ---- uniqueness ------------------------------------------------------
    def _make_unique(
        self,
        fs: FieldSpec,
        table: TableSpec,
        values: List[Any],
        rng: random.Random,
        providers: Providers,
        columns: Dict[str, List[Any]],
    ) -> List[Any]:
        """Replace duplicates with *well-formed* values of the same type.

        * emails get a counter inside the local part (``ana.diaz2@example.com``),
        * urls get a counter inside the host label,
        * numbers, dates and small-domain types are re-drawn from the field's
          own distribution and range (then from the unused remainder of the
          domain, which the parser has proven is large enough),
        * formula columns cannot be re-drawn, so duplicates raise an error.
        """
        t = fs.type
        where = f"'{table.name}.{fs.name}'"
        if t == "formula":
            seen: set = set()
            for v in values:
                if v in seen:
                    raise SchemaError(
                        f"{where} is unique but its formula produced a duplicate value {v!r}."
                    )
                seen.add(v)
            return values
        if t == "email":
            return _unique_by_counter(values, _email_with_counter)
        if t == "url":
            return _unique_by_counter(values, _url_with_counter)

        draw = self._redraw_fn(fs, rng, providers, columns)
        domain_fn = lambda: _finite_domain(fs)  # noqa: E731 - built lazily
        out: List[Any] = []
        seen = set()
        remaining: Optional[List[Any]] = None
        for i, v in enumerate(values):
            candidate = v
            attempts = 0
            while candidate in seen and attempts < 25:
                candidate = draw(i)
                attempts += 1
            if candidate in seen:
                if remaining is None:
                    domain = domain_fn()
                    if domain is None:
                        # Unbounded type (text, address, datetime, raw float):
                        # keep drawing — collisions are astronomically rare.
                        for _ in range(1000):
                            candidate = draw(i)
                            if candidate not in seen:
                                break
                        else:
                            raise SchemaError(
                                f"Cannot satisfy uniqueness on {where}: not enough distinct values."
                            )
                    else:
                        remaining = [d for d in domain if d not in seen]
                if remaining is not None:
                    while remaining:
                        j = rng.randrange(len(remaining))
                        remaining[j], remaining[-1] = remaining[-1], remaining[j]
                        pick = remaining.pop()
                        if pick not in seen:
                            candidate = pick
                            break
                    else:
                        raise SchemaError(
                            f"Cannot satisfy uniqueness on {where}: not enough distinct values."
                        )
            seen.add(candidate)
            out.append(candidate)
        return out

    def _redraw_fn(
        self,
        fs: FieldSpec,
        rng: random.Random,
        providers: Providers,
        columns: Dict[str, List[Any]],
    ) -> Callable[[int], Any]:
        t = fs.type
        if t in ("int", "float"):
            lo, hi = _numeric_bounds(fs)
            dist = fs.get("distribution", "uniform")
            if fs.get("correlate"):
                dist = "uniform"
            is_int = t == "int"
            return lambda i: _finalize_numeric(_draw_numeric(rng, lo, hi, dist, fs), fs, is_int, lo, hi)
        return lambda i: self._gen_scalar(fs, rng, providers, columns, i)


# --------------------------------------------------------------------------
# value helpers
# --------------------------------------------------------------------------
def _numeric_bounds(fs: FieldSpec):
    is_int = fs.type == "int"
    lo = fs.get("min", 0)
    hi = fs.get("max", 100 if is_int else 1.0)
    return lo, hi


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


def _finalize_numeric(v: float, fs: FieldSpec, is_int: bool, lo: float, hi: float) -> Any:
    if is_int:
        # Round, then clamp so rounding can never step outside [min, max].
        return min(max(int(round(v)), math.ceil(lo)), math.floor(hi))
    ndigits = fs.get("round")
    if ndigits is not None:
        return min(max(round(float(v), int(ndigits)), lo), hi)
    return float(v)


def _quota_values(cats: Dict[Any, float], n: int) -> List[Any]:
    """Largest-remainder split of ``n`` rows across weighted labels."""
    total_weight = sum(cats.values())
    labels = list(cats.keys())
    ideal = {k: (v / total_weight) * n for k, v in cats.items()}
    counts = {k: int(v) for k, v in ideal.items()}
    remainder = n - sum(counts.values())
    frac_order = sorted(labels, key=lambda k: ideal[k] - counts[k], reverse=True)
    for k in frac_order[:remainder]:
        counts[k] += 1
    values: List[Any] = []
    for label in labels:
        values.extend([label] * counts[label])
    return values


def _unique_by_counter(values: List[Any], with_counter: Callable[[Any, int], Any]) -> List[Any]:
    seen: set = set(v for v in values)
    first_seen: set = set()
    out: List[Any] = []
    for v in values:
        if v not in first_seen:
            first_seen.add(v)
            out.append(v)
            continue
        k = 2
        candidate = with_counter(v, k)
        while candidate in seen:
            k += 1
            candidate = with_counter(v, k)
        seen.add(candidate)
        first_seen.add(candidate)
        out.append(candidate)
    return out


def _email_with_counter(value: Any, k: int) -> Any:
    if not isinstance(value, str) or "@" not in value:
        return value
    local, domain = value.rsplit("@", 1)
    return f"{local}{k}@{domain}"


def _url_with_counter(value: Any, k: int) -> Any:
    if not isinstance(value, str) or "://" not in value:
        return value
    scheme, rest = value.split("://", 1)
    host, sep, path = rest.partition("/")
    label, dot, domain = host.partition(".")
    return f"{scheme}://{label}{k}{dot}{domain}{sep}{path}"


def _finite_domain(fs: FieldSpec) -> Optional[List[Any]]:
    """Every value a unique field may take, when that set is small enough to
    enumerate (ints, rounded floats, dates, provider pools)."""
    t = fs.type
    if t == "int":
        lo, hi = _numeric_bounds(fs)
        a, b = math.ceil(lo), math.floor(hi)
        if b - a > 5_000_000:
            return None
        return list(range(a, b + 1))
    if t == "float":
        nd = fs.get("round")
        if nd is None:
            return None
        lo, hi = _numeric_bounds(fs)
        scale = 10 ** int(nd)
        a, b = math.ceil(lo * scale - 1e-9), math.floor(hi * scale + 1e-9)
        if b - a > 5_000_000:
            return None
        return [round(i / scale, int(nd)) for i in range(a, b + 1)]
    if t == "date":
        start = parse_date(fs.get("start", DEFAULT_DATE_START))
        end = parse_date(fs.get("end", DEFAULT_DATE_END))
        days = (end - start).days
        return [(start + _dt.timedelta(days=d)).isoformat() for d in range(days + 1)]
    return enumerate_domain(t)


# --------------------------------------------------------------------------
# ordering helpers
# --------------------------------------------------------------------------
def _topo_sort_tables(schema: Schema) -> List[str]:
    """Backwards-compatible alias for :func:`factory.schema.topological_table_order`."""
    return topological_table_order(schema)


def field_dependencies(table: TableSpec, fs: FieldSpec) -> set:
    """Sibling fields that must be generated before ``fs``."""
    field_names = set(table.field_names())
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
    ref = fs.references
    if ref and ref[0] == table.name and ref[1] in field_names:
        d.add(ref[1])  # self-reference: keys first, then the pointers
    d.discard(fs.name)
    return d


def _field_order(table: TableSpec) -> List[str]:
    """Order fields within a table by intra-row dependency."""
    deps: Dict[str, set] = {fs.name: field_dependencies(table, fs) for fs in table.fields}

    ordered: List[str] = []
    visited: set = set()

    def visit(node: str, stack: Sequence[str]) -> None:
        if node in visited:
            return
        for dep in sorted(deps[node]):
            if dep in stack or dep == node:
                raise SchemaError(
                    f"Field dependency cycle in table '{table.name}': "
                    f"{' -> '.join(list(stack) + [node, dep])}"
                )
            visit(dep, list(stack) + [node])
        visited.add(node)
        ordered.append(node)

    for name in table.field_names():
        visit(name, [])
    return ordered


def _parse_date(value: Any) -> _dt.date:
    """Backwards-compatible alias for :func:`factory.schema.parse_date`."""
    return parse_date(value)


# --------------------------------------------------------------------------
# top-level convenience
# --------------------------------------------------------------------------
def generate(schema: Any, seed: Optional[int] = None) -> Dataset:
    """Load a schema (path, dict, or Schema) and generate a dataset."""
    schema_obj = load_schema(schema)
    return TabularGenerator(schema_obj).generate(seed=seed)
