"""Infer a generator schema from real tabular data (``sdf infer``).

Point it at a folder of CSV/JSONL files or a SQLite database and it writes a
schema YAML that ``sdf generate`` accepts, so a synthetic stand-in with the
same *shape* as production is one command away:

* **Types** — sequential/prefixed ids, uuid, email, bool, int, float, date,
  datetime, weighted categories, and pool-backed types (name, phone, city,
  company, ...) whose synthetic values come from fictional pools.
* **Statistics** — min/max, rounding, null rate, a fitted distribution
  (uniform / normal / exponential), category weights, bool true-rate.
* **Relations** — primary keys; foreign keys by value containment in a
  unique parent column (confirmed by naming or non-numeric key values);
  ``min_per_parent`` and ``skew: zipf`` from the observed child counts;
  ``lookup`` columns that always equal a parent column; ``after``/``before``
  when a date always follows/precedes a parent or earlier sibling date;
  ``formula`` columns that are exactly ``a*b``, ``a+b``, ``a-b`` or ``a*c``
  of sibling numbers (after rounding); rank correlations between numeric
  columns; name-derived emails.

Privacy: real values are never copied into the schema except category labels,
and a label seen fewer than ``min_category_count`` times (default 5) is merged
into an ``other`` bucket so rare — potentially identifying — values do not
survive. Free-text, names, emails, phones etc. are regenerated from fictional
pools. This limits disclosure; it is not a formal privacy guarantee.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .load import RawData, RawTable, coerce_to_schema
from .schema import Schema, SchemaError, topological_table_order

_INT_RE = re.compile(r"^[+-]?\d+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?([+-]\d{2}:?\d{2}|Z)?$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PREFIXED_ID_RE = re.compile(r"^(\D*?)(\d+)$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_PERSON_RE = re.compile(r"^[A-Z][a-zA-Z'À-ſ-]+( [A-Z][a-zA-Z'À-ſ-]+){1,2}$")

MIN_EVIDENCE_ROWS = 20  # rows needed before a relation (lookup/after) is trusted


class InferenceError(ValueError):
    """The data could not be turned into a valid schema."""


@dataclass
class Column:
    table: str
    name: str
    raw: List[Any]                 # every value, None for nulls
    kind: str = "empty"            # bool,int,float,date,datetime,uuid,email,string,empty
    parsed: List[Any] = field(default_factory=list)  # typed values aligned with raw (None = null)
    spec: Dict[str, Any] = field(default_factory=dict)
    is_key: bool = False           # unique, non-null, key-like
    is_pk: bool = False
    suppressed: int = 0            # rare category labels merged into "other"

    @property
    def non_null(self) -> List[Any]:
        return [p for p in self.parsed if p is not None]

    @property
    def n(self) -> int:
        return len(self.raw)

    @property
    def nulls(self) -> int:
        return sum(1 for v in self.raw if v is None)


@dataclass
class InferenceResult:
    schema_dict: Dict[str, Any]
    schema: Schema
    source_rows: Dict[str, int]
    relations: List[str]
    suppressed: Dict[str, int]
    notes: List[str] = field(default_factory=list)

    def to_yaml(self, header: str = "") -> str:
        body = yaml.safe_dump(self.schema_dict, sort_keys=False, allow_unicode=True, width=100)
        return header + body

    def summary(self) -> str:
        lines = []
        for table in self.schema.tables:
            src = self.source_rows.get(table.name, 0)
            pk = next((f.name for f in table.fields if f.type in ("id",)), None)
            lines.append(f"  {table.name}: {src} -> {table.rows} rows, {len(table.fields)} fields"
                         + (f", key {pk}" if pk else ""))
        if self.relations:
            lines.append("Relations:")
            lines.extend(f"  {r}" for r in self.relations)
        if self.suppressed:
            total = sum(self.suppressed.values())
            cols = ", ".join(f"{k} ({v})" for k, v in self.suppressed.items())
            lines.append(f"Privacy: {total} rare category value(s) merged into 'other': {cols}")
        lines.extend(self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def infer_schema(
    raw: RawData,
    rows_scale: float = 1.0,
    min_category_count: int = 5,
    seed: int = 42,
    max_categories: int = 50,
) -> InferenceResult:
    """Profile ``raw`` and return a validated schema that reproduces its shape."""
    if rows_scale <= 0:
        raise InferenceError("rows_scale must be positive.")
    columns: Dict[str, List[Column]] = {}
    for name, table in raw.tables.items():
        columns[name] = [_profile_column(name, col, table) for col in table.columns]

    for name, cols in columns.items():
        for col in cols:
            _base_spec(col, min_category_count, max_categories)
        _pick_primary_key(cols)

    rows = {name: _scaled_rows(len(t.rows), rows_scale) for name, t in raw.tables.items()}
    relations: List[str] = []
    fks = _detect_foreign_keys(columns, raw, rows, relations)
    order = _table_order(list(raw.tables), fks)
    _detect_lookups(columns, raw, fks, relations)
    _detect_temporal(columns, raw, fks, order, relations)
    _detect_formulas(columns, relations)
    _detect_correlations(columns, relations)
    _detect_email_names(columns)
    _fix_unique_capacity(columns, rows)

    schema_dict: Dict[str, Any] = {"seed": int(seed), "tables": []}
    for name in order:
        fields_out = [dict(name=c.name, **c.spec) for c in columns[name]]
        schema_dict["tables"].append({"name": name, "rows": rows[name], "fields": fields_out})
    try:
        schema = Schema.from_dict(schema_dict)
    except SchemaError as exc:  # pragma: no cover - would be an inference bug
        raise InferenceError(f"Inferred schema is invalid: {exc}") from exc
    suppressed = {f"{c.table}.{c.name}": c.suppressed for cols in columns.values() for c in cols if c.suppressed}
    return InferenceResult(
        schema_dict=schema_dict,
        schema=schema,
        source_rows={name: len(t.rows) for name, t in raw.tables.items()},
        relations=relations,
        suppressed=suppressed,
    )


# --------------------------------------------------------------------------
# per-column profiling
# --------------------------------------------------------------------------
def _profile_column(table: str, name: str, raw_table: RawTable) -> Column:
    values = [r.get(name) for r in raw_table.rows]
    values = [None if (isinstance(v, str) and v.strip() == "") else v for v in values]
    col = Column(table=table, name=name, raw=values)
    col.kind, col.parsed = _detect_kind(name, values)
    return col


def _detect_kind(name: str, values: List[Any]) -> Tuple[str, List[Any]]:
    present = [v for v in values if v is not None]
    if not present:
        return "empty", [None] * len(values)

    def parse_all(fn) -> Optional[List[Any]]:
        out = []
        for v in values:
            if v is None:
                out.append(None)
                continue
            p = fn(v)
            if p is _FAIL:
                return None
            out.append(p)
        return out

    lname = name.lower()
    boolish_name = lname.startswith(("is_", "has_", "can_", "should_", "was_")) or lname.endswith("_flag")
    for kind, fn in (
        ("bool", lambda v: _parse_bool(v, allow_01=boolish_name)),
        ("int", _parse_int),
        ("float", _parse_float),
        ("date", _parse_date),
        ("datetime", _parse_datetime),
        ("uuid", lambda v: str(v) if _UUID_RE.match(str(v).strip()) else _FAIL),
        ("email", lambda v: str(v).strip() if _EMAIL_RE.match(str(v).strip()) else _FAIL),
    ):
        parsed = parse_all(fn)
        if parsed is not None:
            return kind, parsed
    return "string", [None if v is None else str(v) for v in values]


class _Fail:
    pass


_FAIL = _Fail()


def _parse_bool(v: Any, allow_01: bool) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and allow_01 and v in (0, 1):
        return bool(v)
    text = str(v).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    if allow_01 and text in ("0", "1"):
        return text == "1"
    return _FAIL


def _parse_int(v: Any) -> Any:
    if isinstance(v, bool):
        return _FAIL
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return _FAIL
    text = str(v).strip()
    digits = text.lstrip("+-")
    if _INT_RE.match(text) and not (len(digits) > 1 and digits.startswith("0")):
        return int(text)
    return _FAIL


def _parse_float(v: Any) -> Any:
    if isinstance(v, bool):
        return _FAIL
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else _FAIL
    try:
        f = float(str(v).strip())
    except ValueError:
        return _FAIL
    return f if math.isfinite(f) else _FAIL


def _parse_date(v: Any) -> Any:
    text = str(v).strip()
    if not _DATE_RE.match(text):
        return _FAIL
    try:
        return _dt.date.fromisoformat(text)
    except ValueError:
        return _FAIL


def _parse_datetime(v: Any) -> Any:
    text = str(v).strip()
    if not _DATETIME_RE.match(text):
        return _FAIL
    try:
        value = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _FAIL
    return value.replace(tzinfo=None)


def _decimals(values: Sequence[Any], raw: Sequence[Any]) -> int:
    best = 0
    for v in raw:
        if v is None:
            continue
        text = repr(v) if isinstance(v, float) else str(v).strip()
        if "e" in text.lower():
            return 6
        if "." in text:
            best = max(best, len(text.split(".", 1)[1].rstrip("0")))
    return min(best, 6)


def _base_spec(col: Column, min_count: int, max_categories: int) -> None:
    n_present = len(col.non_null)
    spec: Dict[str, Any] = {}
    kind = col.kind
    values = col.non_null
    distinct = len(set(values))
    unique = n_present > 0 and distinct == n_present and col.nulls == 0

    if kind == "empty":
        spec = {"type": "text", "sentences": 1}
    elif kind == "bool":
        spec = {"type": "bool", "true_rate": round(sum(values) / n_present, 4)}
    elif kind == "int":
        lo, hi = min(values), max(values)
        if unique and sorted(values) == list(range(lo, lo + n_present)):
            spec = {"type": "id", "start": lo}
            col.is_key = True
        else:
            spec = {"type": "int", "min": lo, "max": hi, **_fit_distribution(values, lo, hi, is_int=True)}
            col.is_key = unique
    elif kind == "float":
        lo, hi = min(values), max(values)
        spec = {"type": "float", "min": _num(lo), "max": _num(hi)}
        nd = _decimals(values, col.raw)
        if nd < 6:
            spec["round"] = nd
        spec.update(_fit_distribution(values, lo, hi))
    elif kind in ("date", "datetime"):
        spec = {"type": kind, "start": min(values).isoformat()[:10], "end": max(values).isoformat()[:10]}
    elif kind == "uuid":
        spec = {"type": "uuid"}
        col.is_key = unique
    elif kind == "email":
        spec = {"type": "email"}
        if distinct == n_present:
            spec["unique"] = True
            col.is_key = unique
    else:
        spec = _string_spec(col, values, unique, min_count, max_categories)

    null_rate = col.nulls / col.n if col.n else 0.0
    if null_rate > 0 and kind != "empty":
        spec["null_rate"] = round(null_rate, 4)
    elif kind == "empty":
        spec["null_rate"] = 1.0
    col.spec = spec


def _string_spec(col: Column, values: List[str], unique: bool, min_count: int, max_categories: int) -> Dict[str, Any]:
    n_present = len(values)
    if unique:
        prefixed = _prefixed_id(values)
        if prefixed is not None:
            col.is_key = True
            return prefixed
    counts = Counter(values)
    if len(counts) <= max_categories and len(counts) <= max(1, n_present // 2):
        kept = {v: c for v, c in counts.items() if c >= min_count}
        rare = {v: c for v, c in counts.items() if c < min_count}
        if kept:
            if rare:
                kept["other"] = kept.get("other", 0) + sum(rare.values())
                col.suppressed = len(rare)
            total = sum(kept.values())
            weights = {str(v): round(c / total, 4) for v, c in sorted(kept.items(), key=lambda kv: -kv[1])}
            return {"type": "category", "categories": weights}
    semantic = _semantic_type(col.name, values)
    if unique and semantic["type"] in ("text", "url", "address"):
        col.is_key = " " not in values[0]
    return semantic


def _prefixed_id(values: List[str]) -> Optional[Dict[str, Any]]:
    prefix = None
    numbers = []
    for v in values:
        m = _PREFIXED_ID_RE.match(v.strip())
        if not m:
            return None
        p, digits = m.group(1), m.group(2)
        if prefix is None:
            prefix = p
        elif p != prefix:
            return None
        if len(digits) > 1 and digits.startswith("0"):
            return None  # zero-padded: the generator would not reproduce it
        numbers.append(int(digits))
    if not prefix:
        return None
    lo = min(numbers)
    if sorted(numbers) != list(range(lo, lo + len(numbers))):
        return None
    return {"type": "id", "prefix": prefix, "start": lo}


def _semantic_type(name: str, values: List[str]) -> Dict[str, Any]:
    lname = name.lower()
    hints = [
        ("first_name", ("first_name", "firstname", "given_name")),
        ("last_name", ("last_name", "lastname", "surname", "family_name")),
        ("company", ("company", "employer", "organization", "organisation")),
        ("city", ("city", "town")),
        ("country", ("country",)),
        ("address", ("address", "street")),
        ("job", ("job", "occupation", "job_title", "role")),
        ("phone", ("phone", "mobile", "tel")),
        ("url", ("url", "website", "homepage")),
        ("ipv4", ("ip", "ipv4", "ip_address")),
    ]
    for ftype, keys in hints:
        if any(k == lname or lname.endswith("_" + k) or lname.startswith(k + "_") for k in keys):
            return {"type": ftype}
    sample = values[:500]

    def share(pattern) -> float:
        return sum(1 for v in sample if pattern.match(v.strip())) / len(sample)

    if share(_IPV4_RE) >= 0.95:
        return {"type": "ipv4"}
    if share(_URL_RE) >= 0.95:
        return {"type": "url"}
    if share(_PHONE_RE) >= 0.95 and all(sum(ch.isdigit() for ch in v) >= 7 for v in sample):
        return {"type": "phone"}
    if "name" in lname and share(_PERSON_RE) >= 0.9:
        return {"type": "name"}
    if share(_PERSON_RE) >= 0.95:
        return {"type": "name"}
    sentences = [max(1, v.count(". ") + (1 if v.rstrip().endswith(".") else 0)) for v in sample]
    return {"type": "text", "sentences": max(1, round(sum(sentences) / len(sentences)))}


def _fit_distribution(values: Sequence[float], lo: float, hi: float, is_int: bool = False) -> Dict[str, Any]:
    """Pick uniform / normal / exponential from the first three moments."""
    n = len(values)
    if n < 3 or hi <= lo:
        return {}
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    if std == 0:
        return {}
    skew = sum((x - mean) ** 3 for x in values) / n / std ** 3
    # Standard deviation a uniform draw over [lo, hi] would have (discrete for ints).
    uniform_std = math.sqrt(((hi - lo + 1) ** 2 - 1) / 12) if is_int else (hi - lo) / math.sqrt(12)
    if abs(std - uniform_std) / uniform_std < 0.15 and abs(skew) < 0.5:
        return {}
    if skew > 1.0:
        return {"distribution": "exponential", "scale": _num(max(mean - lo, 1e-9))}
    return {"distribution": "normal", "mean": _num(mean), "std": _num(std)}


def _num(x: float) -> Any:
    if isinstance(x, int):
        return x
    return float(f"{x:.6g}")


def _pick_primary_key(cols: List[Column]) -> None:
    """The first id-like column is the table's primary key; later id-like
    columns are ordinary unique ints/strings (often 1:1 foreign keys)."""
    seen_pk = False
    for col in cols:
        if col.spec.get("type") == "id":
            if seen_pk:
                if "prefix" in col.spec:
                    col.spec = {"type": "text", "sentences": 1}
                else:
                    lo = min(col.non_null)
                    col.spec = {"type": "int", "min": lo, "max": max(col.non_null)}
                col.is_key = True
            else:
                col.is_pk = True
                seen_pk = True


def _scaled_rows(n: int, scale: float) -> int:
    if n == 0:
        return 0
    return max(1, int(round(n * scale)))


# --------------------------------------------------------------------------
# relations
# --------------------------------------------------------------------------
@dataclass
class _FK:
    child: Column
    parent: Column


def _singular(name: str) -> str:
    lname = name.lower()
    if lname.endswith("ies"):
        return lname[:-3] + "y"
    if lname.endswith("ses") or lname.endswith("xes"):
        return lname[:-2]
    if lname.endswith("s") and not lname.endswith("ss"):
        return lname[:-1]
    return lname


def _name_match(child: Column, parent: Column) -> bool:
    c, k, p = child.name.lower(), parent.name.lower(), _singular(parent.table)
    if c == k:
        return True
    if c in (f"{p}_id", f"{p}_uuid", f"{p}_key", f"{p}_code", p, f"{parent.table.lower()}_id"):
        return True
    return k not in ("id", "uuid", "key") and (c.endswith("_" + k) or c.endswith(k) and len(k) > 3)


def _key_type(col: Column) -> Optional[str]:
    if col.kind == "int":
        return "int"
    if col.kind in ("string", "uuid", "email"):
        return "str"
    return None


def _detect_foreign_keys(columns: Dict[str, List[Column]], raw: RawData, rows: Dict[str, int],
                         relations: List[str]) -> List[_FK]:
    keys = [c for cols in columns.values() for c in cols if c.is_key and _key_type(c)]
    key_sets = {(k.table, k.name): set(k.non_null) for k in keys}
    candidates: List[Tuple[tuple, _FK]] = []
    for table, cols in columns.items():
        for col in cols:
            if col.is_pk or _key_type(col) is None or not col.non_null:
                continue
            values = set(col.non_null)
            best = None
            for key in keys:
                if key is col or _key_type(key) != _key_type(col):
                    continue
                if not values <= key_sets[(key.table, key.name)]:
                    continue
                named = _name_match(col, key)
                strong = _key_type(col) == "str" and col.kind != "email"
                if key.table == table:
                    # Self-reference: a nullable *_id column pointing at the PK.
                    if not (key.is_pk and col.nulls > 0 and col.name.lower().endswith(("_id", "_uuid"))):
                        continue
                elif not (named or strong):
                    continue
                score = (named, strong, -len(key_sets[(key.table, key.name)]))
                if best is None or score > best[0]:
                    best = (score, key)
            if best is not None:
                candidates.append((best[0], _FK(child=col, parent=best[1])))

    # Keep the strongest edges first and never create a table cycle.
    candidates.sort(key=lambda c: c[0], reverse=True)
    edges: Dict[str, set] = {t: set() for t in columns}
    chosen: List[_FK] = []
    for _, fk in candidates:
        c_t, p_t = fk.child.table, fk.parent.table
        if c_t != p_t and _reaches(edges, p_t, c_t):
            continue
        if c_t != p_t:
            edges[c_t].add(p_t)
        chosen.append(fk)
        _apply_fk(fk, columns, rows, relations)
    return chosen


def _reaches(edges: Dict[str, set], start: str, target: str) -> bool:
    stack, seen = [start], set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edges.get(node, ()))
    return False


def _apply_fk(fk: _FK, columns: Dict[str, List[Column]], rows: Dict[str, int], relations: List[str]) -> None:
    child, parent = fk.child, fk.parent
    spec: Dict[str, Any] = {"type": "foreign_key", "references": f"{parent.table}.{parent.name}"}
    if parent.spec.get("type") not in ("id", "uuid") and not parent.spec.get("unique"):
        parent.spec["unique"] = True
    note = ""
    values = child.non_null
    self_ref = child.table == parent.table
    if child.nulls:
        spec["null_rate"] = round(child.nulls / child.n, 4)
    if not self_ref and len(set(values)) == len(values) and child.nulls == 0:
        spec["unique"] = True
        note = " (1:1)"
    elif not self_ref:
        counts = Counter(values)
        per_parent = [counts.get(k, 0) for k in parent.non_null]
        min_children = min(per_parent) if per_parent else 0
        if min_children >= 1:
            parent_rows, child_rows = rows[parent.table], rows[child.table]
            feasible = child_rows // parent_rows if parent_rows else 0
            mpp = min(min_children, feasible)
            if mpp >= 1:
                spec["min_per_parent"] = mpp
                note += f", min_per_parent {mpp}"
        s = _zipf_exponent(sorted(per_parent, reverse=True))
        if s is not None:
            spec["skew"] = "zipf"
            spec["zipf_s"] = s
            note += f", zipf s={s}"
    child.spec = spec
    child.suppressed = 0  # FK values are regenerated from parent keys
    kind = "self-reference" if self_ref else "foreign key"
    relations.append(f"{kind}: {child.table}.{child.name} -> {parent.table}.{parent.name}{note}")


def _zipf_exponent(counts_desc: List[int]) -> Optional[float]:
    """Slope of log(count) vs log(rank); ~0 for uniform draws."""
    pts = [(math.log(r), math.log(c)) for r, c in enumerate(counts_desc, start=1) if c > 0]
    if len(pts) < 8:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    s = -slope
    return round(s, 2) if s >= 0.5 else None


def _table_order(names: List[str], fks: List[_FK]) -> List[str]:
    deps = {n: set() for n in names}
    for fk in fks:
        if fk.child.table != fk.parent.table:
            deps[fk.child.table].add(fk.parent.table)
    ordered: List[str] = []
    seen: set = set()

    def visit(node: str) -> None:
        if node in seen:
            return
        seen.add(node)
        for parent in sorted(deps[node]):
            visit(parent)
        ordered.append(node)

    for name in names:
        visit(name)
    return ordered


def _row_index(raw: RawData, col: Column) -> Dict[Any, int]:
    return {v: i for i, v in enumerate(col.parsed) if v is not None}


def _fk_parent_rows(raw: RawData, fk: _FK) -> List[Optional[int]]:
    """For each child row, the index of its parent row (or None)."""
    index = _row_index(raw, fk.parent)
    return [index.get(v) if v is not None else None for v in fk.child.parsed]


def _detect_lookups(columns: Dict[str, List[Column]], raw: RawData, fks: List[_FK],
                    relations: List[str]) -> None:
    for fk in fks:
        if fk.child.table == fk.parent.table:
            continue
        parent_rows = _fk_parent_rows(raw, fk)
        parent_cols = [c for c in columns[fk.parent.table] if c is not fk.parent and c.kind != "empty"]
        for col in columns[fk.child.table]:
            if col.spec.get("type") in ("foreign_key", "id", "lookup") or col.is_key or col.kind == "empty":
                continue
            for pcol in parent_cols:
                if pcol.kind != col.kind:
                    continue
                checked, equal = 0, True
                for i, j in enumerate(parent_rows):
                    if j is None or col.parsed[i] is None:
                        continue
                    checked += 1
                    a, b = col.parsed[i], pcol.parsed[j]
                    if isinstance(a, float) or isinstance(b, float):
                        if b is None or abs(float(a) - float(b)) > 1e-9 * max(1.0, abs(float(b))):
                            equal = False
                            break
                    elif a != b:
                        equal = False
                        break
                if equal and checked >= min(MIN_EVIDENCE_ROWS, max(2, col.n)) and len(set(col.non_null)) > 1:
                    col.spec = {"type": "lookup", "via": fk.child.name, "column": pcol.name}
                    relations.append(
                        f"lookup: {col.table}.{col.name} = {fk.parent.table}.{pcol.name} via {fk.child.name}")
                    break


def _as_dt(v: Any) -> Optional[_dt.datetime]:
    if v is None:
        return None
    if isinstance(v, _dt.datetime):
        return v
    return _dt.datetime.combine(v, _dt.time.min)


def _gaps(values: List[Any], anchors: List[Any], coarse: bool) -> List[float]:
    out = []
    for v, a in zip(values, anchors):
        if v is None or a is None:
            continue
        if coarse:
            out.append(float((_as_dt(v).date() - _as_dt(a).date()).days))
        else:
            out.append((_as_dt(v) - _as_dt(a)).total_seconds() / 86400.0)
    return out


def _detect_temporal(columns: Dict[str, List[Column]], raw: RawData, fks: List[_FK], order: List[str],
                     relations: List[str]) -> None:
    fk_by_child = {(fk.child.table, fk.child.name): fk for fk in fks}
    for table in order:
        cols = columns[table]
        for pos, col in enumerate(cols):
            if col.kind not in ("date", "datetime") or col.spec.get("type") not in ("date", "datetime"):
                continue
            options = []  # (kind, anchor_name, anchor_col, gaps)
            for sib in cols[:pos]:
                if sib.kind in ("date", "datetime") and sib.spec.get("type") in ("date", "datetime"):
                    coarse = "date" in (col.kind, sib.kind)
                    options.append((sib.name, sib, _gaps(col.parsed, sib.parsed, coarse)))
            for other in cols:
                fk = fk_by_child.get((table, other.name))
                if fk is None or fk.parent.table == table:
                    continue
                parent_rows = _fk_parent_rows(raw, fk)
                for pcol in columns[fk.parent.table]:
                    if pcol.kind not in ("date", "datetime") or pcol.spec.get("type") not in ("date", "datetime"):
                        continue
                    anchor_vals = [pcol.parsed[j] if j is not None else None for j in parent_rows]
                    coarse = "date" in (col.kind, pcol.kind)
                    options.append((f"{other.name}.{pcol.name}", pcol, _gaps(col.parsed, anchor_vals, coarse)))
            enough = [o for o in options if len(o[2]) >= min(MIN_EVIDENCE_ROWS, max(2, col.n))]
            afters = [o for o in enough if min(o[2]) >= 0 and max(o[2]) > 0]
            befores = [o for o in enough if max(o[2]) <= 0 and min(o[2]) < 0]
            if afters:
                name, anchor, gaps = min(afters, key=lambda o: max(o[2]))
                span = (_dt.date.fromisoformat(col.spec["end"]) - _dt.date.fromisoformat(col.spec["start"])).days
                col.spec["after"] = name
                if min(gaps) >= 1:
                    col.spec["min_days"] = int(math.floor(min(gaps)))
                if max(gaps) < 0.5 * max(span, 1):
                    col.spec["max_days"] = int(math.ceil(max(gaps)))
                anchor_end = anchor.spec.get("end")
                if anchor_end:
                    need = _dt.date.fromisoformat(anchor_end) + _dt.timedelta(days=col.spec.get("min_days", 0))
                    if need.isoformat() > col.spec["end"]:
                        col.spec["end"] = need.isoformat()
                if "max_days" in col.spec and anchor.spec.get("start"):
                    reach = _dt.date.fromisoformat(anchor.spec["start"]) + _dt.timedelta(days=col.spec["max_days"])
                    if reach.isoformat() < col.spec["start"]:
                        col.spec["start"] = reach.isoformat()
                relations.append(f"temporal: {table}.{col.name} after {name}")
            elif befores:
                name, anchor, gaps = max(befores, key=lambda o: min(o[2]))
                col.spec["before"] = name
                anchor_start = anchor.spec.get("start")
                if anchor_start and anchor_start < col.spec["start"]:
                    col.spec["start"] = anchor_start
                relations.append(f"temporal: {table}.{col.name} before {name}")


def _numeric_operand(col: Column) -> bool:
    t = col.spec.get("type")
    return (t in ("int", "float") and not col.is_key) or (t == "lookup" and col.kind in ("int", "float"))


def _detect_formulas(columns: Dict[str, List[Column]], relations: List[str]) -> None:
    """Recognise columns that are an exact arithmetic function of siblings,
    e.g. ``line_total = round(quantity * unit_price, 2)`` or
    ``cost = round(price * 0.6, 2)``. Operands must be *earlier* columns (a
    derived column comes after its inputs, which also picks the right
    direction between price and cost) and are never formulas themselves, so
    the result cannot contain a dependency cycle."""
    for table, cols in columns.items():
        for pos, target in enumerate(cols):
            if target.spec.get("type") not in ("int", "float") or target.is_key:
                continue
            operands = [c for c in cols[:pos] if _numeric_operand(c)]
            if not operands:
                continue
            nd = target.spec.get("round", 0 if target.spec["type"] == "int" else None)
            found = _match_formula(target, operands, nd)
            if found is None:
                continue
            expr = found if nd is None else (f"round({found}, {nd})" if nd else f"int(round({found}))")
            target.spec = {"type": "formula", "expr": expr}
            relations.append(f"formula: {table}.{target.name} = {expr}")


def _match_formula(target: Column, operands: List[Column], nd: Optional[int]) -> Optional[str]:
    rows = [i for i, v in enumerate(target.parsed) if v is not None]
    if len(rows) < MIN_EVIDENCE_ROWS:
        return None
    # Values produced by rounding match exactly; allow half a unit in the last
    # place for data rounded by another tool.
    tol = 0.5 * 10 ** (-nd) if nd is not None else 0.0

    def fits(fn) -> bool:
        checked = 0
        for i in rows:
            try:
                value = fn(i)
            except (TypeError, ZeroDivisionError, OverflowError):
                return False
            if value is None:
                continue
            checked += 1
            y = float(target.parsed[i])
            expected = float(round(value, nd)) if nd is not None else float(value)
            if abs(y - expected) > tol + 1e-9 * max(1.0, abs(y)):
                return False
        return checked >= MIN_EVIDENCE_ROWS

    def val(col: Column, i: int) -> Optional[float]:
        v = col.parsed[i]
        return None if v is None else float(v)

    def both(a, b, op):
        def fn(i):
            x, y = val(a, i), val(b, i)
            return None if x is None or y is None else op(x, y)
        return fn

    for a_idx, a in enumerate(operands):
        for b in operands[a_idx + 1:]:
            for symbol, op in (("*", lambda x, y: x * y), ("+", lambda x, y: x + y)):
                if fits(both(a, b, op)):
                    return f"{a.name} {symbol} {b.name}"
            if fits(both(a, b, lambda x, y: x - y)):
                return f"{a.name} - {b.name}"
            if fits(both(b, a, lambda x, y: x - y)):
                return f"{b.name} - {a.name}"
    for a in operands:
        ratios = sorted(float(target.parsed[i]) / float(a.parsed[i]) for i in rows
                        if a.parsed[i] not in (None, 0) and target.parsed[i] is not None)
        if len(ratios) < MIN_EVIDENCE_ROWS:
            continue
        median = ratios[len(ratios) // 2]
        for digits in (2, 3, 4, 6):
            c = round(median, digits)
            if c not in (0, 1) and fits(lambda i, a=a, c=c: None if val(a, i) is None else val(a, i) * c):
                return f"{a.name} * {c:g}"
    return None


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def strength_from_rho(rho: float) -> float:
    """Invert rho = s / sqrt(s^2 + (1-s)^2): the generator's blend of a rank
    signal (weight s) and uniform noise has that correlation with the driver."""
    r = min(abs(rho), 0.999)
    if abs(2 * r * r - 1) < 1e-9:
        return 0.5
    s = (r * r - r * math.sqrt(1 - r * r)) / (2 * r * r - 1)
    return round(min(max(s, 0.0), 1.0), 3)


def _detect_correlations(columns: Dict[str, List[Column]], relations: List[str]) -> None:
    for table, cols in columns.items():
        numeric = [c for c in cols if c.spec.get("type") in ("int", "float") and not c.is_key]
        for j, target in enumerate(numeric):
            best = None
            for driver in numeric[:j]:
                pairs = [(a, b) for a, b in zip(driver.parsed, target.parsed) if a is not None and b is not None]
                if len(pairs) < MIN_EVIDENCE_ROWS:
                    continue
                rho = _pearson(_ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs]))
                if abs(rho) >= 0.3 and (best is None or abs(rho) > abs(best[1])):
                    best = (driver, rho)
            if best is None:
                continue
            driver, rho = best
            for key in ("distribution", "mean", "std", "scale"):
                target.spec.pop(key, None)
            target.spec["correlate"] = {
                "field": driver.name,
                "strength": strength_from_rho(rho),
                "direction": "positive" if rho > 0 else "negative",
            }
            relations.append(f"correlation: {table}.{target.name} ~ {driver.name} (rank rho {rho:+.2f})")


def _detect_email_names(columns: Dict[str, List[Column]]) -> None:
    """``anna.lopez@...`` next to a ``full_name`` of "Anna Lopez" -> depends_on."""
    import unicodedata

    def slug(name: str) -> str:
        folded = unicodedata.normalize("NFKD", name.lower())
        return ".".join("".join(ch for ch in part if ch.isascii() and ch.isalnum()) for part in folded.split())

    for cols in columns.values():
        names = [c for c in cols if c.spec.get("type") == "name"]
        for col in cols:
            if col.spec.get("type") != "email" or not names:
                continue
            for name_col in names:
                pairs = [(e, nm) for e, nm in zip(col.parsed, name_col.parsed) if e and nm]
                if not pairs:
                    continue
                hits = sum(1 for e, nm in pairs if e.split("@")[0].lower().rstrip("0123456789") == slug(nm))
                if hits / len(pairs) >= 0.8:
                    col.spec["depends_on"] = name_col.name
                    break


def _fix_unique_capacity(columns: Dict[str, List[Column]], rows: Dict[str, int]) -> None:
    for table, cols in columns.items():
        for col in cols:
            spec = col.spec
            if spec.get("unique") and spec.get("type") == "int":
                need = rows[table]
                if spec["max"] - spec["min"] + 1 < need:
                    spec["max"] = spec["min"] + need - 1


# --------------------------------------------------------------------------
# fidelity: real vs synthetic
# --------------------------------------------------------------------------
def fidelity_report(raw: RawData, schema: Schema, synthetic: Any) -> List[Dict[str, Any]]:
    """Per-column comparison of real data (``raw``) and a synthetic dataset
    generated from ``schema``: total variation distance for categories,
    mean/std/range for numbers, range for dates."""
    real = coerce_to_schema(raw, schema)
    out: List[Dict[str, Any]] = []
    for table in schema.tables:
        r_rows = real.tables.get(table.name, [])
        s_rows = synthetic.tables.get(table.name, [])
        for fs in table.fields:
            r_vals = [r.get(fs.name) for r in r_rows if r.get(fs.name) is not None]
            s_vals = [r.get(fs.name) for r in s_rows if r.get(fs.name) is not None]
            entry: Dict[str, Any] = {"table": table.name, "field": fs.name, "type": fs.type}
            if fs.type == "category":
                cats = {str(k) for k in fs.get("categories")}
                bucket = "other" if "other" in cats else None
                r_map = [str(v) if str(v) in cats or bucket is None else bucket for v in r_vals]
                entry["tvd"] = _tvd(r_map, [str(v) for v in s_vals])
            elif fs.type in ("int", "float", "formula", "lookup") and _numeric(r_vals) and _numeric(s_vals):
                entry.update(_moments("real", r_vals))
                entry.update(_moments("synthetic", s_vals))
            elif fs.type in ("date", "datetime") and r_vals and s_vals:
                entry["real_range"] = (min(map(str, r_vals))[:10], max(map(str, r_vals))[:10])
                entry["synthetic_range"] = (min(map(str, s_vals))[:10], max(map(str, s_vals))[:10])
            else:
                continue
            out.append(entry)
    return out


def _numeric(values: Sequence[Any]) -> bool:
    return bool(values) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _tvd(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    labels = set(ca) | set(cb)
    return round(0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in labels), 4)


def _moments(prefix: str, values: Sequence[Any]) -> Dict[str, float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return {}
    mean = sum(nums) / len(nums)
    std = math.sqrt(sum((x - mean) ** 2 for x in nums) / len(nums))
    return {f"{prefix}_mean": mean, f"{prefix}_std": std, f"{prefix}_min": min(nums), f"{prefix}_max": max(nums)}


def render_fidelity(entries: List[Dict[str, Any]]) -> str:
    lines = ["# Fidelity: real vs synthetic", ""]
    current = None
    for e in entries:
        if e["table"] != current:
            current = e["table"]
            lines.append(f"## {current}")
        label = f"  {e['field']:<22} {e['type']:<9}"
        if "tvd" in e:
            lines.append(f"{label} category TVD {e['tvd']:.3f}")
        elif "real_mean" in e:
            lines.append(
                f"{label} mean {e['real_mean']:.4g} vs {e['synthetic_mean']:.4g}, "
                f"std {e['real_std']:.4g} vs {e['synthetic_std']:.4g}, "
                f"range [{e['real_min']:.4g}, {e['real_max']:.4g}] vs "
                f"[{e['synthetic_min']:.4g}, {e['synthetic_max']:.4g}]"
            )
        elif "real_range" in e:
            lines.append(f"{label} range {e['real_range'][0]}..{e['real_range'][1]} vs "
                         f"{e['synthetic_range'][0]}..{e['synthetic_range'][1]}")
    lines.append("")
    lines.append("TVD = total variation distance between category distributions (0 = identical, 1 = disjoint).")
    return "\n".join(lines)


def yaml_header(path: str, kind: str, min_count: int, rows_scale: float) -> str:
    return (
        f"# Inferred by `sdf infer` from {path} ({kind}).\n"
        f"# rows scaled x{rows_scale:g}; category values seen fewer than {min_count} times\n"
        f"# were merged into 'other'. Names, emails, phones and free text are regenerated\n"
        f"# from fictional pools - review this file before sharing data generated from it.\n"
    )


__all__ = [
    "InferenceError",
    "InferenceResult",
    "infer_schema",
    "fidelity_report",
    "render_fidelity",
    "strength_from_rho",
    "topological_table_order",
]
