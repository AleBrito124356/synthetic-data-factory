"""Validate generated tabular data against its schema and report distributions.

Checks performed
----------------
* **types**        — values conform to the declared type and range: numbers
                     inside [min, max], dates inside [start, end], emails
                     syntactically valid *and* on a reserved RFC 2606 domain,
                     phones in the NANP fictional 555-01xx block, IPs in the
                     RFC 5737 documentation ranges.
* **uniqueness**   — fields marked unique (and ids) contain no duplicates.
* **not null**     — ids and foreign keys are never null (except declared
                     ``null_rate`` and the roots of a self-referencing FK).
* **foreign keys** — every FK value exists in the parent column, and a
                     self-referencing FK forms a hierarchy with no cycles.
* **distributions**— categorical proportions are close to requested weights
                     (per ``when`` case group when the field has one).
* **relations**    — ``temporal_order`` (``after``/``before`` hold row by
                     row), ``lookup_consistency`` (a lookup equals the
                     parent row's column), ``when_bounds`` (each case's range
                     or label set), ``parent_coverage`` (``min_per_parent``).

The distribution report also carries an explicit note that all values are
synthetic and must never be presented as real records.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .providers import IPV4_DOC_NETWORKS, is_reserved_email
from .schema import (
    DEFAULT_DATE_END,
    DEFAULT_DATE_START,
    FieldSpec,
    Schema,
    TableSpec,
    effective_field,
    parse_date,
)

_FICTIONAL_PHONE = re.compile(r"^\+1-\d{3}-555-01\d{2}$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

SYNTHETIC_NOTE = (
    "All values in this dataset are SYNTHETIC and generated for testing, demos, "
    "or model training. They do not describe real people, accounts, or events "
    "and must never be presented as real records."
)


@dataclass
class Check:
    name: str
    table: str
    field: str
    passed: bool
    detail: str = ""

    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class ValidationReport:
    checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def render(self) -> str:
        lines = ["# Validation report", ""]
        passed = sum(1 for c in self.checks if c.passed)
        lines.append(f"{passed}/{len(self.checks)} checks passed "
                     f"({'OK' if self.ok else 'FAILURES PRESENT'}).")
        lines.append("")
        for c in self.checks:
            mark = "[PASS]" if c.passed else "[FAIL]"
            loc = f"{c.table}.{c.field}" if c.field else c.table
            lines.append(f"{mark} {c.name} :: {loc}")
            if c.detail:
                lines.append(f"        {c.detail}")
        lines.append("")
        lines.append(SYNTHETIC_NOTE)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def validate_dataset(dataset: "Any", tolerance: float = 0.05) -> ValidationReport:
    """Validate a :class:`~factory.tabular.Dataset`.

    ``tolerance`` is the maximum allowed absolute deviation between a
    categorical's observed proportion and its requested weight.
    """
    schema: Schema = dataset.schema
    report = ValidationReport()

    ctx = _Context(schema, dataset)
    for table in schema.tables:
        rows = dataset.tables.get(table.name, [])
        _check_row_count(report, table, rows)
        for fs in table.fields:
            values = [r.get(fs.name) for r in rows]
            _check_type(report, table, fs, values)
            if fs.unique or fs.type == "id":
                _check_unique(report, table, fs, values)
            if fs.type in ("id", "foreign_key"):
                _check_not_null(report, table, fs, values)
            if fs.type == "category":
                if fs.get("when") is not None:
                    _check_when_distribution(report, table, fs, rows, tolerance)
                else:
                    _check_distribution(report, table, fs, values, tolerance)
            if fs.type == "foreign_key":
                _check_foreign_key(report, schema, dataset, table, fs, values)
                ref = fs.references
                if ref and ref[0] == table.name:
                    _check_hierarchy(report, table, fs, rows)
                if fs.get("min_per_parent") is not None:
                    _check_parent_coverage(report, ctx, table, fs, values)
            if fs.type == "lookup":
                _check_lookup(report, ctx, table, fs, rows)
            if fs.get("when") is not None:
                _check_when_bounds(report, table, fs, rows)
            if fs.anchors:
                _check_temporal_order(report, ctx, table, fs, rows)
    return report


class _Context:
    """Lazily built {key -> parent row} indexes shared by relational checks."""

    def __init__(self, schema: Schema, dataset: "Any"):
        self.schema = schema
        self.dataset = dataset
        self._indexes: Dict[tuple, Dict[Any, Dict[str, Any]]] = {}

    def parent_row(self, table_name: str, key_col: str, key: Any) -> Optional[Dict[str, Any]]:
        idx = self._indexes.get((table_name, key_col))
        if idx is None:
            idx = {}
            for r in self.dataset.tables.get(table_name, []):
                idx.setdefault(_hashable(r.get(key_col)), r)
            self._indexes[(table_name, key_col)] = idx
        return idx.get(_hashable(key))

    def via_value(self, table: TableSpec, row: Dict[str, Any], fk_name: str, column: str) -> Any:
        fk = table.get_field(fk_name)
        parent_name, key_col = fk.references
        key = row.get(fk_name)
        if key is None:
            return None
        parent = self.parent_row(parent_name, key_col, key)
        return None if parent is None else parent.get(column)

    def anchor_value(self, table: TableSpec, row: Dict[str, Any], anchor: str) -> Any:
        if table.get_field(anchor) is not None:
            return row.get(anchor)
        fk_name, column = anchor.split(".", 1)
        return self.via_value(table, row, fk_name, column)


def _to_datetime(value: Any) -> Optional[_dt.datetime]:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime.combine(value, _dt.time.min)
    try:
        return _dt.datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _is_date_only(value: Any) -> bool:
    if isinstance(value, _dt.datetime):
        return False
    return isinstance(value, _dt.date) or (isinstance(value, str) and len(value.strip()) == 10)


def _check_temporal_order(report: ValidationReport, ctx: _Context, table: TableSpec, fs: FieldSpec,
                          rows: List[Dict[str, Any]]) -> None:
    """``after`` (with min_days/max_days) and ``before`` hold on every row.
    Comparisons happen at day granularity when either side is a date."""
    min_d = float(fs.get("min_days", 0) or 0)
    max_d = fs.get("max_days")
    for kind, anchor in fs.anchors:
        bad = 0
        example = None
        checked = 0
        for r in rows:
            value, ref = r.get(fs.name), ctx.anchor_value(table, r, anchor)
            v_dt, a_dt = _to_datetime(value), _to_datetime(ref)
            if v_dt is None or a_dt is None:
                continue
            checked += 1
            coarse = _is_date_only(value) or _is_date_only(ref)
            if kind == "after":
                lo = a_dt + _dt.timedelta(days=min_d)
                hi = a_dt + _dt.timedelta(days=float(max_d)) if max_d is not None else None
                if coarse:
                    ok = v_dt.date() >= lo.date() and (hi is None or v_dt.date() <= hi.date())
                else:
                    ok = v_dt >= lo and (hi is None or v_dt <= hi)
            else:
                ok = v_dt.date() <= a_dt.date() if coarse else v_dt <= a_dt
            if not ok:
                bad += 1
                if example is None:
                    example = (value, ref)
        gap = ""
        if kind == "after" and (min_d or max_d is not None):
            gap = f" (+{min_d:g}..{'' if max_d is None else f'{float(max_d):g}'} days)"
        detail = f"{kind} {anchor}{gap}: {checked} row(s) checked"
        if bad:
            detail += f"; {bad} violate it, e.g. {example[0]!r} vs {example[1]!r}"
        report.add(Check("temporal_order", table.name, fs.name, bad == 0, detail))


def _check_lookup(report: ValidationReport, ctx: _Context, table: TableSpec, fs: FieldSpec,
                  rows: List[Dict[str, Any]]) -> None:
    via, column = str(fs.get("via")), str(fs.get("column"))
    bad = 0
    example = None
    for r in rows:
        if r.get(via) is None:
            continue
        expected = ctx.via_value(table, r, via, column)
        if not _same_value(r.get(fs.name), expected):
            bad += 1
            if example is None:
                example = (r.get(fs.name), expected)
    detail = f"{fs.name} == {via} -> {column}"
    if bad:
        detail += f"; {bad} mismatch(es), e.g. {example[0]!r} != {example[1]!r}"
    report.add(Check("lookup_consistency", table.name, fs.name, bad == 0, detail))


def _same_value(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    return str(a) == str(b)


def _check_when_bounds(report: ValidationReport, table: TableSpec, fs: FieldSpec,
                       rows: List[Dict[str, Any]]) -> None:
    key_field = str(fs.get("when")["field"])
    per_case: Dict[str, List[int]] = {}
    example = None
    for r in rows:
        v = r.get(fs.name)
        if v is None:
            continue
        eff = effective_field(fs, r.get(key_field))
        case = str(r.get(key_field)) if eff is not fs else "(base)"
        stats = per_case.setdefault(case, [0, 0])
        stats[0] += 1
        if fs.type in ("int", "float"):
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and _range_ok(eff, v)
        elif fs.type == "category":
            ok = str(v) in {str(k) for k in (eff.get("categories") or {})}
        else:
            ok = True
        if not ok:
            stats[1] += 1
            if example is None:
                example = (case, v)
    bad = sum(b for _, b in per_case.values())
    detail = ", ".join(f"{c}: {n - b}/{n} ok" for c, (n, b) in per_case.items())
    if bad:
        detail += f"; e.g. {example[1]!r} for {key_field}={example[0]}"
    report.add(Check("when_bounds", table.name, fs.name, bad == 0, detail))


def _check_when_distribution(report: ValidationReport, table: TableSpec, fs: FieldSpec,
                             rows: List[Dict[str, Any]], tolerance: float) -> None:
    """Weighted proportions checked inside each ``when`` case group. Groups
    too small to measure a proportion within ``tolerance`` are skipped."""
    key_field = str(fs.get("when")["field"])
    groups: Dict[str, List[Any]] = {}
    specs: Dict[str, FieldSpec] = {}
    for r in rows:
        eff = effective_field(fs, r.get(key_field))
        case = str(r.get(key_field)) if eff is not fs else "(base)"
        groups.setdefault(case, []).append(r.get(fs.name))
        specs[case] = eff
    worst, where = 0.0, ""
    skipped = 0
    min_size = int(math.ceil(1.0 / max(tolerance, 1e-6)))
    for case, values in groups.items():
        non_null = [v for v in values if v is not None]
        if len(non_null) < min_size:
            skipped += 1
            continue
        cats = specs[case].get("categories") or {}
        total_w = sum(cats.values()) or 1.0
        for label, weight in cats.items():
            dev = abs(sum(1 for v in non_null if str(v) == str(label)) / len(non_null) - weight / total_w)
            if dev > worst:
                worst, where = dev, f"'{label}' when {key_field}={case}"
    detail = f"max deviation {worst:.3f}" + (f" at {where}" if where else "") + f" (tolerance {tolerance:.3f})"
    if skipped:
        detail += f"; {skipped} small group(s) not measured"
    report.add(Check("distribution", table.name, fs.name, worst <= tolerance, detail))


def _check_parent_coverage(report: ValidationReport, ctx: _Context, table: TableSpec, fs: FieldSpec,
                           values: List[Any]) -> None:
    need = int(fs.get("min_per_parent") or 0)
    parent_name, key_col = fs.references
    counts: Dict[Any, int] = {}
    for v in values:
        if v is not None:
            counts[_hashable(v)] = counts.get(_hashable(v), 0) + 1
    parents = [_hashable(r.get(key_col)) for r in ctx.dataset.tables.get(parent_name, [])]
    short = [p for p in parents if counts.get(p, 0) < need]
    detail = f"every {parent_name} row has >= {need} {table.name} row(s)"
    if short:
        detail += f"; {len(short)} parent(s) below, e.g. {short[0]!r} has {counts.get(short[0], 0)}"
    report.add(Check("parent_coverage", table.name, fs.name, not short, detail))


def _check_row_count(report: ValidationReport, table: TableSpec, rows: List[Dict[str, Any]]) -> None:
    passed = len(rows) == table.rows
    report.add(
        Check(
            name="row_count",
            table=table.name,
            field="",
            passed=passed,
            detail=f"expected {table.rows}, found {len(rows)}",
        )
    )


def _check_type(report: ValidationReport, table: TableSpec, fs: FieldSpec, values: List[Any]) -> None:
    non_null = [v for v in values if v is not None]
    spec = fs
    if fs.get("when") is not None:
        # Ranges differ per case; when_bounds checks them. Here: type only.
        spec = FieldSpec(fs.name, fs.type, {k: v for k, v in fs.params.items()
                                            if k not in ("min", "max", "when")})
    bad: List[Any] = []
    for v in non_null:
        if not _type_ok(spec, v):
            bad.append(v)
    passed = not bad
    detail = "" if passed else f"{len(bad)} value(s) fail type '{fs.type}', e.g. {bad[0]!r}"
    report.add(Check(name="type", table=table.name, field=fs.name, passed=passed, detail=detail))


def _check_not_null(report: ValidationReport, table: TableSpec, fs: FieldSpec, values: List[Any]) -> None:
    nulls = sum(1 for v in values if v is None)
    ref = fs.references
    if fs.type == "foreign_key" and ref and ref[0] == table.name:
        # Roots of a hierarchy are null by design; at least one must exist.
        passed = nulls >= 1 or not values
        detail = "" if passed else "a self-referencing key needs at least one null root"
        report.add(Check("not_null", table.name, fs.name, passed, detail))
        return
    allowed = fs.type == "foreign_key" and fs.null_rate > 0
    passed = nulls == 0 or allowed
    detail = "" if passed else f"{nulls} null value(s) in a {fs.type} column"
    report.add(Check("not_null", table.name, fs.name, passed, detail))


def _check_hierarchy(
    report: ValidationReport, table: TableSpec, fs: FieldSpec, rows: List[Dict[str, Any]]
) -> None:
    """A self-referencing FK must form a forest: following parent pointers
    from any row must reach a null root without revisiting a row."""
    _, ref_col = fs.references
    parent_of: Dict[Any, Any] = {}
    for r in rows:
        parent_of[_hashable(r.get(ref_col))] = r.get(fs.name)
    cycles = 0
    example = None
    state: Dict[Any, int] = {}  # 1 = on the current path, 2 = proven acyclic
    for start in parent_of:
        path = []
        node: Any = start
        while node is not None and _hashable(node) in parent_of:
            key = _hashable(node)
            if state.get(key) == 2:
                break
            if state.get(key) == 1:
                cycles += 1
                if example is None:
                    example = node
                break
            state[key] = 1
            path.append(key)
            node = parent_of[key]
        for key in path:
            state[key] = 2
    passed = cycles == 0
    detail = "" if passed else f"{cycles} cycle(s) in the hierarchy, e.g. through {example!r}"
    report.add(Check("hierarchy_acyclic", table.name, fs.name, passed, detail))


def _check_unique(report: ValidationReport, table: TableSpec, fs: FieldSpec, values: List[Any]) -> None:
    non_null = [v for v in values if v is not None]
    seen: set = set()
    dupes = 0
    for v in non_null:
        key = _hashable(v)
        if key in seen:
            dupes += 1
        seen.add(key)
    passed = dupes == 0
    detail = "" if passed else f"{dupes} duplicate value(s)"
    report.add(Check(name="uniqueness", table=table.name, field=fs.name, passed=passed, detail=detail))


def _check_distribution(
    report: ValidationReport, table: TableSpec, fs: FieldSpec, values: List[Any], tolerance: float
) -> None:
    cats: Dict[str, float] = fs.get("categories")
    non_null = [v for v in values if v is not None]
    total = len(non_null)
    if total == 0:
        report.add(Check("distribution", table.name, fs.name, True, "no non-null values"))
        return
    if fs.unique:
        # Each category appears at most once, so weights cannot apply; only
        # check that every value is a declared category.
        allowed = {str(k) for k in cats}
        unexpected = [v for v in non_null if str(v) not in allowed]
        passed = not unexpected
        detail = "unique: each category at most once" + (
            f"; {len(unexpected)} unexpected value(s), e.g. {unexpected[0]!r}" if unexpected else ""
        )
        report.add(Check("distribution", table.name, fs.name, passed, detail))
        return
    total_weight = sum(cats.values())
    observed = {str(k): 0 for k in cats}
    unexpected = 0
    for v in non_null:
        sv = str(v)
        if sv in observed:
            observed[sv] += 1
        else:
            unexpected += 1

    worst = -1.0
    worst_label = next(iter(cats))
    for label, weight in cats.items():
        expected_p = weight / total_weight
        observed_p = observed[str(label)] / total
        dev = abs(observed_p - expected_p)
        if dev > worst:
            worst = dev
            worst_label = str(label)

    # With n rows a proportion moves in steps of 1/n, so a tiny table cannot
    # hit a weight more closely than that; never demand the impossible.
    effective = max(tolerance, 1.0 / total)
    passed = worst <= effective + 1e-12 and unexpected == 0
    detail = (
        f"max deviation {worst:.3f} at '{worst_label}' (tolerance {effective:.3f})"
        + (f"; {unexpected} unexpected value(s)" if unexpected else "")
    )
    report.add(Check("distribution", table.name, fs.name, passed, detail))


def _check_foreign_key(
    report: ValidationReport,
    schema: Schema,
    dataset: "Any",
    table: TableSpec,
    fs: FieldSpec,
    values: List[Any],
) -> None:
    ref = fs.get("references")
    ref_table, ref_col = ref.split(".", 1)
    parent_rows = dataset.tables.get(ref_table, [])
    parent_values = {_hashable(r.get(ref_col)) for r in parent_rows}
    orphans = 0
    example: Optional[Any] = None
    for v in values:
        if v is None:
            continue
        if _hashable(v) not in parent_values:
            orphans += 1
            if example is None:
                example = v
    passed = orphans == 0
    detail = "" if passed else f"{orphans} orphan value(s) not found in {ref}, e.g. {example!r}"
    report.add(Check("foreign_key", table.name, fs.name, passed, detail))


# --------------------------------------------------------------------------
# distribution report
# --------------------------------------------------------------------------
def distribution_report(dataset: "Any") -> Dict[str, Any]:
    """Per-field summary statistics for the whole dataset."""
    schema: Schema = dataset.schema
    out: Dict[str, Any] = {"note": SYNTHETIC_NOTE, "tables": {}}
    for table in schema.tables:
        rows = dataset.tables.get(table.name, [])
        fields_summary: Dict[str, Any] = {}
        for fs in table.fields:
            values = [r.get(fs.name) for r in rows]
            summary = _summarize_field(fs, values)
            when = fs.get("when")
            if isinstance(when, dict) and fs.type in ("int", "float", "category"):
                key = str(when["field"])
                groups: Dict[str, List[Any]] = {}
                for r in rows:
                    groups.setdefault(str(r.get(key)), []).append(r.get(fs.name))
                summary["when_field"] = key
                summary["cases"] = {
                    case: _summarize_field(effective_field(fs, case), vals)
                    for case, vals in groups.items()
                }
            fields_summary[fs.name] = summary
        out["tables"][table.name] = {"rows": len(rows), "fields": fields_summary}
    return out


def render_distribution_report(report: Dict[str, Any]) -> str:
    lines = ["# Distribution report", ""]
    for table_name, tinfo in report["tables"].items():
        lines.append(f"## {table_name}  ({tinfo['rows']} rows)")
        for fname, summary in tinfo["fields"].items():
            lines.append(f"- {fname} [{summary['type']}] nulls={summary['nulls']}")
            lines.extend(_render_summary(summary, "    "))
            for case, case_summary in (summary.get("cases") or {}).items():
                lines.append(f"    when {summary['when_field']}={case}:")
                lines.extend(_render_summary(case_summary, "      "))
        lines.append("")
    lines.append(report["note"])
    return "\n".join(lines)


def _render_summary(summary: Dict[str, Any], pad: str) -> List[str]:
    if summary["type"] in ("int", "float"):
        if not summary.get("count") or "min" not in summary:
            return [f"{pad}no numeric values"]
        return [
            f"{pad}min={summary['min']:.3g} max={summary['max']:.3g} "
            f"mean={summary['mean']:.3g} std={summary['std']:.3g}"
        ]
    if summary["type"] == "category":
        parts = ", ".join(
            f"{k}={v['observed']:.2f}/{v['expected']:.2f}" for k, v in summary["categories"].items()
        )
        return [f"{pad}observed/expected: {parts}"]
    return [f"{pad}distinct={summary['distinct']} sample={summary.get('sample')!r}"]


def _summarize_field(fs: FieldSpec, values: List[Any]) -> Dict[str, Any]:
    non_null = [v for v in values if v is not None]
    nulls = len(values) - len(non_null)
    base: Dict[str, Any] = {"type": fs.type, "nulls": nulls, "count": len(non_null)}

    if fs.type in ("int", "float") and non_null:
        nums = [float(v) for v in non_null if isinstance(v, (int, float))]
        if nums:
            mean = sum(nums) / len(nums)
            var = sum((x - mean) ** 2 for x in nums) / len(nums)
            base.update(min=min(nums), max=max(nums), mean=mean, std=math.sqrt(var))
    elif fs.type == "category":
        cats: Dict[str, float] = fs.get("categories") or {}
        total_weight = sum(cats.values()) or 1.0
        total = len(non_null) or 1
        counts: Dict[str, int] = {str(k): 0 for k in cats}
        for v in non_null:
            counts[str(v)] = counts.get(str(v), 0) + 1
        base["categories"] = {
            label: {
                "count": counts.get(str(label), 0),
                "observed": counts.get(str(label), 0) / total,
                "expected": weight / total_weight,
            }
            for label, weight in cats.items()
        }
    else:
        distinct = len({_hashable(v) for v in non_null})
        base["distinct"] = distinct
        base["sample"] = non_null[0] if non_null else None
    return base


# --------------------------------------------------------------------------
# type checking helpers
# --------------------------------------------------------------------------
def _type_ok(fs: FieldSpec, v: Any) -> bool:
    t = fs.type
    if t == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            return False
        return _range_ok(fs, v)
    if t == "float":
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        return _range_ok(fs, v)
    if t == "bool":
        return isinstance(v, bool)
    if t == "email":
        return is_reserved_email(v)
    if t == "phone":
        return isinstance(v, str) and bool(_FICTIONAL_PHONE.match(v))
    if t == "ipv4":
        if not isinstance(v, str) or v.count(".") != 3:
            return False
        net, _, host = v.rpartition(".")
        return net in IPV4_DOC_NETWORKS and host.isdigit() and 1 <= int(host) <= 254
    if t in ("date", "datetime"):
        return _is_isoformat(v, t) and _date_in_bounds(fs, v)
    if t == "id":
        if fs.is_sequential_int_id:
            return isinstance(v, int) and not isinstance(v, bool)
        return isinstance(v, str)
    if t == "uuid":
        return isinstance(v, str) and bool(_UUID_RE.match(v))
    if t in ("foreign_key", "formula", "lookup"):
        return True  # value shape depends on referenced/computed types
    return isinstance(v, (str, int, float, bool))


def _date_in_bounds(fs: FieldSpec, v: str) -> bool:
    if fs.anchors:
        # Anchored fields only honour explicit bounds; the default window
        # does not apply (temporal_order checks the anchors).
        start = parse_date(fs.get("start")) if "start" in fs.params else None
        end = parse_date(fs.get("end")) if "end" in fs.params else None
    else:
        start = parse_date(fs.get("start", DEFAULT_DATE_START))
        end = parse_date(fs.get("end", DEFAULT_DATE_END))
    d = parse_date(v[:10])
    return (start is None or start <= d) and (end is None or d <= end)


def _range_ok(fs: FieldSpec, v: Any) -> bool:
    lo = fs.get("min")
    hi = fs.get("max")
    if lo is not None and v < lo:
        return False
    if hi is not None and v > hi:
        return False
    return True


def _is_isoformat(v: Any, t: str) -> bool:
    if not isinstance(v, str):
        return False
    try:
        if t == "date":
            _dt.date.fromisoformat(v)
        else:
            _dt.datetime.fromisoformat(v)
        return True
    except ValueError:
        return False


def _hashable(v: Any) -> Any:
    try:
        hash(v)
        return v
    except TypeError:
        return str(v)


# --------------------------------------------------------------------------
# text / qa record validation
# --------------------------------------------------------------------------
_TEXT_REQUIRED = {
    "paraphrase": ("text", "intent"),
    "classification": ("text", "label"),
    "personas": ("text", "context"),
    "reviews": ("text", "sentiment", "rating", "product"),
    "tickets": ("subject", "body", "text", "category"),
}
_RATING_RANGES = {"positive": (4, 5), "neutral": (3, 3), "negative": (1, 2)}


def validate_records(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    kind: str = "text",
    max_duplicate_rate: Optional[float] = None,
) -> ValidationReport:
    """Validate generated text or Q&A records against the task that asked for them.

    Text tasks: required fields present and non-empty, labels inside the
    configured set, every group exactly at its quota, duplicate rate (exact,
    case/whitespace-insensitive, within a group) at most ``max_duplicate_rate``
    (default 0), and for reviews a rating consistent with the sentiment.

    Q&A: required fields, no ``NOT_IN_PASSAGE`` answers, every pair scored at
    least ``min_quality`` on all three axes, a hard negative that differs from
    the answer (when enabled), and no duplicate questions.
    """
    if max_duplicate_rate is None:
        max_duplicate_rate = float(config.get("max_duplicate_rate", 0.0))
    report = ValidationReport()
    task = "qa" if kind == "qa" else str(config.get("task", "")).lower()
    report.add(Check("record_count", task, "", bool(records), f"{len(records)} record(s)"))
    if kind == "qa":
        _validate_qa_records(report, records, config, max_duplicate_rate)
        return report

    from .text import text_task_quotas

    shape = text_task_quotas(config)
    group_key, quotas = shape["group_key"], {str(k): v for k, v in shape["quotas"].items()}
    _check_required(report, task, records, _TEXT_REQUIRED.get(task, ("text",)))

    labels = [str(r.get(group_key)) for r in records]
    unexpected = sorted({l for l in labels if l not in quotas})
    report.add(Check(
        "label_set", task, group_key, not unexpected,
        f"allowed: {', '.join(quotas)}" + (f"; unexpected: {', '.join(unexpected)}" if unexpected else ""),
    ))

    counts: Dict[str, int] = {g: 0 for g in quotas}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    off = {g: (counts.get(g, 0), q) for g, q in quotas.items() if counts.get(g, 0) != q}
    detail = ", ".join(f"{g}={counts.get(g, 0)}/{q}" for g, q in quotas.items())
    report.add(Check("quota", task, group_key, not off, detail))

    dupes = 0
    seen: set = set()
    for r in records:
        key = (str(r.get(group_key)), _norm_text(r.get("text", "")))
        if key in seen:
            dupes += 1
        seen.add(key)
    rate = dupes / len(records) if records else 0.0
    report.add(Check(
        "duplicates", task, "text", rate <= max_duplicate_rate,
        f"{dupes} duplicate(s), rate {rate:.3f} (max {max_duplicate_rate:.3f})",
    ))

    if task == "reviews":
        bad = [r for r in records
               if str(r.get("sentiment")) in _RATING_RANGES
               and not (_RATING_RANGES[str(r.get("sentiment"))][0]
                        <= _as_int(r.get("rating")) <= _RATING_RANGES[str(r.get("sentiment"))][1])]
        report.add(Check("rating_matches_sentiment", task, "rating", not bad,
                         f"{len(bad)} inconsistent rating(s)" if bad else ""))
    return report


def _validate_qa_records(report: ValidationReport, records: List[Dict[str, Any]],
                         config: Dict[str, Any], max_duplicate_rate: float) -> None:
    required = ["question", "answer", "context", "source", "quality"]
    hard_negatives = bool(config.get("hard_negatives", True))
    if hard_negatives:
        required.append("hard_negative")
    _check_required(report, "qa", records, required)

    unanswerable = [r for r in records if "NOT_IN_PASSAGE" in str(r.get("answer", "")).upper()]
    report.add(Check("answerable", "qa", "answer", not unanswerable,
                     f"{len(unanswerable)} NOT_IN_PASSAGE answer(s)" if unanswerable else ""))

    min_quality = int(config.get("min_quality", 4))
    low = []
    for r in records:
        q = r.get("quality")
        axes = [q.get(a, 0) for a in ("groundedness", "answerability", "clarity")] if isinstance(q, dict) else [0]
        if min(axes) < min_quality:
            low.append(r)
    report.add(Check("quality_min", "qa", "quality", not low,
                     f"min quality {min_quality}" + (f"; {len(low)} pair(s) below" if low else "")))

    if hard_negatives:
        same = [r for r in records if _norm_text(r.get("hard_negative", "")) == _norm_text(r.get("answer", ""))]
        report.add(Check("hard_negative_differs", "qa", "hard_negative", not same,
                         f"{len(same)} hard negative(s) identical to the answer" if same else ""))

    questions = [_norm_text(r.get("question", "")) for r in records]
    dupes = len(questions) - len(set(questions))
    rate = dupes / len(records) if records else 0.0
    report.add(Check("duplicates", "qa", "question", rate <= max_duplicate_rate,
                     f"{dupes} duplicate question(s), rate {rate:.3f} (max {max_duplicate_rate:.3f})"))


def _check_required(report: ValidationReport, task: str, records: List[Dict[str, Any]],
                    required: Any) -> None:
    missing: Dict[str, int] = {}
    for r in records:
        for key in required:
            value = r.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing[key] = missing.get(key, 0) + 1
    detail = ", ".join(f"{k} missing/empty in {n}" for k, n in missing.items())
    report.add(Check("required_fields", task, "", not missing, detail or f"fields: {', '.join(required)}"))


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
