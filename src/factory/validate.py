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
* **distributions**— categorical proportions are close to requested weights.

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
from .schema import DEFAULT_DATE_END, DEFAULT_DATE_START, FieldSpec, Schema, TableSpec, parse_date

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
                _check_distribution(report, table, fs, values, tolerance)
            if fs.type == "foreign_key":
                _check_foreign_key(report, schema, dataset, table, fs, values)
                ref = fs.references
                if ref and ref[0] == table.name:
                    _check_hierarchy(report, table, fs, rows)
    return report


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
    bad: List[Any] = []
    for v in non_null:
        if not _type_ok(fs, v):
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

    passed = worst <= tolerance and unexpected == 0
    detail = (
        f"max deviation {worst:.3f} at '{worst_label}' (tolerance {tolerance:.3f})"
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
            fields_summary[fs.name] = _summarize_field(fs, values)
        out["tables"][table.name] = {"rows": len(rows), "fields": fields_summary}
    return out


def render_distribution_report(report: Dict[str, Any]) -> str:
    lines = ["# Distribution report", ""]
    for table_name, tinfo in report["tables"].items():
        lines.append(f"## {table_name}  ({tinfo['rows']} rows)")
        for fname, summary in tinfo["fields"].items():
            lines.append(f"- {fname} [{summary['type']}] nulls={summary['nulls']}")
            if summary["type"] in ("int", "float") and summary.get("count"):
                lines.append(
                    f"    min={summary['min']:.3g} max={summary['max']:.3g} "
                    f"mean={summary['mean']:.3g} std={summary['std']:.3g}"
                )
            elif summary["type"] == "category":
                parts = ", ".join(
                    f"{k}={v['observed']:.2f}/{v['expected']:.2f}"
                    for k, v in summary["categories"].items()
                )
                lines.append(f"    observed/expected: {parts}")
            else:
                lines.append(f"    distinct={summary['distinct']} sample={summary.get('sample')!r}")
        lines.append("")
    lines.append(report["note"])
    return "\n".join(lines)


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
    if t in ("foreign_key", "formula"):
        return True  # value shape depends on referenced/computed types
    return isinstance(v, (str, int, float, bool))


def _date_in_bounds(fs: FieldSpec, v: str) -> bool:
    start = parse_date(fs.get("start", DEFAULT_DATE_START))
    end = parse_date(fs.get("end", DEFAULT_DATE_END))
    return start <= parse_date(v[:10]) <= end


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
