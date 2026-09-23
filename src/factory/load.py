"""Load tabular data from disk: a folder of CSV/JSONL files or a SQLite file.

Two uses:

* :func:`validate_data` checks files that already exist (an export from this
  tool, or anything else) against a schema — the same checks as
  ``--validate``, plus "are all tables and columns there".
* :mod:`factory.infer` profiles *real* data to build a schema from it.

CSV cells are strings, so :func:`coerce_to_schema` converts every value to
the type its schema field expects (``"42"`` -> ``42`` for an int id,
``"True"`` -> ``True`` for a bool, a foreign key like the column it
references). Values that do not convert are kept as-is so the type check can
report them.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import FieldSpec, Schema, TableSpec
from .tabular import Dataset

SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
_INT_RE = re.compile(r"^[+-]?\d+$")


class DataLoadError(ValueError):
    """The path cannot be read as a CSV/JSONL folder or a SQLite file."""


@dataclass
class RawTable:
    name: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    source: str = ""


@dataclass
class RawData:
    """Tables as read from disk, before any schema is applied."""

    tables: Dict[str, RawTable] = field(default_factory=dict)
    kind: str = ""  # "csv", "jsonl", "sqlite"
    path: str = ""

    def table_names(self) -> List[str]:
        return list(self.tables)


def load_raw(path: str) -> RawData:
    """Read every table under ``path`` (a folder of ``*.csv``/``*.jsonl``, or
    a ``.db``/``.sqlite`` file). CSV empty cells become ``None``."""
    if os.path.isdir(path):
        return _load_folder(path)
    if os.path.isfile(path) and path.lower().endswith(SQLITE_SUFFIXES):
        return _load_sqlite(path)
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        name = os.path.splitext(os.path.basename(path))[0]
        table = _read_csv(path, name)
        return RawData(tables={name: table}, kind="csv", path=path)
    raise DataLoadError(
        f"{path}: expected a folder of .csv/.jsonl files or a SQLite file "
        f"({', '.join(SQLITE_SUFFIXES)})."
    )


def _load_folder(path: str) -> RawData:
    files = sorted(os.listdir(path))
    data = RawData(kind="", path=path)
    kinds = set()
    for fname in files:
        stem, ext = os.path.splitext(fname)
        full = os.path.join(path, fname)
        if ext.lower() == ".csv":
            data.tables[stem] = _read_csv(full, stem)
            kinds.add("csv")
    for fname in files:
        stem, ext = os.path.splitext(fname)
        if ext.lower() == ".jsonl" and stem not in data.tables and not stem.endswith(".chat"):
            data.tables[stem] = _read_jsonl(os.path.join(path, fname), stem)
            kinds.add("jsonl")
    if not data.tables:
        raise DataLoadError(f"{path}: no .csv or .jsonl files found.")
    data.kind = "+".join(sorted(kinds))
    return data


def _read_csv(path: str, name: str) -> RawTable:
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = [{c: (v if v != "" else None) for c, v in row.items() if c is not None} for row in reader]
    return RawTable(name=name, columns=columns, rows=rows, source=path)


def _read_jsonl(path: str, name: str) -> RawTable:
    rows: List[Dict[str, Any]] = []
    columns: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataLoadError(f"{path}:{lineno}: invalid JSON ({exc}).") from exc
            if not isinstance(obj, dict):
                raise DataLoadError(f"{path}:{lineno}: each line must be a JSON object.")
            for key in obj:
                if key not in columns:
                    columns.append(key)
            rows.append(obj)
    return RawTable(name=name, columns=columns, rows=rows, source=path)


def _load_sqlite(path: str) -> RawData:
    uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise DataLoadError(f"{path}: cannot open SQLite file ({exc}).") from exc
    data = RawData(kind="sqlite", path=path)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        for name in names:
            cur = conn.execute(f'SELECT * FROM "{name}"')
            columns = [d[0] for d in cur.description]
            rows = [dict(zip(columns, values)) for values in cur.fetchall()]
            data.tables[name] = RawTable(name=name, columns=columns, rows=rows, source=f"{path}:{name}")
    except sqlite3.DatabaseError as exc:
        raise DataLoadError(f"{path}: not a readable SQLite database ({exc}).") from exc
    finally:
        conn.close()
    if not data.tables:
        raise DataLoadError(f"{path}: the database has no tables.")
    return data


# --------------------------------------------------------------------------
# coercion to a schema
# --------------------------------------------------------------------------
def coerce_to_schema(raw: RawData, schema: Schema) -> Dataset:
    """Build a :class:`Dataset` from raw tables, converting each value to the
    Python type its schema field produces. Tables missing on disk are empty;
    columns missing in a table are ``None`` (and flagged by :func:`validate_data`)."""
    tables: Dict[str, List[Dict[str, Any]]] = {}
    for table in schema.tables:
        raw_table = raw.tables.get(table.name)
        rows = raw_table.rows if raw_table else []
        converters = {fs.name: _converter(schema, table, fs) for fs in table.fields}
        tables[table.name] = [
            {name: conv(row.get(name)) for name, conv in converters.items()} for row in rows
        ]
    return Dataset(schema=schema, tables=tables)


def _converter(schema: Schema, table: TableSpec, fs: FieldSpec, depth: int = 0):
    t = fs.type
    if t == "foreign_key" and depth < 20:
        ref_table, ref_col = fs.references
        parent = schema.get_table(ref_table)
        return _converter(schema, parent, parent.get_field(ref_col), depth + 1)
    if t == "lookup" and depth < 20:
        via = table.get_field(str(fs.get("via")))
        parent = schema.get_table(via.references[0])
        return _converter(schema, parent, parent.get_field(str(fs.get("column"))), depth + 1)
    if t == "int" or (t == "id" and fs.is_sequential_int_id):
        return _to_int
    if t == "float":
        return _to_float
    if t == "bool":
        return _to_bool
    if t == "category":
        keys = {str(k): k for k in (fs.get("categories") or {})}
        return lambda v: v if v is None else keys.get(str(v), v)
    if t == "formula":
        return _to_scalar
    return lambda v: v if v is None or isinstance(v, str) else str(v)


def _to_int(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and _INT_RE.match(v.strip()):
        return int(v.strip())
    return v


def _to_float(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except ValueError:
        return v


def _to_bool(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in ("true", "1", "yes", "t", "y"):
        return True
    if text in ("false", "0", "no", "f", "n"):
        return False
    return v


def _to_scalar(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    text = v.strip()
    if _INT_RE.match(text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        pass
    if text in ("True", "False"):
        return text == "True"
    return v


# --------------------------------------------------------------------------
# validation of files on disk
# --------------------------------------------------------------------------
def validate_data(schema: Schema, path: str, tolerance: float = 0.05):
    """Validate the tables stored at ``path`` against ``schema``.

    Adds a ``columns`` check per table (missing tables or columns fail;
    extra columns are reported) in front of the regular dataset checks.
    """
    from .validate import Check, ValidationReport, validate_dataset

    raw = load_raw(path)
    structural = ValidationReport()
    for table in schema.tables:
        raw_table = raw.tables.get(table.name)
        if raw_table is None:
            structural.add(Check("columns", table.name, "", False,
                                 f"table not found in {path} (have: {', '.join(raw.tables) or 'none'})"))
            continue
        expected = table.field_names()
        missing = [c for c in expected if c not in raw_table.columns]
        extra = [c for c in raw_table.columns if c not in expected]
        detail = f"{len(expected) - len(missing)}/{len(expected)} columns present"
        if missing:
            detail += f"; missing: {', '.join(missing)}"
        if extra:
            detail += f"; extra (ignored): {', '.join(extra)}"
        structural.add(Check("columns", table.name, "", not missing, detail))
    report = validate_dataset(coerce_to_schema(raw, schema), tolerance=tolerance)
    report.checks = structural.checks + report.checks
    return report


def load_dataset(path: str, schema: Optional[Schema] = None) -> "Any":
    """Load ``path`` as a :class:`Dataset` for ``schema`` (or raw tables when
    no schema is given)."""
    raw = load_raw(path)
    return coerce_to_schema(raw, schema) if schema is not None else raw
