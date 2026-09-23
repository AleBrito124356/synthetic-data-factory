"""Export datasets and text/qa records to common formats.

Tabular datasets  -> csv, jsonl, parquet, sqlite (one file / table per table).
Text & qa records -> jsonl, and chat-format JSONL for supervised fine-tuning.

The chat JSONL matches the format expected by most SFT tooling — see the
sibling repo **fine-tuning-playbook** for QLoRA/DPO training that consumes it:
each line is ``{"messages": [{"role": ...}, ...]}``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import warnings
from typing import Any, Dict, List, Optional, Sequence

from .schema import Schema

TABULAR_FORMATS = ("csv", "jsonl", "parquet", "sqlite")


# --------------------------------------------------------------------------
# tabular export
# --------------------------------------------------------------------------
class ExportError(RuntimeError):
    """Raised when an export would produce an inconsistent file."""


def export_dataset(
    dataset: "Any",
    out_dir: str,
    formats: Sequence[str] = ("csv",),
    basename: Optional[str] = None,
) -> List[str]:
    """Write every table in ``dataset`` to ``out_dir`` in each requested format.

    Returns the list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    formats = [f.lower() for f in formats]

    unknown = [f for f in formats if f not in TABULAR_FORMATS]
    if unknown:
        raise ValueError(f"Unknown export format(s): {unknown}. Valid: {list(TABULAR_FORMATS)}")

    for fmt in formats:
        if fmt == "sqlite":
            path = os.path.join(out_dir, (basename or "dataset") + ".db")
            _to_sqlite(dataset, path)
            written.append(path)
            continue
        for table_name, rows in dataset.tables.items():
            path = os.path.join(out_dir, f"{table_name}.{fmt}")
            columns = _columns_for(dataset, table_name, rows)
            if fmt == "csv":
                _rows_to_csv(rows, path, columns)
            elif fmt == "jsonl":
                write_jsonl(rows, path)
            elif fmt == "parquet":
                _rows_to_parquet(rows, path, columns)
            written.append(path)
    return written


def _columns_for(dataset: "Any", table_name: str, rows: List[Dict[str, Any]]) -> List[str]:
    """Column order comes from the schema, so even an empty table gets a
    header row / typed columns."""
    schema = getattr(dataset, "schema", None)
    table = schema.get_table(table_name) if schema is not None else None
    if table is not None:
        return table.field_names()
    return list(rows[0].keys()) if rows else []


def _rows_to_csv(rows: List[Dict[str, Any]], path: str, fieldnames: Optional[List[str]] = None) -> None:
    import csv

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows_to_parquet(rows: List[Dict[str, Any]], path: str, columns: Optional[List[str]] = None) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for parquet export.") from exc
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "parquet export needs pyarrow. Install it with: pip install pyarrow"
        ) from exc
    pd.DataFrame(rows, columns=columns).to_parquet(path, index=False)


# ---- sqlite ---------------------------------------------------------------
def sqlite_column_types(dataset: "Any") -> Dict[str, Dict[str, str]]:
    """Declared SQLite type for every column: ``{table: {field: type}}``.

    Sequential integer ids are ``INTEGER``, foreign keys inherit the type of
    the column they reference, and formula columns are typed from the values
    they produced — so ``WHERE line_total > 1000`` and ``ORDER BY customer_id``
    compare numbers, not strings.
    """
    schema: Schema = dataset.schema
    cache: Dict[tuple, str] = {}

    def col_type(table_name: str, field_name: str, depth: int = 0) -> str:
        key = (table_name, field_name)
        if key in cache:
            return cache[key]
        table = schema.get_table(table_name)
        fs = table.get_field(field_name)
        t = fs.type
        if t == "id":
            out = "INTEGER" if fs.is_sequential_int_id else "TEXT"
        elif t in ("int", "bool"):
            out = "INTEGER"
        elif t == "float":
            out = "REAL"
        elif t == "foreign_key" and depth < 50:
            ref_table, ref_col = fs.references
            out = col_type(ref_table, ref_col, depth + 1)
        elif t == "lookup" and depth < 50:
            via = table.get_field(str(fs.get("via")))
            out = col_type(via.references[0], str(fs.get("column")), depth + 1)
        elif t in ("formula", "category", "lookup"):
            values = [r.get(field_name) for r in dataset.tables.get(table_name, [])]
            if t == "category" and not any(v is not None for v in values):
                values = list((fs.get("categories") or {}).keys())
            out = _affinity_from_values(values)
        else:
            out = "TEXT"
        cache[key] = out
        return out

    return {
        table.name: {fs.name: col_type(table.name, fs.name) for fs in table.fields}
        for table in schema.tables
    }


def _affinity_from_values(values: Sequence[Any]) -> str:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "NUMERIC"
    if all(isinstance(v, (bool, int)) for v in non_null):
        return "INTEGER"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "REAL"
    if all(isinstance(v, str) for v in non_null):
        return "TEXT"
    return ""  # mixed: no declared type (BLOB affinity keeps values as given)


def _to_sqlite(dataset: "Any", path: str) -> List[str]:
    """Write a relational SQLite database and prove it is consistent.

    * tables are created parents-first with typed columns;
    * the first ``id`` field is the PRIMARY KEY (``INTEGER PRIMARY KEY`` for
      sequential integer ids); every other column a foreign key points at gets
      a UNIQUE constraint, so SQLite accepts the FOREIGN KEY clause;
    * foreign-key columns are indexed;
    * after loading, ``PRAGMA foreign_keys=ON`` + ``PRAGMA foreign_key_check``
      must report nothing, or :class:`ExportError` is raised.

    Returns a list of warnings (foreign keys whose target column holds
    duplicate values cannot be declared and are skipped).
    """
    from .schema import topological_table_order

    if os.path.exists(path):
        os.remove(path)
    schema: Schema = dataset.schema
    types = sqlite_column_types(dataset)
    warnings_out: List[str] = []

    # Which parent columns are referenced, and can they carry UNIQUE?
    referenced: Dict[tuple, bool] = {}
    for table in schema.tables:
        for fs in table.fields:
            ref = fs.references
            if not ref:
                continue
            parent_rows = dataset.tables.get(ref[0], [])
            parent_values = [r.get(ref[1]) for r in parent_rows if r.get(ref[1]) is not None]
            referenced[ref] = len(parent_values) == len(set(map(_hashable_key, parent_values)))

    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        for table_name in topological_table_order(schema):
            table = schema.get_table(table_name)
            sql, skipped = _create_table_sql(table, types[table_name], referenced)
            warnings_out.extend(skipped)
            cur.execute(sql)
            for fs in table.fields:
                if fs.type == "foreign_key":
                    cur.execute(
                        f'CREATE INDEX "idx_{table.name}_{fs.name}" ON "{table.name}" ("{fs.name}")'
                    )
        for table_name in topological_table_order(schema):
            table = schema.get_table(table_name)
            rows = dataset.tables.get(table.name, [])
            if not rows:
                continue
            cols = table.field_names()
            placeholders = ", ".join(["?"] * len(cols))
            col_list = ", ".join(f'"{c}"' for c in cols)
            sql = f'INSERT INTO "{table.name}" ({col_list}) VALUES ({placeholders})'
            cur.executemany(sql, [[_sqlite_value(r.get(c)) for c in cols] for r in rows])
        conn.commit()

        cur.execute("PRAGMA foreign_keys=ON")
        try:
            violations = cur.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.DatabaseError as exc:
            raise ExportError(f"SQLite rejected the foreign keys in {path}: {exc}") from exc
        if violations:
            table, rowid, parent, _ = violations[0]
            raise ExportError(
                f"{len(violations)} foreign-key violation(s) in {path}, e.g. "
                f"{table} rowid {rowid} -> {parent}."
            )
    finally:
        conn.close()
    for msg in warnings_out:
        warnings.warn(msg, stacklevel=3)
    return warnings_out


def _hashable_key(v: Any) -> Any:
    try:
        hash(v)
        return v
    except TypeError:
        return json.dumps(v, sort_keys=True, default=str)


def _create_table_sql(table, types: Dict[str, str], referenced: Dict[tuple, bool]):
    col_defs: List[str] = []
    fk_defs: List[str] = []
    skipped: List[str] = []
    pk_done = False
    for fs in table.fields:
        col_type = types.get(fs.name, "TEXT")
        decl = f'"{fs.name}" {col_type}'.rstrip()
        is_referenced = (table.name, fs.name) in referenced
        if fs.type == "id" and not pk_done:
            decl += " PRIMARY KEY NOT NULL"
            pk_done = True
        elif fs.type in ("id", "uuid") or fs.unique:
            decl += " UNIQUE"
            if fs.type == "id":
                decl += " NOT NULL"
        elif is_referenced and referenced[(table.name, fs.name)]:
            decl += " UNIQUE"
        col_defs.append(decl)
        ref = fs.references
        if ref:
            ref_table, ref_col = ref
            if referenced.get(ref):
                fk_defs.append(
                    f'FOREIGN KEY ("{fs.name}") REFERENCES "{ref_table}" ("{ref_col}")'
                )
            else:
                skipped.append(
                    f"sqlite: not declaring FOREIGN KEY {table.name}.{fs.name} -> "
                    f"{ref_table}.{ref_col} because that column holds duplicate values "
                    f"(a foreign key must target a unique column)."
                )
    all_defs = ",\n  ".join(col_defs + fk_defs)
    return f'CREATE TABLE "{table.name}" (\n  {all_defs}\n)', skipped


def _sqlite_type(field_type: str) -> str:
    """Kept for backwards compatibility; see :func:`sqlite_column_types`."""
    if field_type == "int":
        return "INTEGER"
    if field_type == "float":
        return "REAL"
    if field_type == "bool":
        return "INTEGER"
    return "TEXT"


def _sqlite_value(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if v is None or isinstance(v, (int, float, str)):
        return v
    return json.dumps(v, ensure_ascii=False)


# --------------------------------------------------------------------------
# record export (text / qa)
# --------------------------------------------------------------------------
def write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def to_chat_jsonl(
    records: Sequence[Dict[str, Any]],
    path: str,
    system: Optional[str] = None,
    include_context: bool = False,
) -> str:
    """Convert text/qa records into chat-format JSONL for SFT.

    Mapping is auto-detected from record shape:

    * qa            -> user = question (optionally + context), assistant = answer
    * classification-> user = text,      assistant = label   (system = task hint)
    * review/ticket -> user = task hint, assistant = the generated text

    Every line is ``{"messages": [...]}``.
    """
    lines: List[Dict[str, Any]] = []
    for rec in records:
        messages = _record_to_messages(rec, system=system, include_context=include_context)
        lines.append({"messages": messages})
    return write_jsonl(lines, path)


def _record_to_messages(
    rec: Dict[str, Any], system: Optional[str], include_context: bool
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []

    # Q&A pair.
    if "question" in rec and "answer" in rec:
        sys_msg = system
        if include_context and rec.get("context"):
            ctx = f"Use only this context to answer:\n{rec['context']}"
            sys_msg = f"{system}\n\n{ctx}" if system else ctx
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": str(rec["question"])})
        messages.append({"role": "assistant", "content": str(rec["answer"])})
        return messages

    # Classification.
    if "label" in rec and "text" in rec:
        hint = system or "Classify the message into the correct category."
        messages.append({"role": "system", "content": hint})
        messages.append({"role": "user", "content": str(rec["text"])})
        messages.append({"role": "assistant", "content": str(rec["label"])})
        return messages

    # Generic generative record (review/persona/ticket/paraphrase).
    if "text" in rec:
        instruction = system or _instruction_for(rec)
        messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": _user_prompt_for(rec)})
        messages.append({"role": "assistant", "content": str(rec["text"])})
        return messages

    # Fallback: dump the record as the assistant turn.
    messages.append({"role": "user", "content": system or "Generate an example."})
    messages.append({"role": "assistant", "content": json.dumps(rec, ensure_ascii=False)})
    return messages


def _instruction_for(rec: Dict[str, Any]) -> str:
    kind = rec.get("kind", "")
    if kind == "review":
        return "Write a product review with the requested sentiment."
    if kind == "ticket":
        return "Write a customer support ticket for the given category."
    if kind == "persona":
        return "Write a short persona bio for the given audience."
    if kind == "paraphrase":
        return "Paraphrase the given sentence."
    return "Generate the requested text."


def _user_prompt_for(rec: Dict[str, Any]) -> str:
    kind = rec.get("kind", "")
    if kind == "review":
        return f"Product: {rec.get('product', 'unknown')}. Sentiment: {rec.get('sentiment', 'any')}."
    if kind == "ticket":
        return f"Category: {rec.get('category', 'general')}."
    if kind == "persona":
        return f"Audience: {rec.get('context', 'general')}."
    if kind == "paraphrase":
        return str(rec.get("intent", "the sentence"))
    return "Generate one example."
