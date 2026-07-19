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
from typing import Any, Dict, List, Optional, Sequence

from .schema import Schema

TABULAR_FORMATS = ("csv", "jsonl", "parquet", "sqlite")


# --------------------------------------------------------------------------
# tabular export
# --------------------------------------------------------------------------
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
            if fmt == "csv":
                _rows_to_csv(rows, path)
            elif fmt == "jsonl":
                write_jsonl(rows, path)
            elif fmt == "parquet":
                _rows_to_parquet(rows, path)
            written.append(path)
    return written


def _rows_to_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv

    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows_to_parquet(rows: List[Dict[str, Any]], path: str) -> None:
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
    pd.DataFrame(rows).to_parquet(path, index=False)


def _to_sqlite(dataset: "Any", path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
    schema: Schema = dataset.schema
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        for table in schema.tables:
            cur.execute(_create_table_sql(table))
        for table in schema.tables:
            rows = dataset.tables.get(table.name, [])
            if not rows:
                continue
            cols = table.field_names()
            placeholders = ", ".join(["?"] * len(cols))
            col_list = ", ".join(f'"{c}"' for c in cols)
            sql = f'INSERT INTO "{table.name}" ({col_list}) VALUES ({placeholders})'
            cur.executemany(sql, [[_sqlite_value(r.get(c)) for c in cols] for r in rows])
        conn.commit()
    finally:
        conn.close()


def _create_table_sql(table) -> str:
    col_defs: List[str] = []
    fk_defs: List[str] = []
    for fs in table.fields:
        col_type = _sqlite_type(fs.type)
        constraint = ""
        if fs.type == "id":
            constraint = " PRIMARY KEY"
        elif fs.unique:
            constraint = " UNIQUE"
        col_defs.append(f'"{fs.name}" {col_type}{constraint}')
        if fs.type == "foreign_key":
            ref_table, ref_col = fs.get("references").split(".", 1)
            fk_defs.append(
                f'FOREIGN KEY ("{fs.name}") REFERENCES "{ref_table}" ("{ref_col}")'
            )
    all_defs = ",\n  ".join(col_defs + fk_defs)
    return f'CREATE TABLE "{table.name}" (\n  {all_defs}\n)'


def _sqlite_type(field_type: str) -> str:
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
