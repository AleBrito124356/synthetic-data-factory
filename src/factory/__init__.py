"""synthetic-data-factory — realistic synthetic datasets from a schema.

Public surface:

    from factory import generate, Schema, validate_dataset, export_dataset
    from factory import infer_schema, load_raw, validate_data      # real data
    from factory import validate_records, RecordingClient, ReplayClient

The tabular path (``generate``, ``Schema``, ``validate_dataset``,
``export_dataset``, ``infer_schema``, ``validate_data``) has no LLM
dependency. The text/qa helpers import the NIM client lazily, so importing
this package never requires an API key or the ``openai`` package.
"""
from __future__ import annotations

from .schema import Schema, SchemaError, FieldSpec, TableSpec, load_schema
from .tabular import Dataset, TabularGenerator, generate
from .validate import (
    ValidationReport,
    validate_dataset,
    distribution_report,
    render_distribution_report,
    SYNTHETIC_NOTE,
)
from .export import export_dataset, write_jsonl, to_chat_jsonl
from .llm import CassetteMissError, RecordingClient, ReplayClient
from .load import load_raw, validate_data
from .infer import infer_schema, fidelity_report
from .validate import validate_records

__version__ = "0.2.0"

__all__ = [
    "Schema",
    "SchemaError",
    "FieldSpec",
    "TableSpec",
    "load_schema",
    "Dataset",
    "TabularGenerator",
    "generate",
    "ValidationReport",
    "validate_dataset",
    "distribution_report",
    "render_distribution_report",
    "SYNTHETIC_NOTE",
    "export_dataset",
    "write_jsonl",
    "to_chat_jsonl",
    "CassetteMissError",
    "RecordingClient",
    "ReplayClient",
    "load_raw",
    "validate_data",
    "infer_schema",
    "fidelity_report",
    "validate_records",
    "__version__",
]
