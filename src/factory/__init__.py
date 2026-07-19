"""synthetic-data-factory — realistic synthetic datasets from a schema.

Public surface:

    from factory import generate, Schema, validate_dataset, export_dataset

The tabular path (``generate``, ``Schema``, ``validate_dataset``,
``export_dataset``) has no LLM dependency. The text/qa helpers import the
NIM client lazily, so importing this package never requires an API key.
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

__version__ = "0.1.0"

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
    "__version__",
]
