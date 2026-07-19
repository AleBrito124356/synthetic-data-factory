#!/usr/bin/env python3
"""synthetic-data-factory command line.

    python cli.py generate tabular --schema schemas/ecommerce.yaml --out results/ --format csv,sqlite
    python cli.py generate text    --task schemas/text-task.yaml   --out results/reviews.jsonl --chat results/reviews.chat.jsonl
    python cli.py generate qa      --task schemas/qa-example.yaml  --out results/qa.jsonl --chat results/qa.chat.jsonl
    python cli.py validate --schema schemas/ecommerce.yaml
    python cli.py report   --schema schemas/saas-users.yaml

Only the text and qa subcommands need an NVIDIA NIM key. Everything else is
pure Python and runs offline.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make ``src/`` importable when run as ``python cli.py`` from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from factory import (  # noqa: E402
    generate,
    validate_dataset,
    distribution_report,
    render_distribution_report,
    export_dataset,
    write_jsonl,
    to_chat_jsonl,
)
from factory.schema import Schema, SchemaError  # noqa: E402


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_generate_tabular(args: argparse.Namespace) -> int:
    try:
        schema = Schema.from_yaml(args.schema)
    except SchemaError as exc:
        print(f"Schema error: {exc}", file=sys.stderr)
        return 1

    dataset = generate(schema, seed=args.seed)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    written = export_dataset(dataset, args.out, formats=formats, basename=args.basename)

    print("Generated tables:")
    for name, count in dataset.row_counts().items():
        print(f"  {name}: {count} rows")
    print(f"\nWrote {len(written)} file(s) to {args.out}:")
    for path in written:
        print(f"  {path}")

    if args.validate:
        report = validate_dataset(dataset)
        print("\n" + report.render())
        if not report.ok:
            return 1
    return 0


def cmd_generate_text(args: argparse.Namespace) -> int:
    from factory.text import run_text_task

    config = _load_yaml(args.task)
    records = run_text_task(config)
    write_jsonl(records, args.out)
    print(f"Wrote {len(records)} text record(s) to {args.out}")

    _summarize_labels(records)

    if args.chat:
        to_chat_jsonl(records, args.chat, system=args.chat_system)
        print(f"Wrote chat-format SFT file to {args.chat}")
    return 0


def cmd_generate_qa(args: argparse.Namespace) -> int:
    from factory.qa import run_qa_task

    config = _load_yaml(args.task)
    config["_base_dir"] = os.path.dirname(os.path.abspath(args.task))
    records = run_qa_task(config)
    write_jsonl(records, args.out)
    print(f"Wrote {len(records)} Q&A pair(s) to {args.out}")

    if records:
        avg = _avg_quality(records)
        if avg is not None:
            print(f"Average quality (min of 3 axes): {avg:.2f}")

    if args.chat:
        to_chat_jsonl(records, args.chat, system=args.chat_system, include_context=args.with_context)
        print(f"Wrote chat-format SFT file to {args.chat}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        schema = Schema.from_yaml(args.schema)
    except SchemaError as exc:
        print(f"Schema error: {exc}", file=sys.stderr)
        return 1
    dataset = generate(schema, seed=args.seed)
    report = validate_dataset(dataset, tolerance=args.tolerance)
    print(report.render())
    return 0 if report.ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    try:
        schema = Schema.from_yaml(args.schema)
    except SchemaError as exc:
        print(f"Schema error: {exc}", file=sys.stderr)
        return 1
    dataset = generate(schema, seed=args.seed)
    report = distribution_report(dataset)
    print(render_distribution_report(report))
    return 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _summarize_labels(records) -> None:
    keys = ("label", "sentiment", "category")
    for key in keys:
        if records and key in records[0]:
            counts: dict = {}
            for r in records:
                counts[r[key]] = counts.get(r[key], 0) + 1
            dist = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  {key} distribution: {dist}")
            return


def _avg_quality(records):
    vals = []
    for r in records:
        q = r.get("quality")
        if isinstance(q, dict) and q:
            vals.append(min(q.values()))
    if not vals:
        return None
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdf",
        description="Generate realistic synthetic datasets: tabular (offline) + text/qa (NVIDIA NIM).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a dataset.")
    gen_sub = gen.add_subparsers(dest="kind", required=True)

    p_tab = gen_sub.add_parser("tabular", help="Schema-driven tabular data (no API needed).")
    p_tab.add_argument("--schema", required=True, help="Path to a schema YAML file.")
    p_tab.add_argument("--out", default="results", help="Output directory (default: results).")
    p_tab.add_argument(
        "--format",
        default="csv",
        help="Comma-separated formats: csv,jsonl,parquet,sqlite (default: csv).",
    )
    p_tab.add_argument("--seed", type=int, default=None, help="Override the schema seed.")
    p_tab.add_argument("--basename", default="dataset", help="Base name for the sqlite file.")
    p_tab.add_argument("--validate", action="store_true", help="Validate after generating.")
    p_tab.set_defaults(func=cmd_generate_tabular)

    p_text = gen_sub.add_parser("text", help="LLM text dataset (needs NVIDIA_API_KEY).")
    p_text.add_argument("--task", required=True, help="Path to a text-task YAML file.")
    p_text.add_argument("--out", required=True, help="Output JSONL path.")
    p_text.add_argument("--chat", default=None, help="Also write chat-format SFT JSONL here.")
    p_text.add_argument("--chat-system", default=None, help="System prompt for the chat export.")
    p_text.set_defaults(func=cmd_generate_text)

    p_qa = gen_sub.add_parser("qa", help="LLM Q&A/instruction pairs (needs NVIDIA_API_KEY).")
    p_qa.add_argument("--task", required=True, help="Path to a qa-task YAML file.")
    p_qa.add_argument("--out", required=True, help="Output JSONL path.")
    p_qa.add_argument("--chat", default=None, help="Also write chat-format SFT JSONL here.")
    p_qa.add_argument("--chat-system", default=None, help="System prompt for the chat export.")
    p_qa.add_argument("--with-context", action="store_true", help="Embed the source context in the system turn.")
    p_qa.set_defaults(func=cmd_generate_qa)

    p_val = sub.add_parser("validate", help="Generate from a schema and validate the result.")
    p_val.add_argument("--schema", required=True, help="Path to a schema YAML file.")
    p_val.add_argument("--seed", type=int, default=None, help="Override the schema seed.")
    p_val.add_argument("--tolerance", type=float, default=0.05, help="Max categorical deviation (default 0.05).")
    p_val.set_defaults(func=cmd_validate)

    p_rep = sub.add_parser("report", help="Generate from a schema and print a distribution report.")
    p_rep.add_argument("--schema", required=True, help="Path to a schema YAML file.")
    p_rep.add_argument("--seed", type=int, default=None, help="Override the schema seed.")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
