"""synthetic-data-factory command line (installed as ``sdf``).

    sdf generate tabular --schema schemas/ecommerce.yaml --out results/ --format csv,sqlite
    sdf generate text    --task schemas/text-task.yaml   --out results/reviews.jsonl --chat results/reviews.chat.jsonl
    sdf generate qa      --task schemas/qa-example.yaml  --out results/qa.jsonl --chat results/qa.chat.jsonl
    sdf validate --schema schemas/ecommerce.yaml
    sdf validate --schema schemas/ecommerce.yaml --data results/      # check files on disk
    sdf report   --schema schemas/saas-users.yaml
    sdf infer    --data prod_export/ --out schemas/prod-like.yaml     # schema from real data

The same commands work as ``python -m factory ...`` and, from a checkout,
``python cli.py ...``. Only the text and qa subcommands need an NVIDIA NIM key.
Everything else is pure Python and runs offline.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import (
    generate,
    validate_dataset,
    distribution_report,
    render_distribution_report,
    export_dataset,
    write_jsonl,
    to_chat_jsonl,
)
from .schema import Schema, SchemaError


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _llm_client(args: argparse.Namespace):
    """Pick the model client for text/qa: replay (no key), record, or live."""
    from .llm import RecordingClient, ReplayClient

    if args.record and args.replay:
        raise ValueError("--record and --replay are mutually exclusive.")
    if args.replay:
        return ReplayClient(args.replay)
    from .nim import NIMClient

    client = NIMClient()  # raises MissingAPIKeyError without a key
    if args.record:
        return RecordingClient(client, args.record)
    return client


def _run_llm_command(args: argparse.Namespace, body) -> int:
    """Shared error handling for text/qa: friendly key message, clear
    cassette misses, and task-file mistakes without a traceback."""
    from .llm import CassetteMissError, LLMError
    from .nim import MissingAPIKeyError

    try:
        return body()
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except CassetteMissError as exc:
        print(f"Replay error: {exc}", file=sys.stderr)
        return 3
    except (LLMError, ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


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
    from .text import plan_text_task, run_text_task
    from .validate import validate_records

    def body() -> int:
        config = _load_yaml(args.task)
        if args.dry_run:
            print(plan_text_task(config).render())
            return 0
        client = _llm_client(args)
        records = run_text_task(config, client=client)
        write_jsonl(records, args.out)
        print(f"Wrote {len(records)} text record(s) to {args.out}")
        _summarize_labels(records)
        print(f"  requests: {records.requests} chat; dedup: {records.dedup or 'n/a'}")
        if records.shortfall:
            short = ", ".join(f"{g}={records.quotas[g] - n}/{records.quotas[g]}"
                              for g, n in records.shortfall.items())
            print(f"  WARNING shortfall after top-up rounds (got/quota): {short}")
        _report_cassette(args, client)

        if args.chat:
            to_chat_jsonl(records, args.chat, system=args.chat_system)
            print(f"Wrote chat-format SFT file to {args.chat}")
        if args.validate:
            report = validate_records(list(records), config, kind="text")
            print("\n" + report.render())
            if not report.ok:
                return 1
        return 0

    return _run_llm_command(args, body)


def cmd_generate_qa(args: argparse.Namespace) -> int:
    from .qa import plan_qa_task, run_qa_task
    from .validate import validate_records

    def body() -> int:
        config = _load_yaml(args.task)
        config["_base_dir"] = os.path.dirname(os.path.abspath(args.task))
        if args.dry_run:
            print(plan_qa_task(config).render())
            return 0
        client = _llm_client(args)
        records = run_qa_task(config, client=client)
        write_jsonl(records, args.out)
        print(f"Wrote {len(records)} Q&A pair(s) to {args.out}")
        print(
            f"  {records.chunks} chunk(s), {records.candidates} candidate question(s): "
            f"{records.duplicates} duplicate, {records.unanswerable} unanswerable, "
            f"{records.low_quality} below min_quality; {records.requests} chat request(s)"
        )
        if records:
            avg = _avg_quality(records)
            if avg is not None:
                print(f"Average quality (min of 3 axes): {avg:.2f}")
        _report_cassette(args, client)

        if args.chat:
            to_chat_jsonl(records, args.chat, system=args.chat_system, include_context=args.with_context)
            print(f"Wrote chat-format SFT file to {args.chat}")
        if args.validate:
            report = validate_records(list(records), config, kind="qa")
            print("\n" + report.render())
            if not report.ok:
                return 1
        return 0

    return _run_llm_command(args, body)


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        schema = Schema.from_yaml(args.schema)
    except SchemaError as exc:
        print(f"Schema error: {exc}", file=sys.stderr)
        return 1
    if args.data:
        from .load import DataLoadError, validate_data

        try:
            report = validate_data(schema, args.data, tolerance=args.tolerance)
        except DataLoadError as exc:
            print(f"Data error: {exc}", file=sys.stderr)
            return 1
        print(f"Validating {args.data} against {args.schema}\n")
    else:
        dataset = generate(schema, seed=args.seed)
        report = validate_dataset(dataset, tolerance=args.tolerance)
    print(report.render())
    return 0 if report.ok else 1


def cmd_infer(args: argparse.Namespace) -> int:
    from .infer import InferenceError, fidelity_report, infer_schema, render_fidelity, yaml_header
    from .load import DataLoadError, load_raw

    try:
        raw = load_raw(args.data)
        result = infer_schema(raw, rows_scale=args.rows_scale, min_category_count=args.min_category_count,
                              seed=args.seed)
    except (DataLoadError, InferenceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(result.to_yaml(yaml_header(args.data, raw.kind, args.min_category_count, args.rows_scale)))
    print(f"Inferred {len(result.schema.tables)} table(s) from {args.data} ({raw.kind}):")
    print(result.summary())
    print(f"\nWrote {args.out}")
    if not args.no_fidelity:
        synthetic = generate(result.schema)
        print("\n" + render_fidelity(fidelity_report(raw, result.schema, synthetic)))
    return 0


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
def _report_cassette(args: argparse.Namespace, client) -> None:
    if args.record:
        print(f"Recorded {client.calls} request(s) to cassette {args.record}")
    elif args.replay:
        left = client.remaining()
        extra = f" ({left} recorded response(s) unused)" if left else ""
        print(f"Replayed {client.calls} request(s) from cassette {args.replay}{extra}")


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
def _add_llm_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--record", metavar="CASSETTE",
                   help="Record every model request/response to this JSONL cassette.")
    p.add_argument("--replay", metavar="CASSETTE",
                   help="Serve responses from a recorded cassette (no key, no network).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned requests and the first prompt; send nothing.")
    p.add_argument("--validate", action="store_true",
                   help="Check fields, labels, quotas, duplicates (and QA quality) after generating.")


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

    p_text = gen_sub.add_parser(
        "text", help="LLM text dataset (needs NVIDIA_API_KEY, unless --replay or --dry-run).")
    p_text.add_argument("--task", required=True, help="Path to a text-task YAML file.")
    p_text.add_argument("--out", required=True, help="Output JSONL path.")
    p_text.add_argument("--chat", default=None, help="Also write chat-format SFT JSONL here.")
    p_text.add_argument("--chat-system", default=None, help="System prompt for the chat export.")
    _add_llm_flags(p_text)
    p_text.set_defaults(func=cmd_generate_text)

    p_qa = gen_sub.add_parser(
        "qa", help="LLM Q&A/instruction pairs (needs NVIDIA_API_KEY, unless --replay or --dry-run).")
    p_qa.add_argument("--task", required=True, help="Path to a qa-task YAML file.")
    p_qa.add_argument("--out", required=True, help="Output JSONL path.")
    p_qa.add_argument("--chat", default=None, help="Also write chat-format SFT JSONL here.")
    p_qa.add_argument("--chat-system", default=None, help="System prompt for the chat export.")
    p_qa.add_argument("--with-context", action="store_true", help="Embed the source context in the system turn.")
    _add_llm_flags(p_qa)
    p_qa.set_defaults(func=cmd_generate_qa)

    p_val = sub.add_parser(
        "validate", help="Validate a schema's output: freshly generated, or files on disk with --data.")
    p_val.add_argument("--schema", required=True, help="Path to a schema YAML file.")
    p_val.add_argument("--data", default=None,
                       help="Validate existing files instead: a folder of CSV/JSONL or a SQLite file.")
    p_val.add_argument("--seed", type=int, default=None, help="Override the schema seed.")
    p_val.add_argument("--tolerance", type=float, default=0.05, help="Max categorical deviation (default 0.05).")
    p_val.set_defaults(func=cmd_validate)

    p_inf = sub.add_parser(
        "infer", help="Infer a schema from real data (CSV/JSONL folder or SQLite) for a synthetic stand-in.")
    p_inf.add_argument("--data", required=True, help="Folder of CSV/JSONL files, or a .db/.sqlite file.")
    p_inf.add_argument("--out", required=True, help="Where to write the inferred schema YAML.")
    p_inf.add_argument("--rows-scale", type=float, default=1.0,
                       help="Multiply every table's row count (default 1.0).")
    p_inf.add_argument("--min-category-count", type=int, default=5,
                       help="Merge category values seen fewer times than this into 'other' (default 5).")
    p_inf.add_argument("--seed", type=int, default=42, help="Seed written into the schema (default 42).")
    p_inf.add_argument("--no-fidelity", action="store_true",
                       help="Skip generating a sample and comparing it with the real data.")
    p_inf.set_defaults(func=cmd_infer)

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
