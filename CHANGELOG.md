# Changelog

## 0.2.0

### Fixed

- **`sdf` works.** The console script pointed at a `cli` module that was never packaged, so both editable and wheel installs failed with `ModuleNotFoundError: No module named 'cli'`. The command line now lives in `factory/cli.py`, and `python -m factory` works too. The root `cli.py` remains a thin shim for checkouts.
- **`unique: true` keeps values well-formed.** The old code appended `-N` after the whole value. That produced addresses like `x@inbox.example.com-1` (185,300 malformed of 200,000 at scale) and turned unique ints into strings like `57-1`. Now:
  - Emails dedup inside the local part (`ana.diaz2@example.com`) and URLs inside the host label.
  - Numbers, dates and pool-backed types are re-drawn inside their range.
  - A parse-time pigeonhole check rejects requests that cannot be satisfied.
- **Self-referencing foreign keys** passed validation but always crashed. They now build an acyclic hierarchy with null roots.
- **Formulas are null-safe.** A `null_rate` on an input column no longer crashes the formula with `TypeError`.
- **Formulas are bounded.** `9 ** 9 ** 9` used to hang and `"x" * 10**9` allocated a gigabyte. Both now raise `FormulaError` immediately.
- **The SQLite export is usable:**
  - Columns are typed. Sequential ids are `INTEGER PRIMARY KEY` and foreign keys inherit the parent's type, so `WHERE line_total > 1000` and `ORDER BY customer_id` compare numbers.
  - Every referenced column is `UNIQUE`, so `PRAGMA foreign_keys=ON` no longer raises "foreign key mismatch".
  - Tables are created parents-first and FK columns are indexed.
  - Each export ends with `PRAGMA foreign_key_check`.
- **Empty tables** get a CSV header row, and parquet keeps the typed columns.
- **Library code never exits.** Without `NVIDIA_API_KEY`, `TextFactory()` and `QAFactory()` raise `MissingAPIKeyError` instead of calling `sys.exit(2)`.
- **Object-wrapped model replies parse.** `{"questions": [...]}` and similar shapes are unwrapped. Before, the QA step turned the whole dict repr into a single question.
- **The parser now rejects** `end < start`, invalid ISO dates, `correlate`/`depends_on`/formula references to unknown fields, unknown distributions, out-of-range `null_rate`/`true_rate`, and FKs into empty tables.

### Added

- **Relational realism:**
  - `type: lookup` copies a column from the referenced parent row.
  - `after` / `before`, with optional `min_days` / `max_days`, keep dates in order.
  - `when` sets per-category parameter overrides.
  - FK `min_per_parent` guarantees each parent that many children, and `skew: zipf` (with `zipf_s`) sets popularity.
  - Formula date helpers: `year`, `month`, `day`, `weekday`, `days_between`, `hours_between`. Also `coalesce` and `is_null`.
  - The shipped `ecommerce.yaml` and `saas-users.yaml` now use these options. Results: 0 orders before signup (was 579 of 2000), 0 price mismatches (was 5000 of 5000), 0 empty orders (was 159), and 0 free accounts paying MRR (was 165).
- **`sdf infer`** builds a schema from a CSV/JSONL folder or a SQLite database. It detects types, statistics, primary and foreign keys, lookups, formulas, temporal anchors, correlations and skew. Rare category values are merged into `other`. The run prints a fidelity report comparing real and synthetic data.
- **`sdf validate --data`** checks files on disk (CSV/JSONL folder or SQLite) against a schema, including missing tables and columns.
- **Validator checks:**
  - Strict email check (syntax plus reserved RFC 2606 domain), fictional phone and RFC 5737 IP checks, and date bounds.
  - Nulls in id and foreign-key columns, and acyclic self-referencing hierarchies.
  - `temporal_order`, `lookup_consistency`, `when_bounds`, per-case distribution, and `parent_coverage`.
- **LLM path:**
  - `--record` / `--replay` cassettes give byte-identical replays with no key and no network.
  - `--dry-run` shows the plan without a key, and `--validate` runs `validate_records` on text and QA records.
  - Top-up rounds enforce the quota of every text task, and any shortfall is reported.
  - Dedup is incremental: exact first, then embeddings, falling back to lexical.
  - QA runs report duplicate, unanswerable and low-quality counts.
- **Tests** grew from 32 to 225, all offline. The LLM path is tested with the real `NIMClient` and `openai` SDK against a local fake OpenAI-compatible server.

### Changed (behaviour)

- **Phones** use the NANP fictional block `+1-AAA-555-01XX` instead of `+507-2xx/3xx-xxxx`, which is Panama's real landline format. **IPs** now span all three RFC 5737 documentation networks.
- **Shipped example data changed:**
  - `ecommerce.yaml`: `unit_price` is a lookup, `loyalty_years` is a formula on `signup_date`, orders follow signups, and product popularity is Zipf-skewed.
  - `saas-users.yaml`: plan-dependent seats and MRR, `last_login` after account creation, and events before the last login.
- **The validator is stricter:**
  - Emails on non-reserved domains, and phones or IPs outside the fictional ranges, fail the type check.
  - Categorical deviation is measured with a tolerance of at least `1/n`, so tiny tables are not failed for integer rounding.
- **`run_text_task` / `TextFactory` methods return a `TextResult`**, a `list` subclass with `quotas`, `shortfall`, `requests` and `dedup` attributes. **`run_qa_task` returns a `QAResult`** with counters. Code that treated the result as a list keeps working.
- **`NIMClient` networking:**
  - It retries only retryable errors (429, 5xx, timeouts), with exponential backoff, and disables the SDK's own retries so attempts are not multiplied.
  - Embeddings are requested as floats.
  - `.env` is read from the current directory only, and only when no key is passed.
- **Datasets list tables in schema order**, and CSV/parquet columns follow the schema's field order.
