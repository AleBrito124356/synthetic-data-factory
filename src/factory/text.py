"""LLM-generated text datasets on NVIDIA NIM (or any :class:`~factory.llm.ChatClient`).

Five task types, each returning a :class:`TextResult` (a list of record dicts
plus run metadata):

* ``paraphrase``       — N variations of each seed utterance (intent data).
* ``classification``   — a class-balanced text-classification dataset.
* ``personas``         — short persona bios for a given audience.
* ``reviews``          — product reviews with a controlled sentiment mix.
* ``tickets``          — support tickets across categories.

Diversity is encouraged two ways: prompts are seeded with rotating "angles"
(aspects, voices, tones) so the model is nudged away from repetition, and a
dedup pass removes exact and near-duplicates within each group using NIM
embeddings (semantic) or a lexical shingle fallback.

Quotas are enforced, not hoped for: every group (label, sentiment, category,
seed, context) has a target count. When dedup or a short model reply leaves a
group under target, bounded *top-up rounds* ask for exactly the missing
number, showing the model a few existing examples to avoid. Anything still
missing after ``max_rounds`` is reported in ``TextResult.shortfall`` — never
silently dropped.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .llm import CassetteMissError, as_text, as_text_list

# ---- diversity seed pools -------------------------------------------------
_REVIEW_ASPECTS = [
    "build quality", "value for money", "customer support", "ease of setup",
    "battery life", "comfort", "shipping speed", "the mobile app", "durability",
    "sound quality", "packaging", "instructions", "compatibility",
]
_REVIEW_VOICES = [
    "a busy parent", "a college student", "a professional reviewer",
    "a first-time buyer", "a long-time customer", "a skeptical shopper",
    "a gift-giver", "a power user", "a bargain hunter",
]
_TICKET_TONES = [
    "polite and detailed", "frustrated and terse", "confused and rambling",
    "formal and corporate", "casual with typos", "urgent",
]

TEXT_TASKS = ("paraphrase", "classification", "personas", "reviews", "tickets")
DEFAULT_MAX_ROUNDS = 5
PERSONA_BATCH = 10


class TextResult(list):
    """Generated records (a plain ``list`` of dicts) plus run metadata.

    ``quotas``     — target count per group
    ``shortfall``  — groups that stayed under target after all rounds
    ``requests``   — chat requests sent
    ``dedup``      — dedup method actually used ("embedding", "lexical", "none")
    """

    def __init__(self, records: Sequence[Dict[str, Any]] = (), group_key: str = "",
                 quotas: Optional[Dict[str, int]] = None, shortfall: Optional[Dict[str, int]] = None,
                 requests: int = 0, dedup: str = ""):
        super().__init__(records)
        self.group_key = group_key
        self.quotas = dict(quotas or {})
        self.shortfall = dict(shortfall or {})
        self.requests = requests
        self.dedup = dedup


class TextFactory:
    def __init__(
        self,
        client: Optional[Any] = None,
        seed: int = 7,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        dedup: str = "embedding",
        dedup_threshold: float = 0.92,
    ):
        if client is None:
            from .nim import NIMClient  # raises MissingAPIKeyError without a key

            client = NIMClient()
        if dedup not in ("embedding", "lexical", "none"):
            raise ValueError("dedup must be 'embedding', 'lexical' or 'none'.")
        self.client = client
        self.rng = random.Random(seed)
        self.max_rounds = max(1, int(max_rounds))
        self.dedup = dedup
        self.dedup_threshold = float(dedup_threshold)
        self._requests = 0
        self._dedup_used: set = set()

    # ---- public tasks ----------------------------------------------------
    def paraphrase(self, seeds: Sequence[str], variations: int = 5, temperature: float = 0.9) -> TextResult:
        def ask(seed: str, need: int, avoid: List[str]) -> List[Dict[str, Any]]:
            items = as_text_list(self._chat_json(paraphrase_prompt(seed, need, avoid), temperature),
                                 keys=("paraphrases", "variations", "rewrites"))
            return [{"text": _clean(t), "intent": seed, "kind": "paraphrase"} for t in items]

        return self._run_groups("intent", {s: int(variations) for s in seeds}, ask)

    def classification(
        self,
        labels: Sequence[str],
        per_class: int = 10,
        domain: str = "short customer messages",
        temperature: float = 0.9,
    ) -> TextResult:
        labels = list(labels)

        def ask(label: str, need: int, avoid: List[str]) -> List[Dict[str, Any]]:
            prompt = classification_prompt(label, labels, need, domain, avoid)
            items = as_text_list(self._chat_json(prompt, temperature), keys=("examples", "messages"))
            return [{"text": _clean(t), "label": label, "kind": "classification"} for t in items]

        return self._run_groups("label", {label: int(per_class) for label in labels}, ask)

    def personas(self, count: int = 20, context: str = "general consumers", temperature: float = 0.95) -> TextResult:
        def ask(ctx: str, need: int, avoid: List[str]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            remaining = need
            while remaining > 0:
                take = min(PERSONA_BATCH, remaining)
                items = as_text_list(self._chat_json(personas_prompt(take, ctx, avoid), temperature),
                                     keys=("personas", "bios"))
                out.extend({"text": _clean(t), "context": ctx, "kind": "persona"} for t in items)
                remaining -= take
            return out

        return self._run_groups("context", {context: int(count)}, ask)

    def reviews(
        self,
        product: str,
        sentiments: Optional[Dict[str, float]] = None,
        count: int = 30,
        temperature: float = 0.95,
    ) -> TextResult:
        sentiments = sentiments or {"positive": 0.5, "neutral": 0.2, "negative": 0.3}
        plan = _weighted_counts(sentiments, int(count))

        def ask(sentiment: str, need: int, avoid: List[str]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for _ in range(need):
                aspect, voice, rating = self._review_spec(sentiment)
                obj = self._chat_json(review_prompt(product, sentiment, voice, aspect), temperature)
                out.append({
                    "text": _clean(as_text(obj, keys=("review",))),
                    "sentiment": sentiment,
                    "rating": rating,
                    "aspect": aspect,
                    "product": product,
                    "kind": "review",
                })
            return out

        return self._run_groups("sentiment", plan, ask)

    def tickets(
        self,
        categories: Sequence[str],
        per_category: int = 8,
        product: str = "our software product",
        temperature: float = 0.95,
    ) -> TextResult:
        def ask(category: str, need: int, avoid: List[str]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for _ in range(need):
                tone = self.rng.choice(_TICKET_TONES)
                obj = self._chat_json(ticket_prompt(product, category, tone), temperature)
                subject, body = _subject_body(obj)
                subject, body = _clean(subject), _clean(body)
                out.append({
                    "subject": subject,
                    "body": body,
                    "text": f"{subject}\n\n{body}",
                    "category": category,
                    "tone": tone,
                    "kind": "ticket",
                })
            return out

        return self._run_groups("category", {c: int(per_category) for c in categories}, ask)

    # ---- quota engine ----------------------------------------------------
    def _run_groups(self, group_key: str, quotas: Dict[str, int], ask) -> TextResult:
        """For each group: ask, dedup, and top up until the quota is met or
        ``max_rounds`` is exhausted."""
        start_requests = self._requests
        records: List[Dict[str, Any]] = []
        shortfall: Dict[str, int] = {}
        for group, quota in quotas.items():
            pool = _DedupPool(self, self.dedup, self.dedup_threshold)
            for round_no in range(self.max_rounds):
                need = quota - len(pool.kept)
                if need <= 0:
                    break
                avoid = [r["text"] for r in pool.kept[-5:]] if round_no else []
                pool.add(ask(group, need, avoid), limit=quota)
            self._dedup_used.add(pool.method)
            records.extend(pool.kept[:quota])
            if len(pool.kept) < quota:
                shortfall[str(group)] = quota - len(pool.kept)
        used = sorted(self._dedup_used)
        return TextResult(records, group_key=group_key, quotas={str(k): v for k, v in quotas.items()},
                          shortfall=shortfall, requests=self._requests - start_requests,
                          dedup="+".join(used))

    # ---- internals -------------------------------------------------------
    def _chat_json(self, prompt: str, temperature: float) -> Any:
        self._requests += 1
        return self.client.chat_json(prompt, temperature=temperature)

    def _review_spec(self, sentiment: str):
        """All random draws for one review, in a fixed order (shared with the
        dry-run planner so its first prompt matches the real run)."""
        aspect = self.rng.choice(_REVIEW_ASPECTS)
        voice = self.rng.choice(_REVIEW_VOICES)
        rating = self._rating_for(sentiment)
        return aspect, voice, rating

    # Backwards-compatible helpers used by older callers/tests.
    @staticmethod
    def _json_list(obj: Any) -> List[str]:
        return as_text_list(obj)

    @staticmethod
    def _json_scalar(obj: Any) -> str:
        return as_text(obj)

    def _rating_for(self, sentiment: str) -> int:
        if sentiment == "positive":
            return self.rng.choice([4, 5, 5])
        if sentiment == "negative":
            return self.rng.choice([1, 1, 2])
        return 3


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------
class _DedupPool:
    """Accepted records for one group, with incremental near-dup detection.

    Exact duplicates (case/whitespace-insensitive) are always dropped. Near
    duplicates are dropped by cosine similarity of embeddings when the
    endpoint is available, else by shingle Jaccard similarity.
    """

    def __init__(self, factory: TextFactory, method: str, threshold: float):
        self.factory = factory
        self.method = method
        self.threshold = threshold
        self.kept: List[Dict[str, Any]] = []
        self._keys: set = set()
        self._vecs: List[List[float]] = []
        self._shingles: List[set] = []

    def add(self, candidates: List[Dict[str, Any]], limit: int) -> None:
        fresh: List[Dict[str, Any]] = []
        batch_keys: set = set()
        for c in candidates:
            key = _normalize(c.get("text", ""))
            if not key or key in self._keys or key in batch_keys:
                continue
            batch_keys.add(key)
            fresh.append(c)
        if not fresh:
            return

        vectors: Optional[List[List[float]]] = None
        if self.method == "embedding":
            try:
                vectors = self.factory.client.embed([c["text"] for c in fresh], input_type="passage")
                if len(vectors) != len(fresh):
                    raise ValueError("embedding count mismatch")
            except CassetteMissError:
                raise
            except Exception:
                # Embedding endpoint unavailable — degrade to lexical dedup.
                self.method = "lexical"
                vectors = None

        lexical_threshold = min(self.threshold, 0.8)
        for i, c in enumerate(fresh):
            if len(self.kept) >= limit:
                break
            sh = _shingles(c["text"])
            if self.method == "embedding" and vectors is not None:
                vec = vectors[i]
                if any(_cosine(vec, kv) >= self.threshold for kv in self._vecs):
                    continue
                self._vecs.append(vec)
            elif self.method == "lexical":
                if any(_jaccard(sh, k) >= lexical_threshold for k in self._shingles):
                    continue
            self.kept.append(c)
            self._keys.add(_normalize(c["text"]))
            self._shingles.append(sh)


# --------------------------------------------------------------------------
# prompt builders (shared by the factory and the dry-run planner)
# --------------------------------------------------------------------------
def _avoid_block(avoid: Sequence[str]) -> str:
    if not avoid:
        return ""
    lines = "\n".join(f"- {a[:160]}" for a in avoid)
    return f"\n\nThese already exist; do not repeat or closely paraphrase them:\n{lines}"


def paraphrase_prompt(seed: str, n: int, avoid: Sequence[str] = ()) -> str:
    return (
        f"Rewrite the following sentence in {n} different ways. "
        f"Keep the meaning identical but vary wording, tone, and length. "
        f"Return a JSON array of {n} strings.\n\nSentence: {seed}" + _avoid_block(avoid)
    )


def classification_prompt(label: str, labels: Sequence[str], n: int, domain: str,
                          avoid: Sequence[str] = ()) -> str:
    other = [l for l in labels if l != label]
    return (
        f"Generate {n} realistic examples of {domain} that belong to the "
        f"category '{label}'. The other categories in this dataset are: "
        f"{', '.join(other)}. Make examples clearly '{label}' and distinct from "
        f"the others. Vary length, tone, and detail. "
        f"Return a JSON array of {n} strings." + _avoid_block(avoid)
    )


def personas_prompt(n: int, context: str, avoid: Sequence[str] = ()) -> str:
    return (
        f"Write {n} short, distinct persona bios for: {context}. "
        f"Each bio is 2-3 sentences and includes a first name, a role or "
        f"situation, and one goal or frustration. Make them diverse in age, "
        f"background, and voice. Return a JSON array of {n} strings." + _avoid_block(avoid)
    )


def review_prompt(product: str, sentiment: str, voice: str, aspect: str) -> str:
    return (
        f"Write a single realistic {sentiment} product review for: {product}. "
        f"Write as {voice}, focusing partly on {aspect}. 2-4 sentences, first "
        f"person, natural and specific. Do not mention that it is synthetic. "
        f"Return just the review text as a JSON string."
    )


def ticket_prompt(product: str, category: str, tone: str) -> str:
    return (
        f"Write one realistic customer support ticket about {product}. "
        f"The issue category is '{category}'. Tone: {tone}. Include a short "
        f"subject line and a body of 2-4 sentences. "
        f'Return JSON: {{"subject": "...", "body": "..."}}.'
    )


# --------------------------------------------------------------------------
# config-driven entry points (used by the CLI)
# --------------------------------------------------------------------------
def _factory_options(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "seed": int(config.get("seed", 7)),
        "max_rounds": int(config.get("max_rounds", DEFAULT_MAX_ROUNDS)),
        "dedup": str(config.get("dedup", "embedding")),
        "dedup_threshold": float(config.get("dedup_threshold", 0.92)),
    }


def _require(config: Dict[str, Any], key: str, task: str) -> Any:
    if key not in config or config[key] in (None, "", []):
        raise ValueError(f"The '{task}' text task needs '{key}' in the task file.")
    return config[key]


def run_text_task(config: Dict[str, Any], client: Optional[Any] = None) -> TextResult:
    task = str(config.get("task", "")).lower()
    if task not in TEXT_TASKS:
        raise ValueError(
            f"Unknown text task '{task}'. Expected one of: {', '.join(TEXT_TASKS)}."
        )
    factory = TextFactory(client=client, **_factory_options(config))
    temperature = float(config.get("temperature", 0.9))

    if task == "paraphrase":
        return factory.paraphrase(
            seeds=_require(config, "seeds", task),
            variations=int(config.get("variations", 5)),
            temperature=temperature,
        )
    if task == "classification":
        return factory.classification(
            labels=_require(config, "labels", task),
            per_class=int(config.get("per_class", 10)),
            domain=config.get("domain", "short customer messages"),
            temperature=temperature,
        )
    if task == "personas":
        return factory.personas(
            count=int(config.get("count", 20)),
            context=config.get("context", "general consumers"),
            temperature=temperature,
        )
    if task == "reviews":
        return factory.reviews(
            product=_require(config, "product", task),
            sentiments=config.get("sentiments"),
            count=int(config.get("count", 30)),
            temperature=temperature,
        )
    return factory.tickets(
        categories=_require(config, "categories", task),
        per_category=int(config.get("per_category", 8)),
        product=config.get("product", "our software product"),
        temperature=temperature,
    )


def text_task_quotas(config: Dict[str, Any]) -> Dict[str, Any]:
    """``(group_key, {group: quota})`` a text task config asks for."""
    task = str(config.get("task", "")).lower()
    if task == "paraphrase":
        return {"group_key": "intent",
                "quotas": {s: int(config.get("variations", 5)) for s in config.get("seeds", [])}}
    if task == "classification":
        return {"group_key": "label",
                "quotas": {l: int(config.get("per_class", 10)) for l in config.get("labels", [])}}
    if task == "personas":
        return {"group_key": "context",
                "quotas": {config.get("context", "general consumers"): int(config.get("count", 20))}}
    if task == "reviews":
        sentiments = config.get("sentiments") or {"positive": 0.5, "neutral": 0.2, "negative": 0.3}
        return {"group_key": "sentiment",
                "quotas": _weighted_counts(sentiments, int(config.get("count", 30)))}
    if task == "tickets":
        return {"group_key": "category",
                "quotas": {c: int(config.get("per_category", 8)) for c in config.get("categories", [])}}
    raise ValueError(f"Unknown text task '{task}'. Expected one of: {', '.join(TEXT_TASKS)}.")


@dataclass
class RequestPlan:
    """What a run will send before any top-up rounds (for ``--dry-run``)."""

    task: str
    chat_requests: int
    embed_requests: int
    groups: Dict[str, int]
    first_prompt: str
    first_system: str = ""
    max_extra_rounds: int = 0
    notes: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Dry run: task '{self.task}' (no requests sent, no key needed)",
            f"  planned chat requests:      {self.chat_requests}",
            f"  planned embedding requests: {self.embed_requests}",
        ]
        if self.groups:
            quota = ", ".join(f"{k}={v}" for k, v in self.groups.items())
            lines.append(f"  quotas: {quota}")
        if self.max_extra_rounds:
            lines.append(
                f"  top-up: up to {self.max_extra_rounds} more round(s) per group, only when "
                f"dedup or short replies leave it under quota"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        if self.first_system:
            lines += ["", "First request - system:", self.first_system]
        lines += ["", "First request - user prompt:", self.first_prompt]
        return "\n".join(lines)


class _PlanOnlyClient:
    def __getattr__(self, name: str):  # pragma: no cover - defensive
        raise RuntimeError("The dry-run planner never calls the model.")


def plan_text_task(config: Dict[str, Any]) -> RequestPlan:
    """Count the requests a text task will send and build its first prompt,
    without a key and without calling the model."""
    from .llm import json_system

    task = str(config.get("task", "")).lower()
    shape = text_task_quotas(config)
    quotas = shape["quotas"]
    opts = _factory_options(config)
    factory = TextFactory(client=_PlanOnlyClient(), **opts)
    groups_with_quota = [g for g, q in quotas.items() if q > 0]
    embeds = len(groups_with_quota) if opts["dedup"] == "embedding" else 0

    if task == "paraphrase":
        seeds = _require(config, "seeds", task)
        chats = len(seeds)
        first = paraphrase_prompt(seeds[0], int(config.get("variations", 5)))
    elif task == "classification":
        labels = _require(config, "labels", task)
        chats = len(labels)
        first = classification_prompt(labels[0], labels, int(config.get("per_class", 10)),
                                      config.get("domain", "short customer messages"))
    elif task == "personas":
        count = int(config.get("count", 20))
        chats = math.ceil(count / PERSONA_BATCH)
        first = personas_prompt(min(PERSONA_BATCH, count), config.get("context", "general consumers"))
    elif task == "reviews":
        product = _require(config, "product", task)
        chats = sum(quotas.values())
        sentiment = groups_with_quota[0] if groups_with_quota else "positive"
        aspect, voice, _ = factory._review_spec(sentiment)
        first = review_prompt(product, sentiment, voice, aspect)
    else:  # tickets
        categories = _require(config, "categories", task)
        chats = len(categories) * int(config.get("per_category", 8))
        tone = factory.rng.choice(_TICKET_TONES)
        first = ticket_prompt(config.get("product", "our software product"), categories[0], tone)

    return RequestPlan(
        task=task,
        chat_requests=chats,
        embed_requests=embeds,
        groups={str(k): v for k, v in quotas.items()},
        first_prompt=first,
        first_system=json_system(None),
        max_extra_rounds=opts["max_rounds"] - 1,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clean(text: Any) -> str:
    s = str(text).strip()
    # Drop wrapping quotes the model sometimes adds.
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _subject_body(obj: Any) -> tuple:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if isinstance(obj, dict):
        subject = obj.get("subject") or obj.get("title") or "Support request"
        body = obj.get("body") or obj.get("message") or obj.get("description") or ""
        return str(subject), str(body)
    if isinstance(obj, str):
        return "Support request", obj
    return "Support request", str(obj)


def _weighted_counts(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Largest-remainder split of ``total`` across weighted keys."""
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("Sentiment weights must sum to a positive number.")
    ideal = {k: (v / s) * total for k, v in weights.items()}
    counts = {k: int(v) for k, v in ideal.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights.keys(), key=lambda k: ideal[k] - counts[k], reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _dedup_by_vectors(items: List[Dict[str, Any]], vectors: List[List[float]], threshold: float) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    kept_vecs: List[Sequence[float]] = []
    for item, vec in zip(items, vectors):
        if any(_cosine(vec, kv) >= threshold for kv in kept_vecs):
            continue
        kept.append(item)
        kept_vecs.append(vec)
    return kept


def _shingles(text: str, n: int = 3) -> set:
    tokens = [t for t in text.lower().split() if t]
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _dedup_lexical(items: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    kept_sh: List[set] = []
    for item in items:
        sh = _shingles(item["text"])
        if any(_jaccard(sh, k) >= threshold for k in kept_sh):
            continue
        kept.append(item)
        kept_sh.append(sh)
    return kept
