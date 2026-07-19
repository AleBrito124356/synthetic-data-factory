"""LLM-generated text datasets on NVIDIA NIM.

Five task types, each returning a list of record dicts:

* ``paraphrase``       — N variations of each seed utterance (intent data).
* ``classification``   — a class-balanced text-classification dataset.
* ``personas``         — short persona bios for a given audience.
* ``reviews``          — product reviews with controlled sentiment mix.
* ``tickets``          — support tickets across categories.

Diversity is encouraged two ways: prompts are seeded with rotating "angles"
(aspects, styles, scenarios) so the model is nudged away from repetition, and a
dedup pass removes near-duplicates using either NIM embeddings (semantic) or a
lexical shingle fallback (works offline once text exists).
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

from .nim import NIMClient

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


class TextFactory:
    def __init__(self, client: Optional[NIMClient] = None, seed: int = 7):
        self.client = client or NIMClient()
        self.rng = random.Random(seed)

    # ---- public tasks ----------------------------------------------------
    def paraphrase(self, seeds: Sequence[str], variations: int = 5, temperature: float = 0.9) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for seed in seeds:
            prompt = (
                f"Rewrite the following sentence in {variations} different ways. "
                f"Keep the meaning identical but vary wording, tone, and length. "
                f"Return a JSON array of {variations} strings.\n\nSentence: {seed}"
            )
            items = self._json_list(self.client.chat_json(prompt, temperature=temperature))
            for text in items[:variations]:
                records.append({"text": _clean(text), "intent": seed, "kind": "paraphrase"})
        return self._dedup_grouped(records, group_key="intent")

    def classification(
        self,
        labels: Sequence[str],
        per_class: int = 10,
        domain: str = "short customer messages",
        temperature: float = 0.9,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for label in labels:
            other = [l for l in labels if l != label]
            prompt = (
                f"Generate {per_class} realistic examples of {domain} that belong to the "
                f"category '{label}'. The other categories in this dataset are: "
                f"{', '.join(other)}. Make examples clearly '{label}' and distinct from "
                f"the others. Vary length, tone, and detail. "
                f"Return a JSON array of {per_class} strings."
            )
            items = self._json_list(self.client.chat_json(prompt, temperature=temperature))
            for text in items[:per_class]:
                records.append({"text": _clean(text), "label": label, "kind": "classification"})
        return self._dedup_grouped(records, group_key="label")

    def personas(self, count: int = 20, context: str = "general consumers", temperature: float = 0.95) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        batch = 10
        made = 0
        while made < count:
            take = min(batch, count - made)
            prompt = (
                f"Write {take} short, distinct persona bios for: {context}. "
                f"Each bio is 2-3 sentences and includes a first name, a role or "
                f"situation, and one goal or frustration. Make them diverse in age, "
                f"background, and voice. Return a JSON array of {take} strings."
            )
            items = self._json_list(self.client.chat_json(prompt, temperature=temperature))
            for text in items[:take]:
                records.append({"text": _clean(text), "context": context, "kind": "persona"})
            made += take
        return self._dedup_grouped(records, group_key="context")

    def reviews(
        self,
        product: str,
        sentiments: Optional[Dict[str, float]] = None,
        count: int = 30,
        temperature: float = 0.95,
    ) -> List[Dict[str, Any]]:
        sentiments = sentiments or {"positive": 0.5, "neutral": 0.2, "negative": 0.3}
        plan = _weighted_counts(sentiments, count)
        records: List[Dict[str, Any]] = []
        for sentiment, k in plan.items():
            for _ in range(k):
                aspect = self.rng.choice(_REVIEW_ASPECTS)
                voice = self.rng.choice(_REVIEW_VOICES)
                rating = self._rating_for(sentiment)
                prompt = (
                    f"Write a single realistic {sentiment} product review for: {product}. "
                    f"Write as {voice}, focusing partly on {aspect}. 2-4 sentences, first "
                    f"person, natural and specific. Do not mention that it is synthetic. "
                    f"Return just the review text as a JSON string."
                )
                text = self._json_scalar(self.client.chat_json(prompt, temperature=temperature))
                records.append(
                    {
                        "text": _clean(text),
                        "sentiment": sentiment,
                        "rating": rating,
                        "aspect": aspect,
                        "product": product,
                        "kind": "review",
                    }
                )
        return self._dedup_grouped(records, group_key="sentiment")

    def tickets(
        self,
        categories: Sequence[str],
        per_category: int = 8,
        product: str = "our software product",
        temperature: float = 0.95,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for category in categories:
            for _ in range(per_category):
                tone = self.rng.choice(_TICKET_TONES)
                prompt = (
                    f"Write one realistic customer support ticket about {product}. "
                    f"The issue category is '{category}'. Tone: {tone}. Include a short "
                    f"subject line and a body of 2-4 sentences. "
                    f'Return JSON: {{"subject": "...", "body": "..."}}.'
                )
                obj = self.client.chat_json(prompt, temperature=temperature)
                subject, body = _subject_body(obj)
                records.append(
                    {
                        "subject": _clean(subject),
                        "body": _clean(body),
                        "text": f"{_clean(subject)}\n\n{_clean(body)}",
                        "category": category,
                        "tone": tone,
                        "kind": "ticket",
                    }
                )
        return self._dedup_grouped(records, group_key="category")

    # ---- dedup -----------------------------------------------------------
    def _dedup_grouped(
        self, records: List[Dict[str, Any]], group_key: str, method: str = "embedding", threshold: float = 0.92
    ) -> List[Dict[str, Any]]:
        """Dedup within each group so class balance is preserved. Silently
        falls back to lexical dedup if embeddings are unavailable."""
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for r in records:
            groups.setdefault(r.get(group_key), []).append(r)

        kept: List[Dict[str, Any]] = []
        for _, items in groups.items():
            kept.extend(self._dedup(items, method=method, threshold=threshold))
        return kept

    def _dedup(self, items: List[Dict[str, Any]], method: str, threshold: float) -> List[Dict[str, Any]]:
        if len(items) <= 1 or method == "none":
            return items
        texts = [it["text"] for it in items]
        if method == "embedding":
            try:
                vectors = self.client.embed(texts, input_type="passage")
                return _dedup_by_vectors(items, vectors, threshold)
            except Exception:
                # Embedding endpoint unavailable — degrade gracefully.
                pass
        return _dedup_lexical(items, threshold=min(threshold, 0.8))

    # ---- internals -------------------------------------------------------
    @staticmethod
    def _json_list(obj: Any) -> List[str]:
        if isinstance(obj, list):
            return [str(x) if not isinstance(x, dict) else str(next(iter(x.values()), "")) for x in obj]
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
        return [str(obj)]

    @staticmethod
    def _json_scalar(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list) and obj:
            return str(obj[0])
        if isinstance(obj, dict):
            for key in ("text", "review", "content", "value"):
                if key in obj:
                    return str(obj[key])
            return str(next(iter(obj.values()), ""))
        return str(obj)

    def _rating_for(self, sentiment: str) -> int:
        if sentiment == "positive":
            return self.rng.choice([4, 5, 5])
        if sentiment == "negative":
            return self.rng.choice([1, 1, 2])
        return 3


# --------------------------------------------------------------------------
# config-driven entry point (used by the CLI)
# --------------------------------------------------------------------------
def run_text_task(config: Dict[str, Any], client: Optional[NIMClient] = None) -> List[Dict[str, Any]]:
    task = str(config.get("task", "")).lower()
    factory = TextFactory(client=client, seed=int(config.get("seed", 7)))
    temperature = float(config.get("temperature", 0.9))

    if task == "paraphrase":
        return factory.paraphrase(
            seeds=config["seeds"],
            variations=int(config.get("variations", 5)),
            temperature=temperature,
        )
    if task == "classification":
        return factory.classification(
            labels=config["labels"],
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
            product=config["product"],
            sentiments=config.get("sentiments"),
            count=int(config.get("count", 30)),
            temperature=temperature,
        )
    if task == "tickets":
        return factory.tickets(
            categories=config["categories"],
            per_category=int(config.get("per_category", 8)),
            product=config.get("product", "our software product"),
            temperature=temperature,
        )
    raise ValueError(
        f"Unknown text task '{task}'. Expected one of: paraphrase, classification, "
        f"personas, reviews, tickets."
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


def _subject_body(obj: Any) -> tuple:
    if isinstance(obj, dict):
        return obj.get("subject", ""), obj.get("body", obj.get("message", ""))
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
    try:
        import numpy as np

        mat = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = mat / norms
        kept_idx: List[int] = []
        kept_vecs: List[Any] = []
        for i in range(len(items)):
            v = unit[i]
            if kept_vecs:
                sims = np.dot(np.vstack(kept_vecs), v)
                if float(sims.max()) >= threshold:
                    continue
            kept_idx.append(i)
            kept_vecs.append(v)
        return [items[i] for i in kept_idx]
    except ImportError:
        # Pure-Python cosine fallback.
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
