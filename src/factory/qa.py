"""Q&A / instruction-pair generation from seed documents.

Turn reference docs into supervised pairs for RAG evaluation or fine-tuning:

1. **Chunk** each document into overlapping windows.
2. **Generate questions** grounded in each chunk.
3. **Answer** each question using only that chunk (``NOT_IN_PASSAGE`` answers
   are skipped).
4. **Hard negatives** — a plausible-but-wrong answer for the same question,
   useful for training rerankers / answer verifiers or as contrastive data.
5. **Self-critique filter** — the model scores each pair for groundedness,
   answerability, and clarity; weak items are dropped.

The result is a :class:`QAResult` (a list of pair dicts plus counters) that
:mod:`factory.export` can write as chat-format JSONL for SFT.
"""
from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence

from .llm import CassetteMissError, LLMError, as_text, as_text_list

_QGEN_SYSTEM = (
    "You are a meticulous dataset author. You write clear, self-contained "
    "questions that can be answered strictly from a given passage."
)
_ANSWER_SYSTEM = (
    "You answer strictly and only from the provided passage. If the passage "
    "does not contain the answer, say so plainly."
)
_CRITIQUE_SYSTEM = (
    "You are a strict data-quality reviewer. You score question-answer pairs "
    "and never inflate scores."
)
QUALITY_AXES = ("groundedness", "answerability", "clarity")


class QAResult(list):
    """Kept pairs plus what happened to the others."""

    def __init__(self, records: Sequence[Dict[str, Any]] = (), chunks: int = 0,
                 candidates: int = 0, unanswerable: int = 0, low_quality: int = 0,
                 duplicates: int = 0, requests: int = 0):
        super().__init__(records)
        self.chunks = chunks
        self.candidates = candidates
        self.unanswerable = unanswerable
        self.low_quality = low_quality
        self.duplicates = duplicates
        self.requests = requests


class QAFactory:
    def __init__(self, client: Optional[Any] = None, seed: int = 11):
        if client is None:
            from .nim import NIMClient  # raises MissingAPIKeyError without a key

            client = NIMClient()
        self.client = client
        self.rng = random.Random(seed)
        self._requests = 0

    # ---- public ----------------------------------------------------------
    def generate(
        self,
        docs: Sequence[Dict[str, str]],
        questions_per_chunk: int = 2,
        chunk_size: int = 700,
        overlap: int = 100,
        hard_negatives: bool = True,
        min_quality: int = 4,
        temperature: float = 0.7,
    ) -> QAResult:
        pairs: List[Dict[str, Any]] = []
        chunks = 0
        candidates = 0
        unanswerable = 0
        duplicates = 0
        seen_questions: set = set()
        start_requests = self._requests
        for doc in docs:
            source = doc.get("source", "seed")
            for c_index, chunk in enumerate(_chunk_text(doc["text"], chunk_size, overlap)):
                chunks += 1
                for q in self._gen_questions(chunk, questions_per_chunk, temperature):
                    candidates += 1
                    key = _normalize(q)
                    if key in seen_questions:
                        duplicates += 1  # overlapping chunks often repeat a question
                        continue
                    seen_questions.add(key)
                    answer = self._gen_answer(chunk, q, temperature)
                    if not answer or _is_unanswerable(answer):
                        unanswerable += 1
                        continue
                    record: Dict[str, Any] = {
                        "question": q,
                        "answer": answer,
                        "context": chunk,
                        "source": source,
                        "chunk_index": c_index,
                    }
                    if hard_negatives:
                        record["hard_negative"] = self._gen_hard_negative(chunk, q, answer, temperature)
                    pairs.append(record)

        kept = self._filter_by_quality(pairs, min_quality)
        return QAResult(kept, chunks=chunks, candidates=candidates, unanswerable=unanswerable,
                        low_quality=len(pairs) - len(kept), duplicates=duplicates,
                        requests=self._requests - start_requests)

    # ---- steps -----------------------------------------------------------
    def _gen_questions(self, chunk: str, n: int, temperature: float) -> List[str]:
        self._requests += 1
        obj = self.client.chat_json(question_prompt(chunk, n), system=_QGEN_SYSTEM, temperature=temperature)
        out: List[str] = []
        for q in as_text_list(obj, keys=("questions", "question")):
            text = q.strip()
            if text and text not in out:
                out.append(text)
        return out[:n]

    def _gen_answer(self, chunk: str, question: str, temperature: float) -> str:
        self._requests += 1
        prompt = (
            f"Passage:\n{chunk}\n\nQuestion: {question}\n\n"
            f"Answer the question using only the passage, in 1-3 sentences. "
            f"If the passage does not contain the answer, reply exactly: "
            f"NOT_IN_PASSAGE."
        )
        return self.client.chat(prompt, system=_ANSWER_SYSTEM, temperature=min(temperature, 0.4)).strip()

    def _gen_hard_negative(self, chunk: str, question: str, answer: str, temperature: float) -> str:
        self._requests += 1
        prompt = (
            f"Question: {question}\nCorrect answer: {answer}\n\n"
            f"Write a HARD NEGATIVE: an answer that sounds plausible and is on-topic "
            f"but is factually wrong or unsupported by this passage. Keep it similar in "
            f"length and style to the correct answer. Return just the wrong answer text "
            f"as a JSON string.\n\nPassage:\n{chunk}"
        )
        obj = self.client.chat_json(prompt, temperature=temperature)
        return as_text(obj, keys=("hard_negative", "wrong_answer", "answer")).strip()

    def _filter_by_quality(self, pairs: List[Dict[str, Any]], min_quality: int) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for pair in pairs:
            scores = self._critique(pair)
            pair["quality"] = scores
            if min(scores.get(axis, 0) for axis in QUALITY_AXES) >= min_quality:
                kept.append(pair)
        return kept

    def _critique(self, pair: Dict[str, Any]) -> Dict[str, int]:
        self._requests += 1
        prompt = (
            "Score this question-answer pair from 1 (poor) to 5 (excellent) on three axes:\n"
            "- groundedness: is the answer fully supported by the passage?\n"
            "- answerability: is the question answerable from the passage alone and unambiguous?\n"
            "- clarity: is the question well-formed and the answer clear?\n\n"
            f"Passage:\n{pair['context']}\n\nQuestion: {pair['question']}\nAnswer: {pair['answer']}\n\n"
            'Return JSON: {"groundedness": int, "answerability": int, "clarity": int}.'
        )
        try:
            obj = self.client.chat_json(prompt, system=_CRITIQUE_SYSTEM, temperature=0.0)
        except CassetteMissError:
            raise
        except LLMError:
            # An unparseable or failed critique scores zero: the pair is dropped.
            return {axis: 0 for axis in QUALITY_AXES}
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            obj = obj["scores"]
        return {axis: _score(obj, axis) for axis in QUALITY_AXES}


def question_prompt(chunk: str, n: int) -> str:
    return (
        f"Read the passage and write {n} distinct questions that are fully "
        f"answerable from it alone. Prefer specific, non-trivial questions. "
        f"Return a JSON array of {n} strings.\n\nPassage:\n{chunk}"
    )


# --------------------------------------------------------------------------
# config-driven entry point (used by the CLI)
# --------------------------------------------------------------------------
def _qa_options(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "questions_per_chunk": int(config.get("questions_per_chunk", 2)),
        "chunk_size": int(config.get("chunk_size", 700)),
        "overlap": int(config.get("overlap", 100)),
        "hard_negatives": bool(config.get("hard_negatives", True)),
        "min_quality": int(config.get("min_quality", 4)),
        "temperature": float(config.get("temperature", 0.7)),
    }


def run_qa_task(config: Dict[str, Any], client: Optional[Any] = None) -> QAResult:
    docs = _resolve_docs(config.get("docs", []), base_dir=config.get("_base_dir", "."))
    if not docs:
        raise ValueError("The qa task needs at least one document under 'docs'.")
    factory = QAFactory(client=client, seed=int(config.get("seed", 11)))
    return factory.generate(docs=docs, **_qa_options(config))


def plan_qa_task(config: Dict[str, Any]):
    """Count requests and build the first prompt without calling the model."""
    from .llm import json_system
    from .text import RequestPlan

    docs = _resolve_docs(config.get("docs", []), base_dir=config.get("_base_dir", "."))
    if not docs:
        raise ValueError("The qa task needs at least one document under 'docs'.")
    opts = _qa_options(config)
    chunks = [c for d in docs for c in _chunk_text(d["text"], opts["chunk_size"], opts["overlap"])]
    per_question = 2 + (1 if opts["hard_negatives"] else 0)  # answer + critique (+ hard negative)
    q = opts["questions_per_chunk"]
    return RequestPlan(
        task="qa",
        chat_requests=len(chunks) * (1 + q * per_question),
        embed_requests=0,
        groups={},
        first_prompt=question_prompt(chunks[0], q) if chunks else "(no text in docs)",
        first_system=json_system(_QGEN_SYSTEM),
        notes=[
            f"{len(docs)} document(s) -> {len(chunks)} chunk(s) of up to {opts['chunk_size']} words",
            f"upper bound: questions answered NOT_IN_PASSAGE skip their hard-negative and critique calls",
        ],
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resolve_docs(raw_docs: Sequence[Any], base_dir: str = ".") -> List[Dict[str, str]]:
    """Accept docs as plain strings, {text: ...}, or {path: ...}."""
    resolved: List[Dict[str, str]] = []
    for i, item in enumerate(raw_docs):
        if isinstance(item, str):
            resolved.append({"text": item, "source": f"inline-{i}"})
        elif isinstance(item, dict) and "text" in item:
            resolved.append({"text": str(item["text"]), "source": item.get("source", f"inline-{i}")})
        elif isinstance(item, dict) and "path" in item:
            path = item["path"]
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            with open(path, "r", encoding="utf-8") as fh:
                resolved.append({"text": fh.read(), "source": os.path.basename(path)})
        else:
            raise ValueError(f"docs[{i}] must be a string, {{text: ...}}, or {{path: ...}}.")
    return resolved


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Word-based sliding window. ``chunk_size`` and ``overlap`` are in words."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _is_unanswerable(answer: str) -> bool:
    return "NOT_IN_PASSAGE" in (answer or "").upper().replace(" ", "_")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _score(obj: Any, key: str) -> int:
    try:
        if isinstance(obj, dict):
            value = int(round(float(obj.get(key, 0))))
            return max(0, min(5, value))
    except (TypeError, ValueError):
        return 0
    return 0
