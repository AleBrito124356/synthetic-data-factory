"""Q&A / instruction-pair generation from seed documents.

Turn reference docs into supervised pairs for RAG evaluation or fine-tuning:

1. **Chunk** each document into overlapping windows.
2. **Generate questions** grounded in each chunk.
3. **Answer** each question using only that chunk.
4. **Hard negatives** — a plausible-but-wrong answer for the same question,
   useful for training rerankers / answer verifiers or as contrastive data.
5. **Self-critique filter** — the model scores each pair for groundedness,
   answerability, and clarity; weak items are dropped.

The result is a list of pair dicts that :mod:`factory.export` can write as
chat-format JSONL for SFT.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence

from .nim import NIMClient

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


class QAFactory:
    def __init__(self, client: Optional[NIMClient] = None, seed: int = 11):
        self.client = client or NIMClient()
        self.rng = random.Random(seed)

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
    ) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        for doc in docs:
            source = doc.get("source", "seed")
            for c_index, chunk in enumerate(_chunk_text(doc["text"], chunk_size, overlap)):
                questions = self._gen_questions(chunk, questions_per_chunk, temperature)
                for q in questions:
                    answer = self._gen_answer(chunk, q, temperature)
                    if _is_unanswerable(answer):
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

        return self._filter_by_quality(pairs, min_quality)

    # ---- steps -----------------------------------------------------------
    def _gen_questions(self, chunk: str, n: int, temperature: float) -> List[str]:
        prompt = (
            f"Read the passage and write {n} distinct questions that are fully "
            f"answerable from it alone. Prefer specific, non-trivial questions. "
            f"Return a JSON array of {n} strings.\n\nPassage:\n{chunk}"
        )
        obj = self.client.chat_json(prompt, system=_QGEN_SYSTEM, temperature=temperature)
        items = obj if isinstance(obj, list) else [obj]
        out: List[str] = []
        for q in items[:n]:
            text = q if isinstance(q, str) else str(q.get("question", q)) if isinstance(q, dict) else str(q)
            text = text.strip()
            if text:
                out.append(text)
        return out

    def _gen_answer(self, chunk: str, question: str, temperature: float) -> str:
        prompt = (
            f"Passage:\n{chunk}\n\nQuestion: {question}\n\n"
            f"Answer the question using only the passage, in 1-3 sentences. "
            f"If the passage does not contain the answer, reply exactly: "
            f"NOT_IN_PASSAGE."
        )
        return self.client.chat(prompt, system=_ANSWER_SYSTEM, temperature=min(temperature, 0.4)).strip()

    def _gen_hard_negative(self, chunk: str, question: str, answer: str, temperature: float) -> str:
        prompt = (
            f"Question: {question}\nCorrect answer: {answer}\n\n"
            f"Write a HARD NEGATIVE: an answer that sounds plausible and is on-topic "
            f"but is factually wrong or unsupported by this passage. Keep it similar in "
            f"length and style to the correct answer. Return just the wrong answer text "
            f"as a JSON string.\n\nPassage:\n{chunk}"
        )
        obj = self.client.chat_json(prompt, temperature=temperature)
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            return str(obj.get("answer", next(iter(obj.values()), ""))).strip()
        return str(obj).strip()

    def _filter_by_quality(self, pairs: List[Dict[str, Any]], min_quality: int) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for pair in pairs:
            scores = self._critique(pair)
            pair["quality"] = scores
            overall = min(scores.get("groundedness", 0), scores.get("answerability", 0), scores.get("clarity", 0))
            if overall >= min_quality:
                kept.append(pair)
        return kept

    def _critique(self, pair: Dict[str, Any]) -> Dict[str, int]:
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
        except Exception:
            return {"groundedness": 0, "answerability": 0, "clarity": 0}
        return {
            "groundedness": _score(obj, "groundedness"),
            "answerability": _score(obj, "answerability"),
            "clarity": _score(obj, "clarity"),
        }


# --------------------------------------------------------------------------
# config-driven entry point (used by the CLI)
# --------------------------------------------------------------------------
def run_qa_task(config: Dict[str, Any], client: Optional[NIMClient] = None) -> List[Dict[str, Any]]:
    factory = QAFactory(client=client, seed=int(config.get("seed", 11)))
    docs = _resolve_docs(config.get("docs", []), base_dir=config.get("_base_dir", "."))
    if not docs:
        raise ValueError("The qa task needs at least one document under 'docs'.")
    return factory.generate(
        docs=docs,
        questions_per_chunk=int(config.get("questions_per_chunk", 2)),
        chunk_size=int(config.get("chunk_size", 700)),
        overlap=int(config.get("overlap", 100)),
        hard_negatives=bool(config.get("hard_negatives", True)),
        min_quality=int(config.get("min_quality", 4)),
        temperature=float(config.get("temperature", 0.7)),
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
    return "NOT_IN_PASSAGE" in (answer or "").upper()


def _score(obj: Any, key: str) -> int:
    try:
        if isinstance(obj, dict):
            return int(round(float(obj.get(key, 0))))
    except (TypeError, ValueError):
        return 0
    return 0
