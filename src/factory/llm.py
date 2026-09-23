"""Provider-neutral LLM plumbing: the client protocol, JSON-shape
normalization, and record/replay cassettes.

Record/replay
-------------
:class:`RecordingClient` wraps any chat client and appends every ``chat`` and
``embed`` request, with its response, to a JSONL *cassette*.
:class:`ReplayClient` serves those responses back — no key, no network — keyed
by a hash of the request, and raises :class:`CassetteMissError` for any
request that was not recorded. Identical requests are served in the order
they were recorded, so a pipeline that sends the same prompt twice replays
both answers faithfully.

That makes LLM dataset builds reproducible: record once against NIM, commit
the cassette, and every later ``--replay`` run (in CI, on a plane, on a
colleague's machine) produces byte-identical output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

try:  # Python 3.8+: typing.Protocol
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

JSON_INSTRUCTION = "Respond with valid JSON only, no prose, no code fences."
CASSETTE_VERSION = 1


class LLMError(RuntimeError):
    """Base class for errors raised by the LLM layer."""


class ModelJSONError(LLMError):
    """The model reply did not contain parseable JSON."""


class CassetteMissError(LLMError):
    """A replayed run sent a request the cassette does not contain."""


class ReplayedError(LLMError):
    """A request that failed while recording fails the same way on replay."""


class ChatClient(Protocol):
    """What the text and qa generators need from a model client."""

    def chat(self, prompt: str, system: Optional[str] = None, temperature: float = 0.8,
             max_tokens: int = 1024) -> str: ...

    def chat_json(self, prompt: str, system: Optional[str] = None, temperature: float = 0.8,
                  max_tokens: int = 1024) -> Any: ...

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> List[List[float]]: ...


def json_system(system: Optional[str]) -> str:
    """System prompt used for JSON-mode requests."""
    return f"{system.strip()} {JSON_INSTRUCTION}" if system and system.strip() else JSON_INSTRUCTION


class JSONChatMixin:
    """``chat_json`` implemented on top of ``chat`` + :func:`extract_json`."""

    def chat_json(self, prompt: str, system: Optional[str] = None, temperature: float = 0.8,
                  max_tokens: int = 1024) -> Any:
        text = self.chat(prompt, system=json_system(system), temperature=temperature,  # type: ignore[attr-defined]
                         max_tokens=max_tokens)
        return extract_json(text)


# --------------------------------------------------------------------------
# JSON extraction and shape normalization
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model reply.

    Handles bare JSON, Markdown code fences (anywhere in the reply), and JSON
    embedded in prose ("Sure! Here you go: [...] Hope it helps"). The first
    ``[`` or ``{`` that starts a complete JSON value wins, so an object that
    *contains* an array is returned whole instead of losing its wrapper.
    """
    if not isinstance(text, str):
        raise ModelJSONError(f"Expected text from the model, got {type(text).__name__}.")
    stripped = text.strip()
    candidates = [stripped]
    fence = _FENCE.search(stripped)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
    decoder = json.JSONDecoder()
    for candidate in candidates:
        tried = 0
        for idx, ch in enumerate(candidate):
            if ch not in "[{":
                continue
            tried += 1
            if tried > 200:
                break
            try:
                value, _ = decoder.raw_decode(candidate, idx)
                return value
            except (json.JSONDecodeError, ValueError):
                continue
    raise ModelJSONError(f"Model did not return valid JSON. Raw reply:\n{stripped[:500]}")


_TEXT_KEYS = ("text", "content", "value", "message", "body", "review", "question",
              "answer", "example", "sentence", "bio", "persona", "paraphrase")
_LIST_KEYS = ("items", "examples", "data", "results", "questions", "reviews", "texts",
              "sentences", "paraphrases", "personas", "bios", "messages", "list", "output")


def as_text(obj: Any, keys: Sequence[str] = ()) -> str:
    """Best-effort single string from a JSON value.

    ``"x"`` -> ``x``; ``["x", ...]`` -> ``x``; ``{"review": "x"}`` -> ``x``
    (preferred ``keys`` first, then common text keys, then the first string
    value).
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        return as_text(obj[0], keys) if obj else ""
    if isinstance(obj, dict):
        for key in list(keys) + list(_TEXT_KEYS):
            if key in obj and isinstance(obj[key], (str, int, float)):
                return str(obj[key])
        for value in obj.values():
            if isinstance(value, str):
                return value
        for value in obj.values():
            if isinstance(value, (list, dict)):
                inner = as_text(value, keys)
                if inner:
                    return inner
        return ""
    return str(obj)


def as_text_list(obj: Any, keys: Sequence[str] = ()) -> List[str]:
    """Best-effort list of strings from a JSON value.

    Accepts the shapes chat models actually produce:

    * ``["a", "b"]``
    * ``[{"text": "a"}, {"question": "b"}]``
    * ``{"questions": ["a", "b"]}`` / ``{"items": [...]}`` (object-wrapped)
    * ``{"1": "a", "2": "b"}`` (numbered object)
    * ``"a"`` (a single string)
    """
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, list):
        out = [as_text(x, keys) for x in obj]
        return [s for s in out if s and s.strip()]
    if isinstance(obj, dict):
        for key in list(keys) + list(_LIST_KEYS):
            if isinstance(obj.get(key), list):
                return as_text_list(obj[key], keys)
        lists = [v for v in obj.values() if isinstance(v, list)]
        if len(lists) == 1:
            return as_text_list(lists[0], keys)
        if obj and all(str(k).strip().isdigit() for k in obj) \
                and all(isinstance(v, str) for v in obj.values()):
            return [v for v in obj.values() if v.strip()]  # {"1": "a", "2": "b"}
        single = as_text(obj, keys)
        return [single] if single.strip() else []
    return [str(obj)]


# --------------------------------------------------------------------------
# record / replay
# --------------------------------------------------------------------------
def request_key(op: str, payload: Dict[str, Any]) -> str:
    """Stable hash of a request (operation + canonical JSON payload)."""
    canonical = json.dumps({"op": op, **payload}, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _chat_payload(prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> Dict[str, Any]:
    return {"prompt": prompt, "system": system or "", "temperature": float(temperature),
            "max_tokens": int(max_tokens)}


def _embed_payload(texts: Sequence[str], input_type: str) -> Dict[str, Any]:
    return {"texts": list(texts), "input_type": input_type}


def _round_vectors(vectors: Iterable[Iterable[float]], ndigits: int = 6) -> List[List[float]]:
    return [[round(float(x), ndigits) for x in vec] for vec in vectors]


class RecordingClient(JSONChatMixin):
    """Wrap ``inner`` and append every request/response to ``path``.

    The cassette is truncated when the recorder is created and flushed after
    every call, so an interrupted run still leaves a usable prefix. Embedding
    vectors are rounded to 6 decimals *before* they are returned, so the
    recording run and every replay see exactly the same numbers.
    """

    def __init__(self, inner: Any, path: str):
        self.inner = inner
        self.path = path
        self.calls = 0
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"cassette": CASSETTE_VERSION,
                                 "model": getattr(inner, "model", None),
                                 "embed_model": getattr(inner, "embed_model", None)}) + "\n")

    def _append(self, op: str, payload: Dict[str, Any], response: Any = None,
                error: Optional[BaseException] = None) -> None:
        entry: Dict[str, Any] = {"key": request_key(op, payload), "op": op, "request": payload}
        if error is not None:
            entry["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        else:
            entry["response"] = response
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.calls += 1

    def chat(self, prompt: str, system: Optional[str] = None, temperature: float = 0.8,
             max_tokens: int = 1024) -> str:
        payload = _chat_payload(prompt, system, temperature, max_tokens)
        try:
            text = self.inner.chat(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            self._append("chat", payload, error=exc)
            raise
        self._append("chat", payload, response=text)
        return text

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> List[List[float]]:
        payload = _embed_payload(texts, input_type)
        try:
            vectors = _round_vectors(self.inner.embed(list(texts), input_type=input_type))
        except Exception as exc:
            self._append("embed", payload, error=exc)
            raise
        self._append("embed", payload, response=vectors)
        return vectors


class ReplayClient(JSONChatMixin):
    """Serve responses from a cassette written by :class:`RecordingClient`."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise CassetteMissError(f"Cassette not found: {path}")
        self.path = path
        self.model: Optional[str] = None
        self.embed_model: Optional[str] = None
        self.calls = 0
        self._entries: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LLMError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
                if "cassette" in entry:
                    self.model = entry.get("model")
                    self.embed_model = entry.get("embed_model")
                    continue
                self._entries[entry["key"]].append(entry)

    def remaining(self) -> int:
        """Recorded responses not consumed yet."""
        return sum(len(q) for q in self._entries.values())

    def _serve(self, op: str, payload: Dict[str, Any]) -> Any:
        key = request_key(op, payload)
        queue = self._entries.get(key)
        if not queue:
            hint = payload.get("prompt") or " | ".join(payload.get("texts", [])[:2])
            raise CassetteMissError(
                f"{self.path} has no recorded response for this {op} request "
                f"(key {key[:12]}, starts {str(hint)[:120]!r}). The task config or "
                f"prompts changed since recording; re-run with --record to refresh it."
            )
        entry = queue.popleft()
        self.calls += 1
        if "error" in entry:
            err = entry["error"]
            raise ReplayedError(f"(replayed) {err.get('type')}: {err.get('message')}")
        return entry["response"]

    def chat(self, prompt: str, system: Optional[str] = None, temperature: float = 0.8,
             max_tokens: int = 1024) -> str:
        return self._serve("chat", _chat_payload(prompt, system, temperature, max_tokens))

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> List[List[float]]:
        return self._serve("embed", _embed_payload(texts, input_type))
