"""Offline stand-ins for the model: a scripted in-process client and a local
OpenAI-compatible HTTP server. No key, no network beyond 127.0.0.1."""
from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import struct
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Dict, List, Optional

from factory.llm import JSONChatMixin


def fake_vector(text: str, dim: int = 64) -> List[float]:
    """Deterministic pseudo-embedding: identical (normalized) texts map to
    identical vectors, different texts to nearly orthogonal ones."""
    norm = re.sub(r"\s+", " ", text).strip().casefold()
    rng = random.Random(hashlib.sha256(norm.encode("utf-8")).hexdigest())
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


class SmartResponder:
    """Answers each prompt type the text/qa pipelines send, with distinct,
    deterministic content (a per-instance counter keeps items unique)."""

    def __init__(self):
        self.counter = 0

    def _next(self) -> int:
        self.counter += 1
        return self.counter

    def __call__(self, system: str, prompt: str) -> str:
        m = re.search(r"(?:Generate|write|Write|in) (\d+) ", prompt)
        n = int(m.group(1)) if m else 3
        if "product review" in prompt:
            sentiment = re.search(r"realistic (\w+) product review", prompt).group(1)
            aspect = re.search(r"focusing partly on ([^.]+)\.", prompt).group(1)
            return json.dumps(f"Review {self._next()}: a {sentiment} take on the {aspect}.")
        if "customer support ticket" in prompt:
            cat = re.search(r"category is '([^']+)'", prompt).group(1)
            k = self._next()
            return json.dumps({"subject": f"{cat} issue #{k}", "body": f"Details about {cat} problem {k}."})
        if prompt.startswith("Rewrite the following sentence"):
            return json.dumps([f"Variant {self._next()} of the sentence" for _ in range(n)])
        if prompt.startswith("Generate") and "category" in prompt:
            label = re.search(r"category '([^']+)'", prompt).group(1)
            return "```json\n" + json.dumps([f"{label} message {self._next()}" for _ in range(n)]) + "\n```"
        if "persona bios" in prompt:
            return json.dumps({"personas": [f"Persona {self._next()} is a tester." for _ in range(n)]})
        if "distinct questions" in prompt:
            # Object-wrapped array: the shape that used to break the QA step.
            words = prompt.split("Passage:\n", 1)[1].split()
            return json.dumps({"questions": [f"What does the passage say about '{words[i % len(words)]}'?"
                                             for i in range(n)]})
        if "Answer the question using only the passage" in prompt:
            if "unanswerable" in prompt:
                return "NOT_IN_PASSAGE"
            return f"According to the passage, answer {self._next()}."
        if "HARD NEGATIVE" in prompt:
            return json.dumps(f"A plausible but wrong answer {self._next()}.")
        if "Score this question-answer pair" in prompt:
            low = "lowquality" in prompt
            return json.dumps({"groundedness": 2 if low else 5, "answerability": 5, "clarity": 4})
        return json.dumps(f"generic reply {self._next()}")


class ScriptedClient(JSONChatMixin):
    """In-process ChatClient. ``responder(system, prompt) -> str`` supplies
    raw model text; ``chat_json`` parses it like the real client does."""

    def __init__(self, responder: Optional[Callable[[str, str], str]] = None,
                 embed_fn: Optional[Callable[[str], List[float]]] = None,
                 embed_error: Optional[Exception] = None):
        self.responder = responder or SmartResponder()
        self.embed_fn = embed_fn or fake_vector
        self.embed_error = embed_error
        self.chat_calls: List[Dict[str, Any]] = []
        self.embed_calls: List[List[str]] = []
        self.model = "scripted"

    def chat(self, prompt: str, system: Optional[str] = None, temperature: float = 0.8,
             max_tokens: int = 1024) -> str:
        self.chat_calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return self.responder(system or "", prompt)

    def embed(self, texts, input_type: str = "passage"):
        self.embed_calls.append(list(texts))
        if self.embed_error is not None:
            raise self.embed_error
        return [self.embed_fn(t) for t in texts]


# --------------------------------------------------------------------------
# local OpenAI-compatible HTTP server
# --------------------------------------------------------------------------
class FakeOpenAIServer:
    """Serves ``POST /v1/chat/completions`` and ``POST /v1/embeddings`` on
    127.0.0.1 with a random port. ``fail_with`` queues HTTP status codes to
    return (one per request) before answering normally."""

    def __init__(self, responder: Optional[Callable[[str, str], str]] = None, dim: int = 64):
        self.responder = responder or SmartResponder()
        self.dim = dim
        self.requests: List[Dict[str, Any]] = []
        self.fail_with: Deque[int] = deque()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def count(self, kind: str) -> int:
        return sum(1 for r in self.requests if r["path"].endswith(kind))

    def start(self) -> "FakeOpenAIServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, status: int, payload: Dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.requests.append({"path": self.path, "body": body,
                                            "auth": self.headers.get("Authorization")})
                    status = server.fail_with.popleft() if server.fail_with else 200
                if status != 200:
                    self._send(status, {"error": {"message": f"fake error {status}", "type": "fake",
                                                  "code": status}})
                    return
                if self.path.endswith("/chat/completions"):
                    msgs = body.get("messages", [])
                    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
                    user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
                    with server._lock:
                        content = server.responder(system, user)
                    self._send(200, {
                        "id": "chatcmpl-fake", "object": "chat.completion", "created": 0,
                        "model": body.get("model", "fake"),
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": content}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    })
                elif self.path.endswith("/embeddings"):
                    inputs = body.get("input", [])
                    inputs = [inputs] if isinstance(inputs, str) else inputs
                    data = []
                    for i, text in enumerate(inputs):
                        vec = fake_vector(text, server.dim)
                        if body.get("encoding_format") == "base64":
                            vec = base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")
                        data.append({"object": "embedding", "index": i, "embedding": vec})
                    # Return out of order to prove the client re-sorts by index.
                    data.reverse()
                    self._send(200, {"object": "list", "data": data, "model": body.get("model", "fake"),
                                     "usage": {"prompt_tokens": 1, "total_tokens": 1}})
                else:
                    self._send(404, {"error": {"message": "not found"}})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
