"""Thin OpenAI-compatible client for NVIDIA NIM.

Used by the ``text`` and ``qa`` generators. The tabular path never imports
this module, so the zero-dollar workflow has no dependency on ``openai`` or a
network connection.

Get a free key at https://build.nvidia.com — pick any model, click "Get API
Key". Keys start with ``nvapi-``. Set ``NVIDIA_API_KEY`` in your environment or
in a ``.env`` file in the current directory.

Library code never exits the interpreter: a missing key raises
:class:`MissingAPIKeyError` (the CLI turns it into a friendly message and exit
code 2).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from .llm import JSONChatMixin, LLMError, ModelJSONError, extract_json

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CHAT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

MISSING_KEY_MESSAGE = """
NVIDIA_API_KEY is not set - the text/qa generators need it.

It is free and takes about two minutes:
  1. Go to https://build.nvidia.com and sign in.
  2. Open any model, click "Get API Key", copy the key that starts with nvapi-.
  3. Add it to your environment:
        export NVIDIA_API_KEY=nvapi-your-key-here      (macOS/Linux)
        setx  NVIDIA_API_KEY nvapi-your-key-here        (Windows)
     ...or copy .env.example to .env and paste it there.

No key? You can still:
  - preview the requests a task would send:   sdf generate text --task T.yaml --out o.jsonl --dry-run
  - rebuild a dataset from a recorded run:     sdf generate text --task T.yaml --out o.jsonl --replay run.cassette.jsonl
  - generate tabular data (needs no key):      sdf generate tabular --schema schemas/ecommerce.yaml --out results/
""".strip()

# Backwards-compatible name.
_MISSING_KEY_MESSAGE = MISSING_KEY_MESSAGE


class NIMError(LLMError):
    """A NIM request failed (after retries) or returned unusable output."""


class MissingAPIKeyError(NIMError):
    """``NVIDIA_API_KEY`` is not configured."""

    def __init__(self, message: str = MISSING_KEY_MESSAGE):
        super().__init__(message)


def _load_dotenv_if_present() -> None:
    """Load ``./.env`` (current directory only) if the key is not already set.

    Deliberately does not walk up parent directories: which file configures
    the run should be obvious from where you launched it.
    """
    if os.environ.get("NVIDIA_API_KEY"):
        return
    path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(path):
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=path, override=False)
        return
    except ImportError:
        pass
    # Minimal fallback parser so a bare `.env` still works without python-dotenv.
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_api_key() -> str:
    """Return the configured key or raise :class:`MissingAPIKeyError`."""
    _load_dotenv_if_present()
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key or key.startswith("nvapi-XXXX"):
        raise MissingAPIKeyError()
    return key


def _is_retryable(exc: BaseException) -> bool:
    """Rate limits, timeouts, connection problems and 5xx are worth a retry;
    auth errors and bad requests are not."""
    try:
        import openai  # type: ignore
    except ImportError:  # pragma: no cover
        return True
    retryable = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                 openai.InternalServerError)
    if isinstance(exc, retryable):
        return True
    if isinstance(exc, openai.APIStatusError):
        return getattr(exc, "status_code", 0) >= 500
    return not isinstance(exc, openai.OpenAIError)


class NIMClient(JSONChatMixin):
    """Chat + embeddings over the NIM OpenAI-compatible endpoint.

    Retries are handled here (exponential backoff on 429/5xx/timeouts, no
    retry on 4xx such as a bad key), so the SDK's own retry loop is disabled
    to avoid multiplying attempts.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        embed_model: Optional[str] = None,
        retries: int = 3,
        retry_backoff: float = 1.5,
        timeout: float = 60.0,
    ):
        if not api_key:
            api_key = require_api_key()  # also loads ./.env for the settings below
        self.api_key = api_key
        self.base_url = base_url or os.environ.get("NIM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("NIM_MODEL", DEFAULT_CHAT_MODEL)
        self.embed_model = embed_model or os.environ.get("NIM_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.retries = max(1, int(retries))
        self.retry_backoff = float(retry_backoff)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise NIMError(
                "The 'openai' package is required for the text/qa generators. "
                "Install it with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                              max_retries=0, timeout=timeout)

    # ---- retry helper ----------------------------------------------------
    def _with_retries(self, what: str, fn):
        last_err: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                return fn()
            except Exception as exc:  # network/rate-limit/etc.
                last_err = exc
                if not _is_retryable(exc) or attempt == self.retries - 1:
                    break
                time.sleep(self.retry_backoff * (2 ** attempt))
        raise NIMError(f"{what} failed after {attempt + 1} attempt(s): {last_err}") from last_err

    # ---- chat ------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        retries: Optional[int] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_messages(messages, temperature=temperature, max_tokens=max_tokens,
                                  retries=retries)

    def chat_messages(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 1024,
        retries: Optional[int] = None,
    ) -> str:
        if retries is not None:
            saved, self.retries = self.retries, max(1, int(retries))
        try:
            def call():
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()

            return self._with_retries("Chat request", call)
        finally:
            if retries is not None:
                self.retries = saved

    # ---- embeddings ------------------------------------------------------
    def embed(self, texts: Sequence[str], input_type: str = "passage", batch_size: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        items = list(texts)
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]

            def call(batch=batch):
                return self._client.embeddings.create(
                    model=self.embed_model,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "END"},
                )

            resp = self._with_retries("Embedding request", call)
            # Preserve request order.
            for item in sorted(resp.data, key=lambda d: d.index):
                out.append(list(item.embedding))
        return out


def _extract_json(text: str) -> Any:
    """Backwards-compatible wrapper around :func:`factory.llm.extract_json`
    that raises :class:`NIMError` like the original helper."""
    try:
        return extract_json(text)
    except ModelJSONError as exc:
        raise NIMError(str(exc)) from exc
