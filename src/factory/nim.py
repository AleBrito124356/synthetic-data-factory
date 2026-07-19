"""Thin OpenAI-compatible client for NVIDIA NIM.

Used by the ``text`` and ``qa`` generators. The tabular path never imports
this module, so the zero-dollar workflow has no dependency on ``openai`` or a
network connection.

Get a free key at https://build.nvidia.com — pick any model, click "Get API
Key". Keys start with ``nvapi-``. Set ``NVIDIA_API_KEY`` in your environment or
in a ``.env`` file.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CHAT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

_MISSING_KEY_MESSAGE = """
NVIDIA_API_KEY is not set - the text/qa generators need it.

It is free and takes about two minutes:
  1. Go to https://build.nvidia.com and sign in.
  2. Open any model, click "Get API Key", copy the key that starts with nvapi-.
  3. Add it to your environment:
        export NVIDIA_API_KEY=nvapi-your-key-here      (macOS/Linux)
        setx  NVIDIA_API_KEY nvapi-your-key-here        (Windows)
     ...or copy .env.example to .env and paste it there.

The tabular generator does NOT need a key. Try:
  python cli.py generate tabular --schema schemas/ecommerce.yaml --out results/
""".strip()


class NIMError(RuntimeError):
    pass


def _load_dotenv_if_present() -> None:
    """Best-effort .env load without hard-depending on python-dotenv."""
    if os.environ.get("NVIDIA_API_KEY"):
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        # Minimal fallback parser so a bare `.env` still works.
        path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def require_api_key() -> str:
    _load_dotenv_if_present()
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key or key.startswith("nvapi-XXXX"):
        print(_MISSING_KEY_MESSAGE, file=sys.stderr)
        sys.exit(2)
    return key


class NIMClient:
    """Chat + embeddings over the NIM OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ):
        _load_dotenv_if_present()
        self.api_key = api_key or require_api_key()
        self.base_url = base_url or os.environ.get("NIM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("NIM_MODEL", DEFAULT_CHAT_MODEL)
        self.embed_model = embed_model or os.environ.get("NIM_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise NIMError(
                "The 'openai' package is required for the text/qa generators. "
                "Install it with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ---- chat ------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_messages(
            messages, temperature=temperature, max_tokens=max_tokens, retries=retries
        )

    def chat_messages(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # network/rate-limit/etc.
                last_err = exc
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise NIMError(f"Chat request failed after {retries} attempts: {last_err}")

    def chat_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> Any:
        """Chat and parse the reply as JSON, tolerating code fences and prose."""
        text = self.chat(
            prompt,
            system=(system or "") + " Respond with valid JSON only, no prose, no code fences.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_json(text)

    # ---- embeddings ------------------------------------------------------
    def embed(self, texts: Sequence[str], input_type: str = "passage", batch_size: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        items = list(texts)
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            resp = self._client.embeddings.create(
                model=self.embed_model,
                input=batch,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            # Preserve request order.
            for item in sorted(resp.data, key=lambda d: d.index):
                out.append(list(item.embedding))
        return out


def _extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply."""
    text = text.strip()
    # Strip Markdown code fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: locate the outermost bracketed span.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        first = text.find(open_ch)
        last = text.rfind(close_ch)
        if first != -1 and last != -1 and last > first:
            snippet = text[first : last + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    raise NIMError(f"Model did not return valid JSON. Raw reply:\n{text[:500]}")
