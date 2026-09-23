"""Shared fixtures: an offline OpenAI-compatible server and a key-less env."""
import pytest

from tests.fake_llm import FakeOpenAIServer


@pytest.fixture
def fake_server():
    server = FakeOpenAIServer().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    """No NVIDIA_API_KEY and no ./.env: run from an empty temp directory.

    setenv-then-delenv makes monkeypatch remember the key was absent, so
    anything a test loads into os.environ is removed again afterwards.
    """
    for var in ("NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MODEL", "NIM_EMBED_MODEL"):
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)
    monkeypatch.chdir(tmp_path)
    return tmp_path
