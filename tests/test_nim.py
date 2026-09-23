"""The real NIMClient, driven through the openai SDK against a local fake
OpenAI-compatible server. No key, no network beyond 127.0.0.1."""
import subprocess
import sys

import pytest

from factory.llm import ModelJSONError, as_text, as_text_list, extract_json
from factory.nim import MissingAPIKeyError, NIMClient, NIMError, _extract_json, require_api_key


def _client(server, **kw):
    kw.setdefault("retry_backoff", 0)
    return NIMClient(api_key="test-key", base_url=server.url, model="fake/chat",
                     embed_model="fake/embed", **kw)


def test_chat_round_trip(fake_server):
    fake_server.responder = lambda system, prompt: f"echo: {prompt}"
    reply = _client(fake_server).chat("hello", system="be brief", temperature=0.3, max_tokens=50)
    assert reply == "echo: hello"
    req = fake_server.requests[-1]
    assert req["path"].endswith("/chat/completions")
    assert req["auth"] == "Bearer test-key"
    assert req["body"]["model"] == "fake/chat"
    assert req["body"]["temperature"] == 0.3
    assert req["body"]["max_tokens"] == 50
    assert req["body"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_chat_json_parses_fenced_reply_and_adds_json_instruction(fake_server):
    fake_server.responder = lambda system, prompt: 'Sure!\n```json\n{"items": ["a", "b"]}\n```'
    assert _client(fake_server).chat_json("list please") == {"items": ["a", "b"]}
    system = fake_server.requests[-1]["body"]["messages"][0]["content"]
    assert "Respond with valid JSON only" in system


def test_embeddings_keep_request_order_and_send_nim_options(fake_server):
    texts = [f"text {i}" for i in range(5)]
    vectors = _client(fake_server).embed(texts, input_type="query", batch_size=2)
    assert len(vectors) == 5 and all(len(v) == 64 for v in vectors)
    from tests.fake_llm import fake_vector

    for text, vec in zip(texts, vectors):
        assert vec == pytest.approx(fake_vector(text), rel=1e-6)
    assert fake_server.count("/embeddings") == 3  # batches of 2, 2, 1
    body = fake_server.requests[-1]["body"]
    assert body["encoding_format"] == "float"
    assert body["input_type"] == "query" and body["truncate"] == "END"


def test_rate_limit_is_retried_then_succeeds(fake_server):
    fake_server.fail_with.extend([429])
    fake_server.responder = lambda system, prompt: "ok"
    assert _client(fake_server, retries=3).chat("x") == "ok"
    assert fake_server.count("/chat/completions") == 2  # 429, then 200


def test_server_errors_exhaust_retries(fake_server):
    fake_server.fail_with.extend([500, 503, 500])
    with pytest.raises(NIMError, match="after 3 attempt"):
        _client(fake_server, retries=3).chat("x")
    assert fake_server.count("/chat/completions") == 3


def test_auth_error_is_not_retried(fake_server):
    fake_server.fail_with.extend([401])
    with pytest.raises(NIMError, match="after 1 attempt"):
        _client(fake_server, retries=3).chat("x")
    assert fake_server.count("/chat/completions") == 1


def test_embedding_retry(fake_server):
    fake_server.fail_with.extend([429])
    assert len(_client(fake_server).embed(["a"])) == 1
    assert fake_server.count("/embeddings") == 2


def test_missing_key_raises_instead_of_exiting(no_key):
    with pytest.raises(MissingAPIKeyError, match="NVIDIA_API_KEY is not set"):
        require_api_key()
    with pytest.raises(MissingAPIKeyError):
        NIMClient()


def test_placeholder_key_counts_as_missing(no_key, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-XXXXXXXXXXXXXXXX")
    with pytest.raises(MissingAPIKeyError):
        require_api_key()


def test_dotenv_in_current_directory_is_loaded(no_key):
    (no_key / ".env").write_text("NVIDIA_API_KEY=nvapi-from-dotenv\nNIM_MODEL=fake/model\n", encoding="utf-8")
    assert require_api_key() == "nvapi-from-dotenv"


def test_text_factory_without_key_does_not_kill_the_interpreter(no_key):
    code = (
        "from factory.text import TextFactory\n"
        "from factory.qa import QAFactory\n"
        "from factory.nim import MissingAPIKeyError\n"
        "for cls in (TextFactory, QAFactory):\n"
        "    try:\n"
        "        cls()\n"
        "    except MissingAPIKeyError:\n"
        "        print('caught', cls.__name__)\n"
        "print('returned')\n"
    )
    import os

    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
    env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(no_key), env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["caught", "TextFactory", "caught", "QAFactory", "returned"]


# --------------------------------------------------------------------------
# JSON extraction and shape normalization
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply, expected",
    [
        ('["a", "b"]', ["a", "b"]),
        ('```json\n["a"]\n```', ["a"]),
        ('```\n{"k": 1}\n```', {"k": 1}),
        ('Here you go: {"questions": ["q1", "q2"]} Hope that helps!', {"questions": ["q1", "q2"]}),
        ('Sure. ["x", "y"] and also {"z": 1}', ["x", "y"]),
        ('Intro text\n```json\n{"subject": "s", "body": "b [1]"}\n```\nOutro', {"subject": "s", "body": "b [1]"}),
        ('"just a string"', "just a string"),
        ("Note [draft]: {\"a\": 2}", {"a": 2}),
    ],
)
def test_extract_json_edge_cases(reply, expected):
    assert extract_json(reply) == expected


def test_extract_json_failure_is_clear():
    with pytest.raises(ModelJSONError, match="did not return valid JSON"):
        extract_json("no json here at all")
    with pytest.raises(NIMError):  # legacy wrapper keeps its exception type
        _extract_json("nope")


@pytest.mark.parametrize(
    "obj, expected",
    [
        (["a", "b"], ["a", "b"]),
        ([{"text": "a"}, {"question": "b"}], ["a", "b"]),
        ({"questions": ["a", "b"]}, ["a", "b"]),
        ({"data": [{"example": "a"}]}, ["a"]),
        ({"whatever": ["a"]}, ["a"]),
        ({"1": "a", "2": "b"}, ["a", "b"]),
        ("single", ["single"]),
        (["", "  ", "x"], ["x"]),
        (None, []),
    ],
)
def test_as_text_list_shapes(obj, expected):
    assert as_text_list(obj) == expected


@pytest.mark.parametrize(
    "obj, expected",
    [
        ("r", "r"),
        (["first", "second"], "first"),
        ({"review": "r"}, "r"),
        ({"rating": 5, "comment": "c"}, "c"),
        ({"outer": {"text": "deep"}}, "deep"),
        (None, ""),
    ],
)
def test_as_text_shapes(obj, expected):
    assert as_text(obj) == expected
