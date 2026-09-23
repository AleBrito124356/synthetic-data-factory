"""Record/replay cassettes: reproducible LLM dataset builds without a key."""
import json
import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

from factory.cli import main
from factory.llm import CassetteMissError, RecordingClient, ReplayClient, ReplayedError
from factory.text import run_text_task
from tests.fake_llm import ScriptedClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_TASK = os.path.join(ROOT, "schemas", "text-task.yaml")
QA_TASK = os.path.join(ROOT, "schemas", "qa-example.yaml")


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def test_record_then_replay_reproduces_records(tmp_path):
    cassette = str(tmp_path / "c.jsonl")
    config = {"task": "classification", "labels": ["a", "b"], "per_class": 4}
    recorded = run_text_task(config, client=RecordingClient(ScriptedClient(), cassette))
    replay = ReplayClient(cassette)
    replayed = run_text_task(config, client=replay)
    assert list(replayed) == list(recorded)
    assert replay.remaining() == 0
    with open(cassette, encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        entry = json.loads(fh.readline())
    assert header["cassette"] == 1 and header["model"] == "scripted"
    assert entry["op"] == "chat" and "prompt" in entry["request"] and "response" in entry


def test_identical_requests_replay_in_recorded_order(tmp_path):
    cassette = str(tmp_path / "c.jsonl")
    answers = iter(["first", "second"])
    rec = RecordingClient(ScriptedClient(lambda s, p: next(answers)), cassette)
    assert [rec.chat("same"), rec.chat("same")] == ["first", "second"]
    rep = ReplayClient(cassette)
    assert [rep.chat("same"), rep.chat("same")] == ["first", "second"]
    with pytest.raises(CassetteMissError):
        rep.chat("same")  # recorded twice, asked a third time


def test_cassette_miss_is_explicit(tmp_path):
    cassette = str(tmp_path / "c.jsonl")
    RecordingClient(ScriptedClient(lambda s, p: "x"), cassette).chat("hello")
    with pytest.raises(CassetteMissError, match="no recorded response"):
        ReplayClient(cassette).chat("hello", temperature=0.1)  # different parameters
    with pytest.raises(CassetteMissError, match="not found"):
        ReplayClient(str(tmp_path / "missing.jsonl"))


def test_failed_requests_fail_the_same_way_on_replay(tmp_path):
    cassette = str(tmp_path / "c.jsonl")
    client = ScriptedClient(lambda s, p: json.dumps(["alpha one two three", "beta four five six"]),
                            embed_error=RuntimeError("embeddings down"))
    config = {"task": "classification", "labels": ["x"], "per_class": 2}
    recorded = run_text_task(config, client=RecordingClient(client, cassette))
    assert recorded.dedup == "lexical"
    replayed = run_text_task(config, client=ReplayClient(cassette))
    assert list(replayed) == list(recorded) and replayed.dedup == "lexical"
    with pytest.raises(ReplayedError, match="embeddings down"):
        ReplayClient(cassette).embed(["alpha one two three", "beta four five six"])


def test_embeddings_are_rounded_identically_for_record_and_replay(tmp_path):
    cassette = str(tmp_path / "c.jsonl")
    rec = RecordingClient(ScriptedClient(embed_fn=lambda t: [1 / 3, 2 / 3]), cassette)
    assert rec.embed(["t"]) == [[0.333333, 0.666667]]
    assert ReplayClient(cassette).embed(["t"]) == [[0.333333, 0.666667]]


def _live_env(monkeypatch, server, tmp_path):
    monkeypatch.setenv("NVIDIA_API_KEY", "local-fake-key")
    monkeypatch.setenv("NIM_BASE_URL", server.url)
    monkeypatch.chdir(tmp_path)


def test_cli_text_record_then_replay_without_key_is_byte_identical(fake_server, monkeypatch, tmp_path):
    _live_env(monkeypatch, fake_server, tmp_path)
    args = ["generate", "text", "--task", TEXT_TASK, "--validate"]
    assert main(args + ["--out", "rec.jsonl", "--chat", "rec.chat.jsonl", "--record", "run.cassette.jsonl"]) == 0
    sent = len(fake_server.requests)
    assert fake_server.count("/chat/completions") == 24 and fake_server.count("/embeddings") == 3

    monkeypatch.delenv("NVIDIA_API_KEY")
    monkeypatch.delenv("NIM_BASE_URL")
    assert main(args + ["--out", "rep.jsonl", "--chat", "rep.chat.jsonl", "--replay", "run.cassette.jsonl"]) == 0
    assert len(fake_server.requests) == sent  # replay never touched the network
    assert _read(tmp_path / "rep.jsonl") == _read(tmp_path / "rec.jsonl")
    assert _read(tmp_path / "rep.chat.jsonl") == _read(tmp_path / "rec.chat.jsonl")
    with open(tmp_path / "rec.jsonl", encoding="utf-8") as fh:
        sentiments = [json.loads(line)["sentiment"] for line in fh]
    assert {s: sentiments.count(s) for s in set(sentiments)} == {"positive": 12, "neutral": 5, "negative": 7}


def test_cli_qa_record_then_replay_without_key_is_byte_identical(fake_server, monkeypatch, tmp_path):
    _live_env(monkeypatch, fake_server, tmp_path)
    args = ["generate", "qa", "--task", QA_TASK, "--with-context", "--validate"]
    assert main(args + ["--out", "rec.jsonl", "--chat", "rec.chat.jsonl", "--record", "qa.cassette.jsonl"]) == 0
    monkeypatch.delenv("NVIDIA_API_KEY")
    assert main(args + ["--out", "rep.jsonl", "--chat", "rep.chat.jsonl", "--replay", "qa.cassette.jsonl"]) == 0
    assert _read(tmp_path / "rep.jsonl") == _read(tmp_path / "rec.jsonl")
    assert _read(tmp_path / "rep.chat.jsonl") == _read(tmp_path / "rec.chat.jsonl")
    assert _read(tmp_path / "rec.jsonl").count(b"\n") > 0


def test_cli_replay_with_changed_task_reports_a_miss(fake_server, monkeypatch, tmp_path, capsys):
    _live_env(monkeypatch, fake_server, tmp_path)
    assert main(["generate", "text", "--task", TEXT_TASK, "--out", "a.jsonl", "--record", "c.jsonl"]) == 0
    changed = tmp_path / "changed.yaml"
    changed.write_text(open(TEXT_TASK, encoding="utf-8").read().replace("headphones", "earbuds"), encoding="utf-8")
    monkeypatch.delenv("NVIDIA_API_KEY")
    capsys.readouterr()
    assert main(["generate", "text", "--task", str(changed), "--out", "b.jsonl", "--replay", "c.jsonl"]) == 3
    assert "re-run with --record" in capsys.readouterr().err


def test_cli_without_key_exits_2_with_help(no_key, capsys):
    assert main(["generate", "text", "--task", TEXT_TASK, "--out", "o.jsonl"]) == 2
    err = capsys.readouterr().err
    assert "NVIDIA_API_KEY is not set" in err and "--dry-run" in err
    assert main(["generate", "qa", "--task", QA_TASK, "--out", "o.jsonl"]) == 2


def test_cli_dry_run_needs_no_key(no_key, capsys):
    assert main(["generate", "text", "--task", TEXT_TASK, "--out", "o.jsonl", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "planned chat requests:      24" in out and "product review" in out
    assert not os.path.exists("o.jsonl")
    assert main(["generate", "qa", "--task", QA_TASK, "--out", "o.jsonl", "--dry-run"]) == 0
    assert "distinct questions" in capsys.readouterr().out


def test_cli_record_and_replay_are_exclusive(no_key, capsys):
    rc = main(["generate", "text", "--task", TEXT_TASK, "--out", "o.jsonl", "--record", "a", "--replay", "b"])
    assert rc == 1 and "mutually exclusive" in capsys.readouterr().err


def _installed_sdf():
    scripts = sysconfig.get_path("scripts")
    for name in ("sdf", "sdf.exe"):
        candidate = os.path.join(scripts, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("sdf")


@pytest.mark.skipif(_installed_sdf() is None, reason="package not installed (pip install -e .)")
def test_installed_sdf_replays_without_key(fake_server, monkeypatch, tmp_path):
    _live_env(monkeypatch, fake_server, tmp_path)
    assert main(["generate", "text", "--task", TEXT_TASK, "--out", "rec.jsonl", "--record", "c.jsonl"]) == 0
    env = {k: v for k, v in os.environ.items() if k not in ("NVIDIA_API_KEY", "NIM_BASE_URL")}
    proc = subprocess.run(
        [_installed_sdf(), "generate", "text", "--task", TEXT_TASK, "--out", "rep.jsonl", "--replay", "c.jsonl"],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _read(tmp_path / "rep.jsonl") == _read(tmp_path / "rec.jsonl")
