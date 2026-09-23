"""Chat-format JSONL for supervised fine-tuning: one shape per record kind."""
import json

from factory import to_chat_jsonl, write_jsonl


def _lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_qa_record_with_and_without_context(tmp_path):
    rec = {"question": "How long?", "answer": "30 days.", "context": "Refunds within 30 days."}
    to_chat_jsonl([rec], str(tmp_path / "a.jsonl"))
    assert _lines(tmp_path / "a.jsonl")[0]["messages"] == [
        {"role": "user", "content": "How long?"},
        {"role": "assistant", "content": "30 days."},
    ]
    to_chat_jsonl([rec], str(tmp_path / "b.jsonl"), system="Be exact.", include_context=True)
    messages = _lines(tmp_path / "b.jsonl")[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("Be exact.") and "Refunds within 30 days." in messages[0]["content"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_classification_record(tmp_path):
    to_chat_jsonl([{"text": "Card charged twice", "label": "billing"}], str(tmp_path / "c.jsonl"))
    messages = _lines(tmp_path / "c.jsonl")[0]["messages"]
    assert messages[0] == {"role": "system", "content": "Classify the message into the correct category."}
    assert messages[1:] == [
        {"role": "user", "content": "Card charged twice"},
        {"role": "assistant", "content": "billing"},
    ]


def test_generative_records_by_kind(tmp_path):
    records = [
        {"text": "Great sound.", "kind": "review", "product": "headphones", "sentiment": "positive"},
        {"text": "Login broken", "kind": "ticket", "category": "login"},
        {"text": "Ana is an admin.", "kind": "persona", "context": "IT admins"},
        {"text": "How can I reset it?", "kind": "paraphrase", "intent": "Reset password?"},
    ]
    to_chat_jsonl(records, str(tmp_path / "g.jsonl"))
    lines = _lines(tmp_path / "g.jsonl")
    assert lines[0]["messages"][1]["content"] == "Product: headphones. Sentiment: positive."
    assert lines[1]["messages"][1]["content"] == "Category: login."
    assert lines[2]["messages"][1]["content"] == "Audience: IT admins."
    assert lines[3]["messages"][1]["content"] == "Reset password?"
    for line, rec in zip(lines, records):
        assert line["messages"][-1] == {"role": "assistant", "content": rec["text"]}


def test_unknown_record_falls_back_to_json_dump(tmp_path):
    to_chat_jsonl([{"foo": 1}], str(tmp_path / "f.jsonl"))
    messages = _lines(tmp_path / "f.jsonl")[0]["messages"]
    assert messages[-1] == {"role": "assistant", "content": '{"foo": 1}'}


def test_write_jsonl_creates_parent_and_keeps_unicode(tmp_path):
    path = write_jsonl([{"text": "cafe con leche - ñandú"}], str(tmp_path / "nested" / "x.jsonl"))
    with open(path, encoding="utf-8") as fh:
        assert "ñandú" in fh.read()
