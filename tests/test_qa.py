"""Q&A pipeline with scripted fake clients."""
import json
import os

import pytest

from factory.qa import QAFactory, QAResult, _chunk_text, _resolve_docs, plan_qa_task, run_qa_task
from factory.validate import validate_records
from tests.fake_llm import ScriptedClient, SmartResponder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = {"text": "Refunds are accepted within 30 days of delivery. Gift cards are not refundable.",
       "source": "policy"}


def test_chunking_windows_and_overlap():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = _chunk_text(words, chunk_size=10, overlap=3)
    assert chunks[0].split() == [f"w{i}" for i in range(10)]
    assert chunks[1].split()[0] == "w7"  # step = 10 - 3
    assert chunks[-1].split()[-1] == "w24"
    assert _chunk_text("short text", 10, 3) == ["short text"]
    assert _chunk_text("   ", 10, 3) == []


def test_object_wrapped_questions_are_unwrapped():
    def respond(system, prompt):
        if "distinct questions" in prompt:
            return json.dumps({"questions": ["How long is the refund window?", "Are gift cards refundable?"]})
        return SmartResponder()(system, prompt)

    result = QAFactory(client=ScriptedClient(respond)).generate([DOC], questions_per_chunk=2)
    assert [p["question"] for p in result] == ["How long is the refund window?", "Are gift cards refundable?"]
    # Used to yield ONE pair whose question was the dict's repr.
    assert not any(p["question"].startswith("{") for p in result)


def test_full_pipeline_fields_and_counters():
    result = QAFactory(client=ScriptedClient()).generate([DOC], questions_per_chunk=3)
    assert isinstance(result, QAResult)
    assert len(result) == 3 and result.chunks == 1 and result.candidates == 3
    for pair in result:
        assert set(pair) >= {"question", "answer", "context", "source", "chunk_index", "hard_negative", "quality"}
        assert pair["source"] == "policy"
        assert pair["hard_negative"] != pair["answer"]
        assert min(pair["quality"].values()) >= 4
    # 1 question request + 3 x (answer + hard negative + critique)
    assert result.requests == 1 + 3 * 3


def test_not_in_passage_answers_are_skipped():
    def respond(system, prompt):
        if "distinct questions" in prompt:
            return json.dumps(["Answerable?", "Totally unanswerable question?"])
        return SmartResponder()(system, prompt)

    result = QAFactory(client=ScriptedClient(respond)).generate([DOC], questions_per_chunk=2)
    assert [p["question"] for p in result] == ["Answerable?"]
    assert result.unanswerable == 1


def test_quality_filter_drops_weak_pairs():
    def respond(system, prompt):
        if "distinct questions" in prompt:
            return json.dumps(["Good question?", "A lowquality question?"])
        return SmartResponder()(system, prompt)

    result = QAFactory(client=ScriptedClient(respond)).generate([DOC], questions_per_chunk=2, min_quality=4)
    assert [p["question"] for p in result] == ["Good question?"]
    assert result.low_quality == 1


def test_unparseable_critique_scores_zero():
    def respond(system, prompt):
        if "Score this" in prompt:
            return "I would rate it highly!"
        return SmartResponder()(system, prompt)

    result = QAFactory(client=ScriptedClient(respond)).generate([DOC], questions_per_chunk=1)
    assert len(result) == 0 and result.low_quality == 1


def test_duplicate_questions_across_overlapping_chunks_are_skipped():
    text = " ".join(f"w{i}" for i in range(30))

    def respond(system, prompt):
        if "distinct questions" in prompt:
            return json.dumps(["What is the policy?"])
        return SmartResponder()(system, prompt)

    result = QAFactory(client=ScriptedClient(respond)).generate(
        [{"text": text, "source": "d"}], questions_per_chunk=1, chunk_size=10, overlap=2)
    assert result.chunks == 4 and len(result) == 1 and result.duplicates == 3


def test_no_hard_negatives_option():
    result = QAFactory(client=ScriptedClient()).generate([DOC], questions_per_chunk=1, hard_negatives=False)
    assert "hard_negative" not in result[0]
    assert result.requests == 1 + 2


def test_run_qa_task_with_shipped_example_docs():
    import yaml

    path = os.path.join(ROOT, "schemas", "qa-example.yaml")
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["_base_dir"] = os.path.dirname(path)
    result = run_qa_task(config, client=ScriptedClient())
    assert len(result) > 0
    assert {p["source"] for p in result} >= {"refund_policy.md", "shipping_policy.md", "plans-blurb"}
    report = validate_records(list(result), config, kind="qa")
    assert report.ok, report.render()

    plan = plan_qa_task(config)
    assert plan.chat_requests == result.chunks * (1 + 2 * 3)
    assert "distinct questions" in plan.first_prompt


def test_validate_records_qa_failures():
    records = [
        {"question": "Q?", "answer": "NOT_IN_PASSAGE", "context": "c", "source": "s",
         "hard_negative": "NOT_IN_PASSAGE", "quality": {"groundedness": 2, "answerability": 5, "clarity": 5}},
        {"question": "q? ", "answer": "a", "context": "c", "source": "s",
         "hard_negative": "b", "quality": {"groundedness": 5, "answerability": 5, "clarity": 5}},
    ]
    failed = {c.name for c in validate_records(records, {"min_quality": 4}, kind="qa").failures()}
    assert failed == {"answerable", "quality_min", "hard_negative_differs", "duplicates"}


def test_resolve_docs_forms_and_errors(tmp_path):
    (tmp_path / "d.md").write_text("from a file", encoding="utf-8")
    docs = _resolve_docs(["inline", {"text": "t", "source": "named"}, {"path": "d.md"}], base_dir=str(tmp_path))
    assert [d["source"] for d in docs] == ["inline-0", "named", "d.md"]
    with pytest.raises(ValueError):
        _resolve_docs([42])
    with pytest.raises(ValueError, match="at least one document"):
        run_qa_task({"docs": []}, client=ScriptedClient())
