"""Text tasks with scripted fake clients: quotas, top-ups, dedup, planning."""
import json
from collections import Counter

import pytest

from factory.text import (
    TextFactory,
    TextResult,
    plan_text_task,
    run_text_task,
    text_task_quotas,
)
from factory.validate import validate_records
from tests.fake_llm import ScriptedClient, fake_vector


def _short_and_duplicated():
    """Replies with only 3 items when asked for more, one of them a duplicate
    of another in the same reply."""
    state = {"n": 0}

    def respond(system, prompt):
        state["n"] += 1
        k = state["n"]
        return json.dumps([f"unique message {k}a", f"unique message {k}b", f"unique message {k}a"])

    return respond


def test_classification_tops_up_to_exact_quota():
    client = ScriptedClient(_short_and_duplicated())
    result = TextFactory(client=client, max_rounds=10).classification(["billing", "technical"], per_class=10)
    assert isinstance(result, TextResult)
    assert Counter(r["label"] for r in result) == {"billing": 10, "technical": 10}  # was 2 / 2
    assert result.shortfall == {}
    assert result.requests == len(client.chat_calls) == 10  # 5 rounds x 2 labels
    # Top-up prompts ask only for the missing number and show examples to avoid.
    topups = [c["prompt"] for c in client.chat_calls if "already exist" in c["prompt"]]
    assert topups and "Generate 8 realistic examples" in topups[0]


def test_shortfall_is_reported_when_model_keeps_repeating():
    client = ScriptedClient(lambda s, p: json.dumps(["same", "same", "other"]))
    result = TextFactory(client=client, max_rounds=3).classification(["a", "b"], per_class=5)
    assert Counter(r["label"] for r in result) == {"a": 2, "b": 2}
    assert result.shortfall == {"a": 3, "b": 3}
    assert len(client.chat_calls) == 6  # bounded: 3 rounds per label
    report = validate_records(list(result), {"task": "classification", "labels": ["a", "b"], "per_class": 5})
    assert [c.name for c in report.failures()] == ["quota"]


def test_all_five_tasks_meet_their_quotas_and_validate():
    configs = [
        {"task": "paraphrase", "seeds": ["How do I reset my password?", "Where is my invoice?"], "variations": 4},
        {"task": "classification", "labels": ["billing", "bug", "praise"], "per_class": 6},
        {"task": "personas", "count": 23, "context": "IT admins"},
        {"task": "reviews", "product": "headphones", "count": 11,
         "sentiments": {"positive": 0.5, "neutral": 0.2, "negative": 0.3}},
        {"task": "tickets", "categories": ["login", "billing"], "per_category": 4, "product": "an app"},
    ]
    for config in configs:
        result = run_text_task(config, client=ScriptedClient())
        expected = text_task_quotas(config)
        counts = Counter(str(r[expected["group_key"]]) for r in result)
        assert counts == {str(k): v for k, v in expected["quotas"].items() if v}, config["task"]
        assert result.shortfall == {}
        report = validate_records(list(result), config)
        assert report.ok, report.render()


def test_reviews_carry_consistent_ratings_and_metadata():
    result = run_text_task({"task": "reviews", "product": "a kettle", "count": 20}, client=ScriptedClient())
    ranges = {"positive": {4, 5}, "neutral": {3}, "negative": {1, 2}}
    for r in result:
        assert r["rating"] in ranges[r["sentiment"]]
        assert r["product"] == "a kettle" and r["kind"] == "review" and r["aspect"]


def test_tickets_have_subject_and_body():
    result = run_text_task({"task": "tickets", "categories": ["bug"], "per_category": 3}, client=ScriptedClient())
    assert len(result) == 3
    for r in result:
        assert r["subject"].startswith("bug issue") and r["body"]
        assert r["text"] == f"{r['subject']}\n\n{r['body']}"


def test_semantic_near_duplicates_are_removed_with_embeddings():
    replies = iter([
        json.dumps(["Refund please", "I want my money back", "Card charged twice"]),
        json.dumps(["Another billing question"]),
    ])

    def embed(text):
        # Pretend the model sees the first two as the same meaning.
        if text in ("Refund please", "I want my money back"):
            return fake_vector("refund")
        return fake_vector(text)

    client = ScriptedClient(lambda s, p: next(replies), embed_fn=embed)
    result = TextFactory(client=client).classification(["billing"], per_class=3)
    assert [r["text"] for r in result] == ["Refund please", "Card charged twice", "Another billing question"]
    assert result.dedup == "embedding"


def test_embedding_failure_falls_back_to_lexical():
    client = ScriptedClient(
        lambda s, p: json.dumps([
            "my invoice from last month shows a charge that I do not recognize",
            "my invoice from last month shows a charge that I do not recognize!",
            "dogs",
        ]),
        embed_error=RuntimeError("embeddings down"),
    )
    result = TextFactory(client=client, max_rounds=1).classification(["x"], per_class=3)
    assert result.dedup == "lexical"
    assert [r["text"] for r in result] == [
        "my invoice from last month shows a charge that I do not recognize", "dogs"]


def test_dedup_none_keeps_near_duplicates_but_not_exact_ones():
    client = ScriptedClient(lambda s, p: json.dumps(["a b c d", "a b c d!", "A  B C D"]))
    result = TextFactory(client=client, dedup="none", max_rounds=1).classification(["x"], per_class=3)
    assert [r["text"] for r in result] == ["a b c d", "a b c d!"]
    assert client.embed_calls == []


def test_same_seed_same_prompts():
    c1, c2 = ScriptedClient(), ScriptedClient()
    run_text_task({"task": "reviews", "product": "p", "count": 6, "seed": 3}, client=c1)
    run_text_task({"task": "reviews", "product": "p", "count": 6, "seed": 3}, client=c2)
    assert [c["prompt"] for c in c1.chat_calls] == [c["prompt"] for c in c2.chat_calls]


@pytest.mark.parametrize(
    "config, chats, embeds",
    [
        ({"task": "paraphrase", "seeds": ["a", "b", "c"], "variations": 5}, 3, 3),
        ({"task": "classification", "labels": ["a", "b"], "per_class": 12}, 2, 2),
        ({"task": "personas", "count": 25}, 3, 1),
        ({"task": "reviews", "product": "p", "count": 24}, 24, 3),
        ({"task": "tickets", "categories": ["a", "b"], "per_category": 8, "dedup": "lexical"}, 16, 0),
    ],
)
def test_dry_run_plan_matches_first_round(config, chats, embeds):
    plan = plan_text_task(config)
    assert plan.chat_requests == chats
    assert plan.embed_requests == embeds
    # With a well-behaved model no top-up is needed, so the plan is exact.
    client = ScriptedClient()
    run_text_task(config, client=client)
    assert len(client.chat_calls) == chats
    assert len(client.embed_calls) == embeds
    assert client.chat_calls[0]["prompt"] == plan.first_prompt
    assert "Dry run" in plan.render()


def test_task_config_errors_are_clear():
    with pytest.raises(ValueError, match="Unknown text task"):
        run_text_task({"task": "poems"}, client=ScriptedClient())
    with pytest.raises(ValueError, match="needs 'labels'"):
        run_text_task({"task": "classification"}, client=ScriptedClient())
    with pytest.raises(ValueError, match="dedup"):
        TextFactory(client=ScriptedClient(), dedup="fuzzy")


def test_validate_records_flags_bad_labels_and_duplicates():
    config = {"task": "classification", "labels": ["a", "b"], "per_class": 2}
    records = [
        {"text": "x", "label": "a"}, {"text": "X ", "label": "a"},
        {"text": "y", "label": "b"}, {"text": "", "label": "zzz"},
    ]
    failed = {c.name for c in validate_records(records, config).failures()}
    assert failed == {"required_fields", "label_set", "quota", "duplicates"}


def test_legacy_json_helpers_still_work():
    assert TextFactory._json_list({"items": ["a"]}) == ["a"]
    assert TextFactory._json_scalar({"review": "r"}) == "r"
