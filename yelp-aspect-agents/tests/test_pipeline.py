import json

import pytest

from yelp_agents import AggregatorAgent, AspectAgent, Orchestrator, find_conflicts
from yelp_core import Review, Settings, build_llm
from yelp_core.llm import extract_json
from yelp_core.schemas import AspectResult, Mention
from yelp_eval import business_rollup, evaluate, format_report, load_results, results_frame
from yelp_eval.figures import aspect_space_3d


@pytest.fixture
def mock_llm():
    s = Settings()
    s.provider = "mock"
    return build_llm(s)


REVIEWS = [
    Review("r1", "b1", 5, "The pasta was delicious and fresh. Our waiter was friendly and attentive. Cozy place.", "Luigi's", "Tampa"),
    Review("r2", "b1", 1, "Food was cold and bland. The staff were rude and slow. Dirty tables everywhere.", "Luigi's", "Tampa"),
    Review("r3", "b2", 3, "Parking was fine I guess.", "Lot 9", "Reno"),
]


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_agents_survive_llm_failure():
    class Boom:
        provider, model = "boom", "x"
        def chat_json(self, *a, **k):
            raise RuntimeError("down")
    res = AspectAgent("service", Boom()).run("anything")
    assert res.error and res.sentiment == "not_mentioned" and res.score == 0
    good = AspectResult("food", mentions=[Mention("q", "positive", 2)], score=2 / 3, sentiment="positive")
    overall = AggregatorAgent(Boom()).run("x", {"food": good})
    assert overall.error and overall.sentiment == "positive" and overall.score == pytest.approx(0.6667, abs=1e-4)


def test_malformed_agent_output_is_an_error_not_a_score():
    class Weird:
        provider, model = "weird", "x"
        def chat_json(self, *a, **k):
            return {"sentiment": "positive", "score": 0.9}   # old format / hallucinated number
    res = AspectAgent("food", Weird()).run("The food was great.")
    assert res.error and res.score == 0


def test_single_mention_object_is_accepted():
    class Single:
        provider, model = "single", "x"
        def chat_json(self, *a, **k):
            return {"quote": "The food was great", "polarity": "positive", "intensity": 2}
    res = AspectAgent("food", Single()).run("The food was great.")
    assert res.error is None and res.score == pytest.approx(2 / 3, abs=1e-4)


def test_retry_errors_reanalyses_only_failed_reviews(tmp_path, mock_llm):
    out = tmp_path / "r.jsonl"
    class Flaky:
        provider, model = "mock", "flaky"
        def chat_json(self, msgs, max_tokens=0):
            if "Parking" in msgs[-1]["content"] and "<aspect>food</aspect>" in msgs[-1]["content"]:
                raise RuntimeError("timeout")
            return mock_llm.chat_json(msgs)
    Orchestrator(Flaky()).run(REVIEWS, out)
    assert Orchestrator(mock_llm).run(REVIEWS, out) and len(out.read_text().splitlines()) == 3  # plain resume
    fixed = Orchestrator(mock_llm).run(REVIEWS, out, retry_errors=True)
    assert len(out.read_text().splitlines()) == 4                                         # only r3 re-run
    assert all(not a.error for r in fixed for a in r.aspects.values())
    assert len(load_results(out)) == 3


def test_find_conflicts_groups_overlapping_evidence():
    asp = {
        "service": AspectResult("service", mentions=[Mention("a", "negative", 2, True, [[0, 20]])]),
        "ambience": AspectResult("ambience", mentions=[Mention("b", "negative", 2, True, [[5, 15]]),
                                                       Mention("c", "positive", 1, True, [[30, 40]])]),
        "food": AspectResult("food"),
    }
    assert find_conflicts(asp) == [[("service", 0), ("ambience", 0)]]


def test_arbiter_removes_misattributed_evidence(mock_llm):
    text = "Slowest service ever from the waiter. The decor is lovely."
    class Greedy:  # ambience agent wrongly claims the service sentence
        provider, model = "mock", "greedy"
        def chat_json(self, msgs, max_tokens=0):
            p = msgs[-1]["content"]
            if "<candidates>" in p:
                return mock_llm.chat_json(msgs)
            if "<aspect>ambience</aspect>" in p:
                return {"mentions": [{"quote": "Slowest service ever from the waiter.", "polarity": "negative", "intensity": 3},
                                     {"quote": "The decor is lovely.", "polarity": "positive", "intensity": 2}]}
            if "<aspect>service</aspect>" in p:
                return {"mentions": [{"quote": "Slowest service ever", "polarity": "negative", "intensity": 3}]}
            if "<aspect>" in p:
                return {"mentions": []}
            return {"polarity": "negative", "intensity": 1}
    a = Orchestrator(Greedy()).analyze(Review("x", "b", 2, text))
    assert a.meta["arbitrations"][0]["kept"] == ["service"]
    assert a.aspects["ambience"].score == pytest.approx(2 / 3, abs=1e-4)      # only the decor remains
    assert a.aspects["ambience"].dropped[0].note.startswith("arbiter")
    assert a.aspects["service"].score == -1.0


def test_end_to_end_with_mock(tmp_path, mock_llm):
    out = tmp_path / "results.jsonl"
    results = Orchestrator(mock_llm).run(REVIEWS, out, workers=2)
    by_id = {r.review.review_id: r for r in results}
    assert all(v > 0 for v in by_id["r1"].vector)
    assert all(v < 0 for v in by_id["r2"].vector)
    assert by_id["r3"].vector == [0.0, 0.0, 0.0]
    assert by_id["r1"].overall.sentiment == "positive" and by_id["r2"].overall.sentiment == "negative"
    for r in results:                     # every kept quote is really in the text
        for asp in r.aspects.values():
            for m in asp.mentions:
                assert all(r.review.text[s:e] for s, e in m.spans)

    Orchestrator(mock_llm).run(REVIEWS, out)          # resume: no duplicate work
    assert len(out.read_text().splitlines()) == 3

    with out.open("a") as f:                          # a half-written line from a live run is tolerated
        f.write('{"review": {"review_id": "partial"')
    loaded = load_results(out)
    assert len(loaded) == 3
    assert "food" in format_report(evaluate(loaded))
    assert len(business_rollup(loaded, min_reviews=2)) == 1
    assert "scatter3d" in aspect_space_3d(results_frame(loaded)).to_json()


def test_old_result_files_are_not_resumed(tmp_path, mock_llm):
    out = tmp_path / "old.jsonl"
    out.write_text(json.dumps({"review": {"review_id": "r1"}, "meta": {}}) + "\n")
    with pytest.raises(ValueError, match="older version"):
        Orchestrator(mock_llm).run(REVIEWS, out)
