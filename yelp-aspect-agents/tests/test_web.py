import json
import time

import pytest
from fastapi.testclient import TestClient

from yelp_core import Review
from yelp_data import write_reviews
from yelp_web.server import create_app

TEXTS = {
    1: "Food was cold and bland. The staff were rude and slow. Dirty tables everywhere.",
    2: "The burger was bland. Our server was slow.",
    3: "Food was fine. Service was okay. The place is noisy.",
    4: "Great pasta and friendly staff. Nice patio.",
    5: "Delicious fresh sushi, attentive waiter, cozy place.",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    for k in ("NVIDIA_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    sample = tmp_path / "sample.jsonl"
    write_reviews([Review(f"r{i}", f"b{i % 4}", s, TEXTS[s], f"Biz {i % 4}", "Tampa", date="2020-01-01")
                   for i, s in enumerate([1, 2, 3, 4, 5] * 3)], sample)
    return TestClient(create_app(sample, tmp_path / "results.jsonl"))


def _events(body: str):
    out = []
    for chunk in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _wait_for_job(client):
    for _ in range(200):
        if not client.get("/api/runs/current").json()["running"]:
            break
        time.sleep(0.05)
    return client.get("/api/runs/current").json()


def test_ui_and_status(client):
    assert "Yelp Aspect Agents" in client.get("/").text
    assert 'data-tab="compare"' in client.get("/").text
    for f in ("app.js", "styles.css"):
        assert client.get(f"/static/{f}").status_code == 200
    s = client.get("/api/status").json()
    assert s["default"]["provider"] == "mock" and s["sample_size"] == 15
    assert set(s["aspects"]) == {"food", "service", "ambience"}


def test_analyze_streams_python_scores(client):
    text = "The pasta was delicious. Our waiter was rude."
    ev = _events(client.post("/api/analyze", json={"text": text, "provider": "mock"}).text)
    names = [e for e, _ in ev]
    assert names[0] == "start" and names[-1] == "done"
    assert names.count("agent_start") == 3 and "aggregator_start" in names and "overall" in names
    final = dict(ev)["done"]["analysis"]
    food, service = final["aspects"]["food"], final["aspects"]["service"]
    assert food["sentiment"] == "positive" and food["score"] > 0
    assert service["sentiment"] == "negative" and service["score"] < 0
    assert final["aspects"]["ambience"]["sentiment"] == "not_mentioned"
    for a in final["aspects"].values():          # spans point at the quoted text
        for m in a["mentions"]:
            (s, e), = m["spans"]
            assert text[s:e].lower() in m["quote"].lower() or m["quote"].lower() in text[s:e].lower()
    assert final["overall"]["stars_rule"] == pytest.approx(3 + 2 * final["overall"]["score"], abs=1e-3)
    assert final["vector"] == [final["aspects"][a]["score"] for a in ("food", "service", "ambience")]


def test_sample_review_keeps_its_real_stars(client):
    r = client.get("/api/sample/random").json()
    done = dict(_events(client.post("/api/analyze", json={"text": r["text"], "review_id": r["review_id"],
                                                          "provider": "mock"}).text))["done"]
    assert done["analysis"]["review"]["stars"] == r["stars"] and done["analysis"]["label"] == r["label"]


def test_batch_run_then_results_and_metrics(client):
    assert client.get("/api/metrics").json() == {"n": 0}
    client.post("/api/runs", json={"n": 15, "workers": 2, "provider": "mock"}).raise_for_status()
    job = _wait_for_job(client)
    assert job["done"] == 15 and job["error"] is None
    results = client.get("/api/results").json()
    assert len(results) == 15 and all(r["stars_calibrated"] is not None for r in results)
    m = client.get("/api/metrics").json()
    assert m["n"] == 15 and "oof_predictions" not in m["calibrated"]
    assert set(m["lead_agent"]) >= {"accuracy", "accuracy_ci", "macro_f1", "report", "confusion"}
    assert sum(map(sum, m["lead_agent"]["confusion"])) == 15
    assert m["grounding"]["rate"] == 1.0 and m["businesses"]
    json.dumps(m, allow_nan=False)                 # strictly valid JSON (no NaN)

    c = client.get("/api/compare").json()          # 4 businesses, 3-4 reviews each
    assert c["summary"]["restaurants"] == 4 and c["summary"]["reviews"] == 15
    assert sorted(len(x["review_ids"]) for x in c["restaurants"]) == [3, 4, 4, 4]
    assert len(c["reviews"]) == 15 and c["source"].endswith("results.jsonl")
    assert set(c["summary"]["aspects"]) == {"food", "service", "ambience"}
    json.dumps(c, allow_nan=False)

    # a live analysis now also gets a calibrated star prediction
    done = dict(_events(client.post("/api/analyze", json={"text": "Great food.", "provider": "mock"}).text))["done"]
    assert done["calibrated_stars"] is not None and done["n_train"] == 15


def test_bad_provider_is_a_400(client):
    assert client.post("/api/analyze", json={"text": "hello there", "provider": "nvidia"}).status_code == 400


def test_compare_reads_the_restaurant_results_when_present(tmp_path, monkeypatch):
    from yelp_agents import Orchestrator
    from yelp_core import Settings, build_llm
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    s = Settings()
    s.provider = "mock"
    restaurants = tmp_path / "restaurant_results.jsonl"
    Orchestrator(build_llm(s)).run([Review(f"x{i}", "bx", 1 + i, TEXTS[1 + i], "Only Biz", "Reno", business_stars=3.5)
                                    for i in range(4)], restaurants)
    app = create_app(tmp_path / "sample.jsonl", tmp_path / "results.jsonl", restaurants)
    client = TestClient(app)
    assert client.get("/api/results").json() == []          # other tabs keep the main results
    c = client.get("/api/compare").json()
    assert c["source"] == str(restaurants) and [x["business"] for x in c["restaurants"]] == ["Only Biz"]
    assert c["restaurants"][0]["yelp_stars"] == 3.5 and len(c["reviews"]) == 4
    missing = TestClient(create_app(tmp_path / "s.jsonl", tmp_path / "results.jsonl", tmp_path / "nope.jsonl"))
    assert missing.get("/api/compare").json()["restaurants"] == []   # falls back to the (empty) main results
