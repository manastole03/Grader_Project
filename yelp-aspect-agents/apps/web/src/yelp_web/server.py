"""FastAPI backend for the web UI. Every number the UI shows is computed here, in Python
(yelp_scoring / yelp_eval); the browser only renders it.

  GET  /                       single-page UI (static/)
  GET  /api/status             backends available, sample / result counts
  GET  /api/sample/random      a random review from the sample (with its real stars)
  POST /api/analyze            run the agent team on one review; Server-Sent Events stream
                               (start, agent_start, aspect, arbiter, aggregator_start, overall, done)
  GET  /api/results            every analysis so far, with out-of-fold calibrated stars
  GET  /api/metrics            evaluation vs stars (with 95% CIs) + business roll-up
  GET  /api/compare            reviews of the same restaurant compared, per aspect (from the
                               restaurant sample's results when they exist, else from /api/results)
  POST /api/runs               start a batch run over the sample in the background
  GET  /api/runs/current       progress of that batch run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from yelp_agents import Orchestrator
from yelp_agents.prompts import ASPECT_DEFINITIONS
from yelp_core import ASPECTS, LLMClient, Review, ReviewAnalysis, Settings, build_llm
from yelp_core.llm import LLMError, OllamaClient
from yelp_data import load_reviews
from yelp_eval import business_rollup, evaluate, load_results
from yelp_eval.compare import compare_restaurants
from yelp_scoring import StarModel

STATIC = Path(__file__).parent / "static"
# Other ASU-hosted models offered in the UI for comparison (all tested with this pipeline).
ASU_ALTERNATIVES = ("qwen3-235b-a22b-instruct-2507", "llama4-maverick-17b")


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=8000)
    review_id: str | None = None   # set when the text is an unedited review from the sample
    provider: str | None = None
    few_shot: bool = True
    arbitrate: bool = True


class RunRequest(BaseModel):
    n: int = Field(default=20, ge=1, le=5000)
    workers: int = Field(default=2, ge=1, le=16)
    provider: str | None = None
    few_shot: bool = True
    arbitrate: bool = True


class State:
    def __init__(self, sample_path: Path, results_path: Path, compare_path: Path | None = None):
        self.sample_path = sample_path
        self.results_path = results_path
        self.compare_path = compare_path
        self._llms: dict[str, LLMClient] = {}
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple[str, float], Any]] = {}
        self.job: dict[str, Any] = {"running": False}

    def llm(self, provider: str | None) -> LLMClient:
        """provider: None (LLM_PROVIDER / .env default), a provider id, or "provider:model"."""
        key = provider or "default"
        with self._lock:
            if key not in self._llms:
                s = Settings.from_env()
                if provider:
                    s.provider, _, model = provider.partition(":")
                    s.model = model or None
                self._llms[key] = build_llm(s)
            return self._llms[key]

    def _cached(self, name: str, build, path: Path | None = None) -> Any:
        """Recompute derived data only when its source file (default: the results file) changes."""
        path = path or self.results_path
        stamp = (str(path), path.stat().st_mtime if path.exists() else 0.0)
        hit = self._cache.get(name)
        if hit is None or hit[0] != stamp:
            self._cache[name] = (stamp, build())
        return self._cache[name][1]

    def results(self) -> list[ReviewAnalysis]:
        return self._cached("results", lambda: load_results(self.results_path) if self.results_path.exists() else [])

    def metrics(self) -> dict[str, Any]:
        return self._cached("metrics", lambda: evaluate(self.results()))

    def compare_source(self) -> Path:
        """The restaurant sample's results if they exist, so the Compare tab works next to the
        stratified sample; otherwise the main results file."""
        p = self.compare_path
        return p if p is not None and p.exists() else self.results_path

    def compare_results(self) -> list[ReviewAnalysis]:
        src = self.compare_source()
        return self._cached("compare_results", lambda: load_results(src) if src.exists() else [], src)

    def comparison(self) -> dict[str, Any]:
        src = self.compare_source()
        return self._cached("compare", lambda: compare_restaurants(self.compare_results()), src)

    def star_model(self) -> StarModel | None:
        rs = self.results()
        return self._cached("model", lambda: StarModel().fit(rs) if StarModel.usable(rs) else None)

    def sample(self) -> list[Review]:
        return load_reviews(self.sample_path) if self.sample_path.exists() else []


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python numbers, tuples -> lists."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def create_app(sample_path: Path, results_path: Path, compare_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Yelp Aspect Agents")
    st = State(sample_path, results_path, compare_path)
    app.state.yelp = st

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        s = Settings.from_env()
        available = []
        if s.asu_api_key:
            for m in dict.fromkeys([s.model or s.asu_model, *ASU_ALTERNATIVES]):
                available.append({"id": f"asu:{m}", "label": f"ASU · {m}"})
        if s.nvidia_api_key:
            available.append({"id": "nvidia", "label": f"NVIDIA NIM · {s.model or s.nvidia_model}"})
        if s.hf_token:
            available.append({"id": "hf", "label": f"Hugging Face · {s.model or s.hf_model}"})
        if OllamaClient.reachable(s.ollama_base_url):
            available.append({"id": "ollama", "label": f"Ollama · {s.model or s.ollama_model}"})
        available.append({"id": "mock", "label": "Offline keyword baseline"})
        default = st.llm(None)
        default_id = f"{default.provider}:{default.model}" if default.provider == "asu" else default.provider
        return {"default": {"provider": default.provider, "model": default.model, "id": default_id},
                "available": available, "aspects": {a: ASPECT_DEFINITIONS[a] for a in ASPECTS},
                "sample_size": len(st.sample()), "results": len(st.results()), "job": st.job}

    @app.get("/api/sample/random")
    def random_review() -> dict[str, Any]:
        sample = st.sample()
        if not sample:
            raise HTTPException(404, "no sample yet: run `yelp-agents sample` first")
        r = random.choice(sample)
        return {**r.__dict__, "label": r.label}

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest) -> StreamingResponse:
        try:
            llm = st.llm(req.provider)
        except LLMError as exc:
            raise HTTPException(400, str(exc)) from exc
        review = next((r for r in st.sample() if r.review_id == req.review_id
                       and r.text.strip() == req.text.strip()), None) if req.review_id else None
        review = review or Review("custom", "custom", 0.0, req.text.strip())
        orch = Orchestrator(llm, few_shot=req.few_shot, arbitrate=req.arbitrate)

        def stream() -> Iterator[str]:
            t0 = time.perf_counter()
            yield _sse("start", {"provider": llm.provider, "model": llm.model})
            for a in ASPECTS:
                yield _sse("agent_start", {"aspect": a})
            t_lead = t0
            for kind, payload in orch.analyze_steps(review):
                if kind == "aspect":
                    yield _sse("aspect", {**asdict(payload), "latency_s": round(time.perf_counter() - t0, 2)})
                elif kind == "arbiter":
                    yield _sse("arbiter", payload)
                elif kind == "lead_start":
                    t_lead = time.perf_counter()
                    yield _sse("aggregator_start", {})
                elif kind == "overall":
                    yield _sse("overall", {**asdict(payload), "latency_s": round(time.perf_counter() - t_lead, 2)})
                elif kind == "done":
                    model = st.star_model()
                    calibrated = None
                    if model is not None:
                        # never predict a review with a model that was trained on it
                        train = [r for r in st.results() if r.review.review_id != review.review_id]
                        m = model if len(train) == len(st.results()) else (
                            StarModel().fit(train) if StarModel.usable(train) else None)
                        calibrated = round(float(m.predict([payload])[0]), 3) if m else None
                    yield _sse("done", {"analysis": payload.to_dict(), "calibrated_stars": calibrated,
                                        "n_train": len(st.results()) if calibrated is not None else 0,
                                        "latency_s": round(time.perf_counter() - t0, 2)})

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/results")
    def results() -> list[dict[str, Any]]:
        oof = (st.metrics().get("calibrated") or {}).get("oof_predictions", {})
        return [{**r.to_dict(), "stars_calibrated": oof.get(r.review.review_id)} for r in st.results()]

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        m = st.metrics()
        if not m.get("n"):
            return {"n": 0}
        out = {k: v for k, v in m.items() if k != "calibrated"}
        if m.get("calibrated"):
            out["calibrated"] = {k: v for k, v in m["calibrated"].items() if k != "oof_predictions"}
        biz = business_rollup(st.results(), min_reviews=1)
        out["businesses"] = json.loads(biz.to_json(orient="records")) if len(biz) else []
        return _clean(out)

    @app.get("/api/compare")
    def compare() -> dict[str, Any]:
        c = st.comparison()
        ids = {i for x in c["restaurants"] for i in x["review_ids"]}
        reviews = [r.to_dict() for r in st.compare_results() if r.review.review_id in ids]
        return _clean({**c, "source": str(st.compare_source()), "reviews": reviews})

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict[str, Any]:
        if st.job.get("running"):
            raise HTTPException(409, "a batch run is already in progress")
        sample = st.sample()
        if not sample:
            raise HTTPException(404, "no sample yet: run `yelp-agents sample` first")
        try:
            llm = st.llm(req.provider)
        except LLMError as exc:
            raise HTTPException(400, str(exc)) from exc
        reviews = sample[: req.n]
        st.job = {"running": True, "done": 0, "total": len(reviews), "error": None,
                  "backend": repr(llm), "started": time.time(), "last": None}

        def on_result(i: int, n: int, a: ReviewAnalysis) -> None:
            st.job.update(done=i, total=n, last={
                "business": a.review.business_name, "stars": a.review.stars,
                "sentiment": a.overall.sentiment, "vector": a.vector})

        def work() -> None:
            try:
                Orchestrator(llm, few_shot=req.few_shot, arbitrate=req.arbitrate).run(
                    reviews, st.results_path, workers=req.workers, on_result=on_result)
            except Exception as exc:  # surfaced to the UI
                st.job["error"] = str(exc)[:300]
            finally:
                st.job["running"] = False
                st.job["finished"] = time.time()

        threading.Thread(target=work, daemon=True).start()
        return st.job

    @app.get("/api/runs/current")
    def current_run() -> dict[str, Any]:
        return st.job

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main() -> None:
    import uvicorn
    p = argparse.ArgumentParser(prog="yelp-web")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--restaurants", action="store_true",
                   help="use the restaurant sample (sample-restaurants) and its results in every tab")
    p.add_argument("--sample", type=Path)
    p.add_argument("--results", type=Path)
    p.add_argument("--compare-results", type=Path, default=Path("outputs/restaurant_results.jsonl"),
                   help="results the Compare tab reads (falls back to --results if missing)")
    a = p.parse_args()
    a.sample = a.sample or Path("data/restaurant_reviews.jsonl" if a.restaurants else "data/sample_reviews.jsonl")
    a.results = a.results or Path("outputs/restaurant_results.jsonl" if a.restaurants else "outputs/results.jsonl")
    print(f"Yelp Aspect Agents UI -> http://{a.host}:{a.port}")
    uvicorn.run(create_app(a.sample, a.results, a.compare_results), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
