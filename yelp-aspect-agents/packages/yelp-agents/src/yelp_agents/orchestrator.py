"""Coordinates the agent team.

Per review:
    food-agent ─┐
    service-agent ├─(parallel)─> Python: ground + score ─> arbiter-agent (only if two
    ambience-agent┘                                         aspects cite the same text)
                                                            ─> lead-agent ─> Python: overall score
Across reviews: a worker pool; results are appended to JSONL as they finish, so an
interrupted run resumes where it stopped.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from yelp_core.llm import LLMClient
from yelp_core.schemas import ASPECTS, AspectResult, Review, ReviewAnalysis
from yelp_scoring import rescore

from yelp_agents.aggregator import AggregatorAgent
from yelp_agents.arbiter import ArbiterAgent
from yelp_agents.aspect import AspectAgent

# Bump when the stored result format changes; older result files are not resumed into.
SCHEMA_VERSION = 2

Event = tuple[str, Any]


def _overlap(a: list[list[int]], b: list[list[int]]) -> bool:
    return any(x0 < y1 and y0 < x1 for x0, x1 in a for y0, y1 in b)


def find_conflicts(aspects: dict[str, AspectResult]) -> list[list[tuple[str, int]]]:
    """Groups of (aspect, mention index) whose evidence overlaps across different aspects."""
    nodes = [(a, i) for a, r in aspects.items() for i in range(len(r.mentions))]
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, (a1, m1) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            a2, m2 = nodes[j]
            if a1 != a2 and _overlap(aspects[a1].mentions[m1].spans, aspects[a2].mentions[m2].spans):
                parent[find(i)] = find(j)
    groups: dict[int, list[tuple[str, int]]] = {}
    for i, n in enumerate(nodes):
        groups.setdefault(find(i), []).append(n)
    return [g for g in groups.values() if len({a for a, _ in g}) > 1]


class Orchestrator:
    def __init__(self, llm: LLMClient, aggregator_llm: LLMClient | None = None,
                 few_shot: bool = True, aspects: Iterable[str] = ASPECTS, arbitrate: bool = True):
        # The lead agent may run on a different (e.g. larger) model, mirroring the
        # notebook's "chain two LLMs" pattern.
        self.aspect_agents = [AspectAgent(a, llm, few_shot=few_shot) for a in aspects]
        self.arbiter = ArbiterAgent(llm) if arbitrate else None
        self.aggregator = AggregatorAgent(aggregator_llm or llm)
        self.llm = llm

    def analyze_steps(self, review: Review) -> Iterator[Event]:
        """Yields ("aspect", AspectResult) as each agent finishes, ("arbiter", info) per conflict,
        ("lead_start", None), ("overall", OverallResult) and finally ("done", ReviewAnalysis)."""
        t0 = time.perf_counter()
        aspects: dict[str, AspectResult] = {}
        with ThreadPoolExecutor(max_workers=len(self.aspect_agents)) as pool:
            futures = {pool.submit(a.run, review.text): a.aspect for a in self.aspect_agents}
            for fut in as_completed(futures):
                res = fut.result()
                aspects[res.aspect] = res
                yield "aspect", res

        arbitrations = []
        if self.arbiter:
            for group in find_conflicts(aspects):
                spans = [s for a, i in group for s in aspects[a].mentions[i].spans]
                passage = review.text[min(s[0] for s in spans):max(s[1] for s in spans)]
                candidates = sorted({a for a, _ in group}, key=list(ASPECTS).index)
                keep = self.arbiter.run(review.text, passage, candidates)
                info = {"passage": passage, "candidates": candidates, "kept": keep}
                arbitrations.append(info)
                for a in candidates:
                    if a in keep:
                        continue
                    idx = sorted({i for x, i in group if x == a}, reverse=True)
                    for i in idx:
                        m = aspects[a].mentions.pop(i)
                        m.note = f"arbiter: passage is about {', '.join(keep)}"
                        aspects[a].dropped.append(m)
                    rescore(aspects[a])
                yield "arbiter", info

        yield "lead_start", None
        overall = self.aggregator.run(review.text, aspects)
        yield "overall", overall
        yield "done", ReviewAnalysis(
            review=review, aspects={a: aspects[a] for a in ASPECTS if a in aspects}, overall=overall,
            provider=self.llm.provider, model=self.llm.model,
            latency_s=round(time.perf_counter() - t0, 2),
            meta={"schema": SCHEMA_VERSION, "aggregator_model": repr(self.aggregator.llm),
                  "few_shot": self.aspect_agents[0].few_shot, "arbitrations": arbitrations},
        )

    def analyze(self, review: Review) -> ReviewAnalysis:
        for kind, payload in self.analyze_steps(review):
            if kind == "done":
                return payload
        raise RuntimeError("pipeline ended without a result")

    def run(self, reviews: list[Review], out_path: str | Path, workers: int = 2,
            on_result: Callable[[int, int, ReviewAnalysis], None] | None = None,
            retry_errors: bool = False) -> list[ReviewAnalysis]:
        """Analyse *reviews*, appending to *out_path*. Already-analysed reviews are skipped, unless
        retry_errors is set and their stored analysis had an agent error (the retry is appended;
        readers keep the latest line per review)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done: dict[str, ReviewAnalysis] = {}
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("meta", {}).get("schema") != SCHEMA_VERSION:
                    raise ValueError(f"{out_path} was written by an older version; use a new --out path")
                a = ReviewAnalysis.from_dict(d)
                done[a.review.review_id] = a
        def has_error(a: ReviewAnalysis) -> bool:
            return any(x.error for x in a.aspects.values()) or bool(a.overall.error)

        todo = [r for r in reviews if r.review_id not in done or (retry_errors and has_error(done[r.review_id]))]
        total, finished = len(reviews), len(reviews) - len(todo)

        # on_result runs in the calling thread, so callers can update progress state directly.
        with out_path.open("a") as f, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(self.analyze, r) for r in todo]
            for fut in as_completed(futures):
                analysis = fut.result()
                f.write(json.dumps(analysis.to_dict()) + "\n")
                f.flush()
                done[analysis.review.review_id] = analysis
                finished += 1
                if on_result:
                    on_result(finished, total, analysis)
        return [done[r.review_id] for r in reviews if r.review_id in done]
