from __future__ import annotations

import json

from yelp_core.schemas import AspectResult, OverallResult
from yelp_scoring import build_overall, stars_rule

from yelp_agents.base import Agent
from yelp_agents.prompts import AGGREGATOR_SYSTEM, AGGREGATOR_USER


class AggregatorAgent(Agent):
    """Lead agent: gives the overall polarity + intensity; Python turns it into a score."""

    name = "lead-agent"

    def run(self, review_text: str, aspects: dict[str, AspectResult]) -> OverallResult:
        payload = {a: {"label": r.sentiment,
                       "mentions": [{"quote": m.quote, "polarity": m.polarity, "intensity": m.intensity}
                                    for m in r.mentions]}
                   for a, r in aspects.items()}
        msgs = self.messages(AGGREGATOR_SYSTEM, [], AGGREGATOR_USER.format(
            review=review_text, aspects_json=json.dumps(payload, indent=1)))
        try:
            raw = self.llm.chat_json(msgs, max_tokens=300)
        except Exception as exc:
            return self.fallback(aspects, error=str(exc)[:300])
        overall = build_overall(raw)
        return overall or self.fallback(aspects, error=f"malformed output: {str(raw)[:200]}")

    @staticmethod
    def fallback(aspects: dict[str, AspectResult], error: str | None = None) -> OverallResult:
        """Used only when the lead agent fails: mean of the mentioned aspect scores."""
        scores = [r.score for r in aspects.values() if r.mentioned]
        score = round(sum(scores) / len(scores), 4) if scores else 0.0
        sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        return OverallResult(polarity=sentiment, intensity=0, score=score, sentiment=sentiment,
                             stars_rule=stars_rule(score),
                             rationale="fallback: mean of mentioned aspect scores", error=error)
