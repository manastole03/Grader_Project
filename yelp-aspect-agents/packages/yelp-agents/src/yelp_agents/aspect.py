from __future__ import annotations

from yelp_core.llm import LLMClient
from yelp_core.schemas import AspectResult
from yelp_scoring import build_aspect_result

from yelp_agents.base import Agent
from yelp_agents.prompts import ASPECT_DEFINITIONS, ASPECT_SYSTEM, ASPECT_USER, FEW_SHOT, INTENSITY_RUBRIC


class AspectAgent(Agent):
    """Extracts opinion mentions for one dimension; Python grounds and scores them."""

    def __init__(self, aspect: str, llm: LLMClient, few_shot: bool = True):
        super().__init__(llm)
        if aspect not in ASPECT_DEFINITIONS:
            raise ValueError(f"unknown aspect {aspect!r}")
        self.aspect = aspect
        self.name = f"{aspect}-agent"
        self.few_shot = few_shot

    def build_messages(self, review_text: str):
        system = ASPECT_SYSTEM.format(aspect=self.aspect, aspect_upper=self.aspect.upper(),
                                      definition=ASPECT_DEFINITIONS[self.aspect], rubric=INTENSITY_RUBRIC)
        shots = [(ASPECT_USER.format(aspect=self.aspect, review=q), a)
                 for q, a in FEW_SHOT[self.aspect]] if self.few_shot else []
        return self.messages(system, shots, ASPECT_USER.format(aspect=self.aspect, review=review_text))

    def run(self, review_text: str) -> AspectResult:
        try:
            raw = self.llm.chat_json(self.build_messages(review_text), max_tokens=500)
        except Exception as exc:  # one failed dimension must not sink the whole review
            return AspectResult(aspect=self.aspect, error=str(exc)[:300])
        mentions = raw.get("mentions")
        if mentions is None and "quote" in raw:  # model returned a single mention object
            mentions = [raw]
        if not isinstance(mentions, list):
            return AspectResult(aspect=self.aspect, error=f"malformed output: {str(raw)[:200]}")
        return build_aspect_result(self.aspect, review_text, mentions)
