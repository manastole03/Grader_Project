from __future__ import annotations

from yelp_agents.base import Agent
from yelp_agents.prompts import ARBITER_SYSTEM, ARBITER_USER, ASPECT_DEFINITIONS


class ArbiterAgent(Agent):
    """Resolves evidence cited by more than one aspect agent."""

    name = "arbiter-agent"

    def run(self, review_text: str, quote: str, candidates: list[str]) -> list[str]:
        """Aspects (a subset of *candidates*) the quote really belongs to.

        On failure or an empty/invalid answer every candidate is kept, i.e. no change.
        """
        msgs = self.messages(ARBITER_SYSTEM.format(**ASPECT_DEFINITIONS), [], ARBITER_USER.format(
            review=review_text, quote=quote, candidates=", ".join(candidates)))
        try:
            raw = self.llm.chat_json(msgs, max_tokens=80)
        except Exception:
            return list(candidates)
        chosen = [a for a in raw.get("aspects", []) if a in candidates] if isinstance(raw.get("aspects"), list) else []
        return chosen or list(candidates)
