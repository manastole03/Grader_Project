from __future__ import annotations

from yelp_core.llm import LLMClient, Message


class Agent:
    """An LLM-backed worker with a fixed role (system prompt) and a structured output."""

    name = "agent"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def messages(self, system: str, shots: list[tuple[str, str]], user: str) -> list[Message]:
        msgs: list[Message] = [{"role": "system", "content": system}]
        for q, a in shots:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        return msgs
