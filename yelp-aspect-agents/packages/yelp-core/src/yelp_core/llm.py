"""LLM provider clients behind one tiny interface.

Resolution order for provider="auto":
    1. ASU               (ASU_API_KEY or API_KEY set) - openai.rc.asu.edu, OpenAI-compatible
    2. NVIDIA NIM        (NVIDIA_API_KEY set)
    3. Hugging Face      (HF_TOKEN set)            - OpenAI-compatible router
    4. Ollama            (local server reachable)  - e.g. gemma4
    5. mock              (offline keyword heuristic, for tests / dry runs)
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from yelp_core.config import Settings

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMClient(ABC):
    provider: str = "base"

    def __init__(self, model: str, settings: Settings):
        self.model = model
        self.settings = settings

    @abstractmethod
    def _chat(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str: ...

    def chat(self, messages: list[Message], json_mode: bool = False, max_tokens: int = 400) -> str:
        last: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return self._chat(messages, json_mode, max_tokens)
            except RateLimited as exc:  # wait as long as the server asks (exponential otherwise)
                last = exc
                time.sleep(exc.retry_after or min(60.0, 2.0 * 2 ** attempt))
            except (requests.RequestException, LLMError, KeyError, ValueError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"{self.provider} call failed after retries: {last}") from last

    def chat_json(self, messages: list[Message], max_tokens: int = 400) -> dict[str, Any]:
        return extract_json(self.chat(messages, json_mode=True, max_tokens=max_tokens))

    def __repr__(self) -> str:
        return f"{self.provider}:{self.model}"


class OpenAICompatClient(LLMClient):
    """Works for ASU, NVIDIA NIM and the Hugging Face router (all speak /v1/chat/completions)."""

    def __init__(self, provider: str, base_url: str, api_key: str, model: str, settings: Settings,
                 json_mode: bool = False):
        super().__init__(model, settings)
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.supports_json_mode = json_mode  # send response_format={"type": "json_object"}

    def __repr__(self) -> str:  # never include the key
        return f"{self.provider}:{self.model}"

    def _chat(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and self.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=self.settings.timeout_s,
        )
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            raise RateLimited("HTTP 429 rate limited", float(retry) if retry and retry.isdigit() else None)
        if r.status_code >= 500:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code in (401, 403):
            raise PermissionError(f"{self.provider}: HTTP {r.status_code} - check the API key")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, base_url: str, model: str, settings: Settings):
        super().__init__(model, settings)
        self.base_url = base_url.rstrip("/")

    def _chat(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # gemma4 / qwen3 style thinking off: we want terse JSON
            "options": {"temperature": self.settings.temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.settings.timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"]

    @staticmethod
    def reachable(base_url: str) -> bool:
        try:
            return requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False


class MockClient(LLMClient):
    """Deterministic keyword baseline so the whole pipeline runs with no model at all.

    Agents wrap inputs in <aspect>/<review>/<aspects_json> tags; this client reads them.
    """

    provider = "mock"
    LEXICON = {
        "food": ["food", "dish", "meal", "taste", "flavor", "pizza", "burger", "pasta", "sushi",
                 "chicken", "menu", "delicious", "fresh", "portion", "cooked", "sauce", "coffee"],
        "service": ["service", "staff", "waiter", "waitress", "server", "manager", "friendly",
                    "rude", "wait", "attentive", "host", "bartender", "owner", "slow"],
        "ambience": ["ambience", "ambiance", "atmosphere", "decor", "music", "clean", "dirty",
                     "loud", "noisy", "cozy", "vibe", "seating", "patio", "view", "place"],
    }
    POS = {"good", "great", "amazing", "delicious", "friendly", "love", "loved", "excellent",
           "best", "fresh", "nice", "clean", "cozy", "attentive", "perfect", "awesome", "tasty"}
    NEG = {"bad", "terrible", "awful", "rude", "slow", "cold", "dirty", "worst", "bland",
           "horrible", "disappointing", "loud", "noisy", "overpriced", "never", "gross", "stale"}

    def __init__(self, settings: Settings):
        super().__init__("keyword-heuristic", settings)

    @staticmethod
    def _tag(text: str, name: str) -> str:
        m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
        return m.group(1).strip() if m else ""

    def _polarity(self, sentence: str) -> int:
        words = re.findall(r"[a-z']+", sentence.lower())
        return sum(w in self.POS for w in words) - sum(w in self.NEG for w in words)

    def _chat(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        prompt = messages[-1]["content"]
        if "<candidates>" in prompt:  # arbiter: keep candidates whose lexicon the passage uses
            quote = self._tag(prompt, "quote").lower()
            cands = [c.strip() for c in self._tag(prompt, "candidates").split(",")]
            return json.dumps({"aspects": [c for c in cands if any(k in quote for k in self.LEXICON.get(c, []))]})
        aspect = self._tag(prompt, "aspect")
        review = self._tag(prompt, "review")
        if aspect:
            mentions = []
            for s in re.split(r"(?<=[.!?])\s+", review):
                if not any(k in s.lower() for k in self.LEXICON.get(aspect, [])):
                    continue
                pol = self._polarity(s)
                mentions.append({"quote": s.strip(),
                                 "polarity": "positive" if pol > 0 else "negative" if pol < 0 else "neutral",
                                 "intensity": max(1, min(3, abs(pol)))})
            return json.dumps({"mentions": mentions})
        aspects = json.loads(self._tag(prompt, "aspects_json") or "{}")
        sign = {"positive": 1, "negative": -1, "neutral": 0}
        vals = [sign[m["polarity"]] * m["intensity"] for v in aspects.values() for m in v.get("mentions", [])]
        mean = sum(vals) / len(vals) if vals else self._polarity(review)
        polarity = "positive" if mean > 0.5 else "negative" if mean < -0.5 else "neutral"
        return json.dumps({"polarity": polarity, "intensity": max(1, min(3, round(abs(mean)))),
                           "other_factors": [], "rationale": "keyword heuristic over aspect mentions"})


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply (tolerates ```json fences and chatter)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


def build_llm(settings: Settings | None = None) -> LLMClient:
    s = settings or Settings.from_env()
    p = s.provider
    if p in ("auto", "asu") and s.asu_api_key:
        return OpenAICompatClient("asu", s.asu_base_url, s.asu_api_key, s.model or s.asu_model, s, json_mode=True)
    if p == "asu":
        raise LLMError("LLM_PROVIDER=asu but neither ASU_API_KEY nor API_KEY is set")
    if p in ("auto", "nvidia") and s.nvidia_api_key:
        return OpenAICompatClient("nvidia", s.nvidia_base_url, s.nvidia_api_key, s.model or s.nvidia_model, s)
    if p == "nvidia":
        raise LLMError("LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set")
    if p in ("auto", "hf") and s.hf_token:
        return OpenAICompatClient("hf", s.hf_base_url, s.hf_token, s.model or s.hf_model, s)
    if p == "hf":
        raise LLMError("LLM_PROVIDER=hf but HF_TOKEN is not set")
    if p in ("auto", "ollama") and OllamaClient.reachable(s.ollama_base_url):
        return OllamaClient(s.ollama_base_url, s.model or s.ollama_model, s)
    if p == "ollama":
        raise LLMError(f"Ollama not reachable at {s.ollama_base_url} (run `ollama serve`)")
    if p in ("auto", "mock"):
        return MockClient(s)
    raise LLMError(f"unknown LLM_PROVIDER={p!r}")
