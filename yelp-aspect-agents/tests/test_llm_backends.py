"""Backend selection and the ASU OpenAI-compatible client (HTTP is faked; no network)."""
import pytest

from yelp_core import Settings, build_llm
from yelp_core import llm as llm_mod
from yelp_core.llm import LLMError

FAKE_KEY = "sk-test-not-a-real-key"


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers, self.text = status, payload or {}, headers or {}, ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise llm_mod.requests.HTTPError(str(self.status_code))


def ok(content):
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def settings(**kw) -> Settings:
    s = Settings()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_auto_prefers_asu_when_key_present():
    c = build_llm(settings(asu_api_key=FAKE_KEY, nvidia_api_key="n"))
    assert (c.provider, c.model) == ("asu", "gemma4-31b-it")
    assert FAKE_KEY not in repr(c)
    assert build_llm(settings(asu_api_key=FAKE_KEY, model="llama4-maverick-17b")).model == "llama4-maverick-17b"
    with pytest.raises(LLMError, match="API_KEY"):
        build_llm(settings(provider="asu"))


def test_key_read_from_api_key_or_asu_api_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    for k in ("ASU_API_KEY", "API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("API_KEY", "a")
    assert Settings.from_env().asu_api_key == "a"
    monkeypatch.setenv("ASU_API_KEY", "b")
    assert Settings.from_env().asu_api_key == "b"


def test_json_mode_and_auth_header(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return ok('{"mentions": []}')
    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    c = build_llm(settings(asu_api_key=FAKE_KEY))
    assert c.chat_json([{"role": "user", "content": "hi"}]) == {"mentions": []}
    assert seen["url"] == "https://openai.rc.asu.edu/v1/chat/completions"
    assert seen["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["model"] == "gemma4-31b-it"


def test_rate_limit_waits_then_succeeds(monkeypatch):
    calls, sleeps = [], []
    responses = [FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(429), ok('{"a": 1}')]
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: calls.append(1) or responses.pop(0))
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    assert build_llm(settings(asu_api_key=FAKE_KEY)).chat_json([{"role": "user", "content": "x"}]) == {"a": 1}
    assert len(calls) == 3 and sleeps == [3.0, 4.0]  # server's Retry-After, then exponential backoff


def test_bad_key_fails_fast_without_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: calls.append(1) or FakeResponse(401))
    with pytest.raises(PermissionError, match="API key"):
        build_llm(settings(asu_api_key=FAKE_KEY)).chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_dotenv_strips_inline_comments_and_quotes(monkeypatch, tmp_path):
    from yelp_core.config import load_dotenv
    env = tmp_path / ".env"
    env.write_text("API_KEY=sk-abc123                # or ASU_API_KEY\n"
                   "ASU_MODEL=\"gemma4-31b-it\"  # quoted\n"
                   "LLM_MODEL=            # overrides the model\n"
                   "HF_TOKEN='a#b'\n")
    for k in ("API_KEY", "ASU_MODEL", "LLM_MODEL", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(env)
    import os
    assert os.environ["API_KEY"] == "sk-abc123" and os.environ["ASU_MODEL"] == "gemma4-31b-it"
    assert os.environ["LLM_MODEL"] == "" and os.environ["HF_TOKEN"] == "a#b"
