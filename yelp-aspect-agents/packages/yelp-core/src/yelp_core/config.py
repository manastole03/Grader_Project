"""Environment-driven settings. Reads a .env file at the repo root if present."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency); never overrides real env vars."""
    path = path or Path.cwd() / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    provider: str = "auto"  # auto | asu | nvidia | hf | ollama | mock
    model: str | None = None
    temperature: float = 0.1
    timeout_s: float = 120.0
    max_retries: int = 4

    # ASU Research Computing OpenAI-compatible endpoint (key from ASU_API_KEY, or API_KEY in .env)
    asu_api_key: str | None = None
    asu_base_url: str = "https://openai.rc.asu.edu/v1"
    asu_model: str = "gemma4-31b-it"

    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"

    hf_token: str | None = None
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:latest"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        e = os.environ.get
        s = cls()
        s.provider = e("LLM_PROVIDER", s.provider).lower()
        s.model = e("LLM_MODEL") or None
        s.temperature = float(e("LLM_TEMPERATURE", s.temperature))
        s.timeout_s = float(e("LLM_TIMEOUT_S", s.timeout_s))
        s.asu_api_key = e("ASU_API_KEY") or e("API_KEY") or None
        s.asu_base_url = e("ASU_BASE_URL", s.asu_base_url)
        s.asu_model = e("ASU_MODEL", s.asu_model)
        s.nvidia_api_key = e("NVIDIA_API_KEY") or None
        s.nvidia_base_url = e("NVIDIA_BASE_URL", s.nvidia_base_url)
        s.nvidia_model = e("NVIDIA_MODEL", s.nvidia_model)
        s.hf_token = e("HF_TOKEN") or e("HUGGINGFACEHUB_API_TOKEN") or None
        s.hf_base_url = e("HF_BASE_URL", s.hf_base_url)
        s.hf_model = e("HF_MODEL", s.hf_model)
        s.ollama_base_url = e("OLLAMA_BASE_URL", s.ollama_base_url)
        s.ollama_model = e("OLLAMA_MODEL", s.ollama_model)
        return s
