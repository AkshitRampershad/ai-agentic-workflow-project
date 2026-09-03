"""Configuration helpers for LLM access.

Defaults to Groq (OpenAI-compatible chat completions, free tier
available) since Vocareum's gateway is only reachable with a Udacity
course subscription. Any OpenAI-compatible provider - Groq, OpenAI
itself, Vocareum, etc. - works by setting the matching env vars; the
first API key found below wins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv
try:
    from openai import OpenAI
except ModuleNotFoundError:  # Allows offline tests before dependencies are installed.
    OpenAI = None

load_dotenv()

_DEFAULT_BASE_URLS = {
    "GROQ_API_KEY": "https://api.groq.com/openai/v1",
    "VOC_API_KEY": "https://openai.vocareum.com/v1",
    "OPENAI_API_KEY": "https://api.openai.com/v1",
}
_DEFAULT_MODELS = {
    # llama-3.3-70b-versatile was deprecated by Groq (June 2026); this is
    # their recommended replacement for general-purpose/agentic use.
    "GROQ_API_KEY": "openai/gpt-oss-120b",
    "VOC_API_KEY": "gpt-4o-mini",
    "OPENAI_API_KEY": "gpt-4o-mini",
}


def _detect_provider() -> str:
    """Pick the first configured provider's env-var name, in preference
    order (Groq first - it's the one this project ships working
    defaults for).
    """
    for env_var in ("GROQ_API_KEY", "VOC_API_KEY", "OPENAI_API_KEY"):
        if os.getenv(env_var):
            return env_var
    return "GROQ_API_KEY"


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for LLM calls. Any field left as None picks
    up a provider-appropriate default independently - passing just
    api_key explicitly, for example, still resolves base_url/model from
    whichever provider that key's env var indicates (Groq by default).
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    def __post_init__(self):
        provider = _detect_provider()
        if self.api_key is None:
            object.__setattr__(self, "api_key", os.getenv(provider))
        if self.base_url is None:
            object.__setattr__(self, "base_url", os.getenv("OPENAI_BASE_URL") or _DEFAULT_BASE_URLS[provider])
        if self.model is None:
            object.__setattr__(self, "model", os.getenv("OPENAI_MODEL") or _DEFAULT_MODELS[provider])


def build_client(config: LLMConfig | None = None):
    """Build an OpenAI-compatible client.

    Returns None when no API key is configured so tests can run in offline/mock mode.
    """

    cfg = config or LLMConfig()
    if not cfg.api_key or OpenAI is None:
        return None
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
