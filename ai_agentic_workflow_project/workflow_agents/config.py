"""Configuration helpers for Vocareum/OpenAI access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv
try:
    from openai import OpenAI
except ModuleNotFoundError:  # Allows offline tests before dependencies are installed.
    OpenAI = None

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for LLM calls."""

    api_key: str | None = os.getenv("VOC_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))


def build_client(config: LLMConfig | None = None):
    """Build an OpenAI-compatible client.

    Returns None when no API key is configured so tests can run in offline/mock mode.
    """

    cfg = config or LLMConfig()
    if not cfg.api_key or OpenAI is None:
        return None
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
