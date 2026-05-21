"""Reusable LLM service wrapper with offline fallback support."""

from __future__ import annotations

import json
from typing import Any

from .config import LLMConfig, build_client


class LLMService:
    """Centralized generation service used by all agents."""

    def __init__(self, config: LLMConfig | None = None, offline_mode: bool = False):
        self.config = config or LLMConfig()
        self.client = None if offline_mode else build_client(self.config)
        self.offline_mode = offline_mode or self.client is None

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response from an OpenAI-compatible chat completion endpoint."""

        if self.offline_mode:
            return self._offline_response(prompt)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def _offline_response(self, prompt: str) -> str:
        """Deterministic response for tests and budget-safe demos."""

        lowered = prompt.lower()
        if "evaluate" in lowered or "score" in lowered:
            return json.dumps({
                "score": 8,
                "passed": True,
                "feedback": "Output is structured, testable, and aligned with project-management criteria."
            }, indent=2)
        if "engineering task" in lowered or "development engineer" in lowered:
            return """1. Implement email ingestion connector with mailbox/API integration.\n2. Build parser for headers, body, metadata, and attachment metadata.\n3. Implement classification service with confidence scoring.\n4. Create routing-rule engine and queue assignment logic.\n5. Add audit logging for decision traceability.\n6. Build admin configuration screens and reporting APIs."""
        if "feature" in lowered:
            return """Feature 1: Intelligent Email Classification\nFeature 2: Rule-Based Routing Engine\nFeature 3: Human Review Exception Queue\nFeature 4: Audit Logging and Decision Traceability\nFeature 5: Admin Routing Configuration\nFeature 6: Operational Reporting Dashboard"""
        if "user stor" in lowered:
            return """As a support representative, I want inbound emails automatically categorized so that I can prioritize work quickly.\nAs an operations manager, I want uncertain emails placed into a review queue so that routing errors are controlled.\nAs an administrator, I want to configure routing rules so that queues reflect current business operations.\nAs a TPM, I want audit logs for every routing decision so that compliance and debugging are easier."""
        if "plan" in lowered or "break down" in lowered:
            return """1. Analyze the product specification.\n2. Generate user stories.\n3. Define product features.\n4. Create engineering implementation tasks.\n5. Evaluate each artifact for completeness and feasibility.\n6. Compile final project backlog."""
        if "route" in lowered:
            return json.dumps({
                "route": "product_manager_team",
                "reason": "The task requires translating product context into planning artifacts."
            }, indent=2)
        return "Generated response based on the supplied project-management context."
