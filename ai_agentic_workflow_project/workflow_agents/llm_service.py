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

    def generate(self, prompt: str, system_prompt: str | None = None, response_hint: str | None = None) -> str:
        """Generate a response from an OpenAI-compatible chat completion endpoint.

        response_hint lets a caller that knows exactly what kind of
        artifact it needs (routing, action_planning, user_stories, ...)
        pick the matching offline canned response deterministically,
        instead of relying on substring-sniffing the prompt - which
        breaks when unrelated context (e.g. a product spec containing
        the word "scores") happens to contain another category's
        trigger word. Ignored in live mode; only affects offline_mode.
        """

        if self.offline_mode:
            return self._offline_response(prompt, response_hint)

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

    _HINTED_RESPONSES = {
        "evaluation": lambda: json.dumps({
            "score": 8,
            "passed": True,
            "feedback": "Output is structured, testable, and aligned with project-management criteria."
        }, indent=2),
        "engineering_tasks": lambda: (
            "1. Implement email ingestion connector with mailbox/API integration.\n"
            "2. Build parser for headers, body, metadata, and attachment metadata.\n"
            "3. Implement classification service with confidence scoring.\n"
            "4. Create routing-rule engine and queue assignment logic.\n"
            "5. Add audit logging for decision traceability.\n"
            "6. Build admin configuration screens and reporting APIs."
        ),
        "product_features": lambda: (
            "Feature 1: Intelligent Email Classification\n"
            "Feature 2: Rule-Based Routing Engine\n"
            "Feature 3: Human Review Exception Queue\n"
            "Feature 4: Audit Logging and Decision Traceability\n"
            "Feature 5: Admin Routing Configuration\n"
            "Feature 6: Operational Reporting Dashboard"
        ),
        "user_stories": lambda: (
            "As a support representative, I want inbound emails automatically categorized so that I can prioritize work quickly.\n"
            "As an operations manager, I want uncertain emails placed into a review queue so that routing errors are controlled.\n"
            "As an administrator, I want to configure routing rules so that queues reflect current business operations.\n"
            "As a TPM, I want audit logs for every routing decision so that compliance and debugging are easier."
        ),
        "action_planning": lambda: json.dumps([
            {"id": "T1", "objective": "Generate user stories", "expected_output": "User stories with acceptance criteria", "recommended_team": "product_manager_team"},
            {"id": "T2", "objective": "Define product features", "expected_output": "Feature backlog with value statements", "recommended_team": "program_manager_team"},
            {"id": "T3", "objective": "Create engineering tasks", "expected_output": "Implementation tasks with dependencies", "recommended_team": "development_engineer_team"},
        ], indent=2),
        # Deliberately NOT a valid {"route": ...} JSON response: offline
        # mode has no real reasoning to offer over the task text, so it
        # intentionally fails RoutingAgent's parse step and lets that
        # agent's own keyword fallback pick the route instead - which is
        # actually input-sensitive, unlike a single fixed canned answer
        # would be.
        "routing": lambda: (
            "Offline mode: no live LLM available in this environment to reason about "
            "routing - falling back to keyword-based routing."
        ),
    }

    def _offline_response(self, prompt: str, response_hint: str | None) -> str:
        """Deterministic response for tests and budget-safe demos."""

        if response_hint and response_hint in self._HINTED_RESPONSES:
            return self._HINTED_RESPONSES[response_hint]()

        # No explicit hint (DirectPromptAgent/AugmentedPromptAgent/
        # RAGKnowledgePromptAgent have no fixed category) - fall back to
        # substring sniffing on the prompt itself, most-specific first.
        lowered = prompt.lower()
        if "engineering task" in lowered or "development engineer" in lowered:
            return self._HINTED_RESPONSES["engineering_tasks"]()
        if "feature" in lowered:
            return self._HINTED_RESPONSES["product_features"]()
        if "user stor" in lowered:
            return self._HINTED_RESPONSES["user_stories"]()
        if "evaluate" in lowered or "score" in lowered:
            return self._HINTED_RESPONSES["evaluation"]()
        if "plan" in lowered or "break down" in lowered:
            return """1. Analyze the product specification.\n2. Generate user stories.\n3. Define product features.\n4. Create engineering implementation tasks.\n5. Evaluate each artifact for completeness and feasibility.\n6. Compile final project backlog."""
        if "route" in lowered:
            return self._HINTED_RESPONSES["routing"]()
        return "Generated response based on the supplied project-management context."
