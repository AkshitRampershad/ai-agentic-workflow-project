"""Reusable agent library for AI-powered project-management workflows."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from .llm_service import LLMService


@dataclass
class AgentResult:
    """Standard result object returned by agents."""

    agent_name: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base interface for all workflow agents."""

    def __init__(self, name: str, llm_service: LLMService | None = None):
        self.name = name
        self.llm_service = llm_service or LLMService(offline_mode=True)

    @abstractmethod
    def run(self, input_data: Any) -> AgentResult:
        """Execute the agent task."""


class DirectPromptAgent(BaseAgent):
    """Agent that sends the user prompt directly to the LLM."""

    def run(self, input_data: str) -> AgentResult:
        output = self.llm_service.generate(str(input_data))
        return AgentResult(self.name, output, {"agent_type": "direct_prompt"})


class AugmentedPromptAgent(BaseAgent):
    """Agent that augments a user prompt with explicit instruction/context."""

    def __init__(self, name: str, instruction: str, llm_service: LLMService | None = None):
        super().__init__(name, llm_service)
        self.instruction = instruction

    def run(self, input_data: str) -> AgentResult:
        prompt = f"{self.instruction}\n\nInput:\n{input_data}"
        output = self.llm_service.generate(prompt)
        return AgentResult(self.name, output, {"agent_type": "augmented_prompt"})


class KnowledgeAugmentedPromptAgent(BaseAgent):
    """Agent that injects fixed domain/product knowledge into the prompt."""

    def __init__(
        self,
        name: str,
        knowledge: str,
        task_instruction: str,
        llm_service: LLMService | None = None,
        response_hint: str | None = None,
    ):
        super().__init__(name, llm_service)
        self.knowledge = knowledge
        self.task_instruction = task_instruction
        # Lets a caller that knows exactly what artifact this instance
        # produces (e.g. "user_stories") pick the right offline canned
        # response deterministically - see LLMService.generate().
        self.response_hint = response_hint

    def run(self, input_data: str) -> AgentResult:
        prompt = f"""
You are a specialized technical project-management agent.

Knowledge Base:
{self.knowledge}

Task Instruction:
{self.task_instruction}

User/Workflow Input:
{input_data}

Return a clear, structured output suitable for technical project managers.
""".strip()
        output = self.llm_service.generate(prompt, response_hint=self.response_hint)
        return AgentResult(self.name, output, {"agent_type": "knowledge_augmented_prompt"})


class RAGKnowledgePromptAgent(BaseAgent):
    """Simple retrieval-augmented agent over an in-memory text corpus."""

    def __init__(self, name: str, documents: Iterable[str], llm_service: LLMService | None = None):
        super().__init__(name, llm_service)
        self.documents = list(documents)

    def _retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_terms = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            score = len(query_terms.intersection(set(doc.lower().split())))
            scored_docs.append((score, doc))
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k] if score > 0]

    def run(self, input_data: str) -> AgentResult:
        retrieved = self._retrieve(input_data)
        context = "\n---\n".join(retrieved) if retrieved else "No relevant documents found."
        prompt = f"Context:\n{context}\n\nQuestion/Task:\n{input_data}\n\nAnswer using the context."
        output = self.llm_service.generate(prompt)
        return AgentResult(self.name, output, {"agent_type": "rag_knowledge_prompt", "retrieved_count": len(retrieved)})


class EvaluationAgent(BaseAgent):
    """Agent that evaluates outputs against configurable criteria."""

    def __init__(self, name: str, criteria: list[str], llm_service: LLMService | None = None):
        super().__init__(name, llm_service)
        self.criteria = criteria

    def run(self, input_data: Any) -> AgentResult:
        prompt = f"""
Evaluate the following output against these criteria:
{json.dumps(self.criteria, indent=2)}

Output to evaluate:
{input_data}

Return JSON with score from 1-10, passed boolean, and feedback.
""".strip()
        raw = self.llm_service.generate(prompt, response_hint="evaluation")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"score": None, "passed": False, "feedback": raw}
        return AgentResult(self.name, parsed, {"agent_type": "evaluation"})


class RoutingAgent(BaseAgent):
    """Agent that routes tasks to specialized teams.

    Asks the LLM to pick a route from the configured team descriptions
    and return its reasoning; a task's actual content (not just
    keywords) can then influence the routing decision. Falls back to a
    fast keyword heuristic only if the LLM call fails or returns
    something that isn't one of the configured routes, so the workflow
    never breaks on a malformed response.
    """

    def __init__(self, name: str, routes: dict[str, str], llm_service: LLMService | None = None):
        super().__init__(name, llm_service)
        self.routes = routes

    def _keyword_fallback(self, text: str) -> str:
        text = text.lower()
        if "user stor" in text or "acceptance" in text:
            return "product_manager_team"
        if "feature" in text or "program" in text or "dependency" in text:
            return "program_manager_team"
        if "engineering" in text or "task" in text or "implement" in text:
            return "development_engineer_team"
        return "product_manager_team"

    def run(self, input_data: str) -> AgentResult:
        route_list = "\n".join(f'- "{name}": {desc}' for name, desc in self.routes.items())
        prompt = f"""
Choose the single best team to handle this task, from the options below.

Task:
{input_data}

Available teams:
{route_list}

Respond with JSON only: {{"route": "<one of the exact team names above>", "reason": "<one sentence>"}}
""".strip()

        used_fallback = False
        raw = self.llm_service.generate(prompt, response_hint="routing")
        try:
            parsed = json.loads(raw)
            route = parsed.get("route")
            reason = parsed.get("reason", "")
            if route not in self.routes:
                raise ValueError(f"LLM returned an unknown route: {route!r}")
        except (json.JSONDecodeError, ValueError):
            route = self._keyword_fallback(input_data)
            reason = "Keyword-based fallback (LLM did not return a valid route)."
            used_fallback = True

        return AgentResult(
            self.name,
            {"route": route, "description": self.routes.get(route, "Unknown route"), "reason": reason},
            {"agent_type": "routing", "fallback_used": used_fallback},
        )


VALID_TEAMS = {"product_manager_team", "program_manager_team", "development_engineer_team"}

# Used only if the LLM's plan can't be parsed/validated - not the normal
# path. Matches the shape a valid response must have.
_DEFAULT_TASKS = [
    {"id": "T1", "objective": "Generate user stories", "expected_output": "User stories with acceptance criteria", "recommended_team": "product_manager_team"},
    {"id": "T2", "objective": "Define product features", "expected_output": "Feature backlog with value statements", "recommended_team": "program_manager_team"},
    {"id": "T3", "objective": "Create engineering tasks", "expected_output": "Implementation tasks with dependencies", "recommended_team": "development_engineer_team"},
]

_REQUIRED_TASK_KEYS = {"id", "objective", "expected_output", "recommended_team"}


class ActionPlanningAgent(BaseAgent):
    """Agent that decomposes high-level goals into logical workflow tasks.

    Uses the LLM's own decomposition of the request - the number and
    content of sub-tasks can vary with the input - rather than always
    returning the same fixed 3-task list. Falls back to that fixed list
    only if the LLM's response can't be parsed into valid tasks, so a
    malformed response never breaks the workflow.
    """

    def run(self, input_data: str) -> AgentResult:
        prompt = f"""
Break down this technical project-management request into ordered sub-tasks.

Request:
{input_data}

Respond with JSON only: a list of objects, each with exactly these keys:
- "id": short task id (e.g. "T1")
- "objective": what this sub-task accomplishes
- "expected_output": what artifact it produces
- "recommended_team": one of "product_manager_team", "program_manager_team", "development_engineer_team"
""".strip()
        raw = self.llm_service.generate(prompt, response_hint="action_planning")

        tasks, used_fallback = self._parse_tasks(raw)
        return AgentResult(
            self.name,
            tasks,
            {"agent_type": "action_planning", "llm_notes": raw, "fallback_used": used_fallback},
        )

    @staticmethod
    def _parse_tasks(raw: str) -> tuple[list[dict], bool]:
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("Expected a non-empty JSON list of tasks.")
            for task in parsed:
                if not isinstance(task, dict) or not _REQUIRED_TASK_KEYS.issubset(task):
                    raise ValueError(f"Task missing required keys: {task!r}")
                if task["recommended_team"] not in VALID_TEAMS:
                    raise ValueError(f"Unknown recommended_team: {task['recommended_team']!r}")
            return parsed, False
        except (json.JSONDecodeError, ValueError):
            return list(_DEFAULT_TASKS), True
