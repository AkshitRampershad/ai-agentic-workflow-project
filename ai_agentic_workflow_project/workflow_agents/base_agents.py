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
    ):
        super().__init__(name, llm_service)
        self.knowledge = knowledge
        self.task_instruction = task_instruction

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
        output = self.llm_service.generate(prompt)
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
        raw = self.llm_service.generate(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"score": None, "passed": False, "feedback": raw}
        return AgentResult(self.name, parsed, {"agent_type": "evaluation"})


class RoutingAgent(BaseAgent):
    """Agent that routes tasks to specialized teams."""

    def __init__(self, name: str, routes: dict[str, str], llm_service: LLMService | None = None):
        super().__init__(name, llm_service)
        self.routes = routes

    def run(self, input_data: str) -> AgentResult:
        text = input_data.lower()
        if "user stor" in text or "acceptance" in text:
            route = "product_manager_team"
        elif "feature" in text or "program" in text or "dependency" in text:
            route = "program_manager_team"
        elif "engineering" in text or "task" in text or "implement" in text:
            route = "development_engineer_team"
        else:
            route = "product_manager_team"
        return AgentResult(self.name, {"route": route, "description": self.routes.get(route, "Unknown route")}, {"agent_type": "routing"})


class ActionPlanningAgent(BaseAgent):
    """Agent that decomposes high-level goals into logical workflow tasks."""

    def run(self, input_data: str) -> AgentResult:
        prompt = f"""
Break down this technical project-management request into ordered sub-tasks.
Each sub-task should have an id, objective, expected output, and recommended team.

Request:
{input_data}
""".strip()
        raw = self.llm_service.generate(prompt)

        tasks = [
            {"id": "T1", "objective": "Generate user stories", "expected_output": "User stories with acceptance criteria", "recommended_team": "product_manager_team"},
            {"id": "T2", "objective": "Define product features", "expected_output": "Feature backlog with value statements", "recommended_team": "program_manager_team"},
            {"id": "T3", "objective": "Create engineering tasks", "expected_output": "Implementation tasks with dependencies", "recommended_team": "development_engineer_team"},
        ]
        return AgentResult(self.name, tasks, {"agent_type": "action_planning", "llm_notes": raw})
