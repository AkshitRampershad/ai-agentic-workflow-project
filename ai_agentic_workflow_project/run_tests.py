"""Runs each agent in offline mode and writes terminal-style outputs for submission."""

from pathlib import Path
import json

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    AugmentedPromptAgent,
    DirectPromptAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RAGKnowledgePromptAgent,
    RoutingAgent,
)
from workflow_agents.llm_service import LLMService
from agentic_workflow import run_agentic_workflow

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
llm = LLMService(offline_mode=True)

cases = []
cases.append(DirectPromptAgent("DirectPromptAgent", llm).run("Explain technical project management."))
cases.append(AugmentedPromptAgent("AugmentedPromptAgent", "Answer as a TPM.", llm).run("What is a product backlog?"))
cases.append(KnowledgeAugmentedPromptAgent("KnowledgeAugmentedPromptAgent", "Email Router routes emails.", "Generate user stories.", llm).run("Create stories."))
cases.append(RAGKnowledgePromptAgent("RAGKnowledgePromptAgent", ["Email Router supports routing rules.", "SLA dashboards report breaches."], llm).run("routing rules"))
cases.append(EvaluationAgent("EvaluationAgent", ["clear", "testable"], llm).run("As a user, I want email routing so messages reach the correct team."))
cases.append(RoutingAgent("RoutingAgent", {"product_manager_team": "PM", "program_manager_team": "PgM", "development_engineer_team": "Eng"}, llm).run("Create engineering tasks."))
cases.append(ActionPlanningAgent("ActionPlanningAgent", llm).run("Create an Email Router delivery plan."))

for i, case in enumerate(cases, start=1):
    text = f"AGENT: {case.agent_name}\nMETADATA: {json.dumps(case.metadata, indent=2)}\nOUTPUT:\n{case.output}\n"
    file = OUT / f"agent_test_output_{i}_{case.agent_name}.txt"
    file.write_text(text, encoding="utf-8")
    print(text)

workflow_result = run_agentic_workflow(offline_mode=True)
(OUT / "agentic_workflow_terminal_output.txt").write_text(json.dumps(workflow_result, indent=2), encoding="utf-8")
print("Workflow output written to outputs/agentic_workflow_terminal_output.txt")
