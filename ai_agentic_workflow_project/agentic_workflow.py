"""General-purpose agentic workflow for technical project management.

Pilot use case: Email Router product specification.
"""

from __future__ import annotations

import json
from pathlib import Path

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)
from workflow_agents.llm_service import LLMService


ROOT = Path(__file__).parent
SPEC_PATH = ROOT / "Product-Spec-Email-Router.txt"
OUTPUT_PATH = ROOT / "outputs" / "email_router_project_plan.json"


def load_product_spec(path: Path = SPEC_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing product spec: {path}")
    return path.read_text(encoding="utf-8")


def build_workflow(offline_mode: bool = False):
    llm = LLMService(offline_mode=offline_mode)
    product_spec = load_product_spec()

    routes = {
        "product_manager_team": "Creates user stories and acceptance criteria.",
        "program_manager_team": "Defines features, milestones, dependencies, and risks.",
        "development_engineer_team": "Creates detailed engineering implementation tasks.",
    }

    agents = {
        "planner": ActionPlanningAgent("ActionPlanningAgent", llm),
        "router": RoutingAgent("RoutingAgent", routes, llm),
        "pm": KnowledgeAugmentedPromptAgent(
            "ProductManagerAgent",
            product_spec,
            "Generate Agile user stories with acceptance criteria for the Email Router pilot.",
            llm,
        ),
        "pgm": KnowledgeAugmentedPromptAgent(
            "ProgramManagerAgent",
            product_spec,
            "Define product features, delivery dependencies, milestones, and risks.",
            llm,
        ),
        "eng": KnowledgeAugmentedPromptAgent(
            "DevelopmentEngineerAgent",
            product_spec,
            "Create detailed engineering tasks, technical dependencies, APIs, data needs, and validation steps.",
            llm,
        ),
        "story_eval": EvaluationAgent(
            "UserStoryEvaluationAgent",
            [
                "Uses Agile user-story format",
                "Includes clear acceptance criteria",
                "Aligns with product specification",
                "Is testable by QA/TPM teams",
            ],
            llm,
        ),
        "feature_eval": EvaluationAgent(
            "FeatureEvaluationAgent",
            [
                "Features map to product goals",
                "Includes business value",
                "Identifies dependencies or risks",
                "Is clear enough for backlog planning",
            ],
            llm,
        ),
        "task_eval": EvaluationAgent(
            "EngineeringTaskEvaluationAgent",
            [
                "Tasks are technically actionable",
                "Includes implementation details",
                "Covers integrations and audit requirements",
                "Supports non-functional requirements",
            ],
            llm,
        ),
    }
    return agents


def run_agentic_workflow(offline_mode: bool = False) -> dict:
    agents = build_workflow(offline_mode=offline_mode)

    high_level_prompt = (
        "Transform the Email Router product specification into a structured technical project plan "
        "containing user stories, product features, and detailed engineering tasks."
    )

    plan = agents["planner"].run(high_level_prompt)
    final_output = {
        "project": "Email Router",
        "workflow_goal": high_level_prompt,
        "action_plan": plan.output,
        "work_products": {},
        "evaluation_summary": {},
    }

    for task in plan.output:
        routing_result = agents["router"].run(task["objective"])
        route = routing_result.output["route"]

        if route == "product_manager_team":
            artifact = agents["pm"].run(task["objective"])
            evaluation = agents["story_eval"].run(artifact.output)
            key = "user_stories"
        elif route == "program_manager_team":
            artifact = agents["pgm"].run(task["objective"])
            evaluation = agents["feature_eval"].run(artifact.output)
            key = "product_features"
        else:
            artifact = agents["eng"].run(task["objective"])
            evaluation = agents["task_eval"].run(artifact.output)
            key = "engineering_tasks"

        final_output["work_products"][key] = {
            "assigned_route": route,
            "artifact": artifact.output,
        }
        final_output["evaluation_summary"][key] = evaluation.output

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    return final_output


if __name__ == "__main__":
    result = run_agentic_workflow(offline_mode=False)
    print(json.dumps(result, indent=2))
