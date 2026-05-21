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


def get_llm():
    return LLMService(offline_mode=True)


def test_direct_prompt_agent():
    agent = DirectPromptAgent("DirectPromptAgent", get_llm())
    result = agent.run("Explain technical project management.")
    assert result.output


def test_augmented_prompt_agent():
    agent = AugmentedPromptAgent("AugmentedPromptAgent", "Answer in concise bullets.", get_llm())
    result = agent.run("What is an epic?")
    assert result.metadata["agent_type"] == "augmented_prompt"


def test_knowledge_augmented_prompt_agent():
    agent = KnowledgeAugmentedPromptAgent(
        "KnowledgeAugmentedPromptAgent",
        "Email Router classifies and routes inbound emails.",
        "Generate user stories.",
        get_llm(),
    )
    result = agent.run("Create user stories")
    assert "As a" in result.output


def test_rag_knowledge_prompt_agent():
    docs = ["Email Router supports routing rules.", "Dashboards show SLA breaches."]
    agent = RAGKnowledgePromptAgent("RAGKnowledgePromptAgent", docs, get_llm())
    result = agent.run("routing rules")
    assert result.metadata["retrieved_count"] >= 1


def test_evaluation_agent():
    agent = EvaluationAgent("EvaluationAgent", ["clear", "testable"], get_llm())
    result = agent.run("As a user, I want routing so that emails reach the correct team.")
    assert result.output["passed"] is True


def test_routing_agent():
    routes = {
        "product_manager_team": "PM work",
        "program_manager_team": "PgM work",
        "development_engineer_team": "Engineering work",
    }
    agent = RoutingAgent("RoutingAgent", routes, get_llm())
    result = agent.run("Create engineering tasks")
    assert result.output["route"] == "development_engineer_team"


def test_action_planning_agent():
    agent = ActionPlanningAgent("ActionPlanningAgent", get_llm())
    result = agent.run("Build project plan")
    assert len(result.output) == 3
