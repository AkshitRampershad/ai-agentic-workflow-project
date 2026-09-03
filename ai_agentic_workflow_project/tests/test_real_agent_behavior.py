"""
Verifies the agents that are supposed to be LLM-driven actually are -
i.e. their decisions come from the LLM's response, not a hardcoded
value or a keyword heuristic alone - and that the fallback paths kick
in correctly when the LLM's response can't be used. Also regression-
tests the offline-mode keyword-collision bug (a knowledge base
containing the word "scores" used to hijack every artifact response).

Since this sandbox has no route to a live LLM API, "live mode" here
means mocking LLMService.generate (or, for one end-to-end check, the
underlying OpenAI client) rather than actually calling out.
"""

import json
from unittest.mock import MagicMock, patch

from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, RoutingAgent
from workflow_agents.config import LLMConfig
from workflow_agents.llm_service import LLMService


ROUTES = {
    "product_manager_team": "Creates user stories and acceptance criteria.",
    "program_manager_team": "Defines features, milestones, dependencies, and risks.",
    "development_engineer_team": "Creates detailed engineering implementation tasks.",
}


def test_routing_agent_uses_llm_decision_not_just_keywords():
    """A task with no routing keywords at all should still route
    correctly when the LLM says so - proving the route comes from the
    LLM's response, not string matching on the input.
    """
    llm = LLMService(offline_mode=True)
    llm.generate = MagicMock(return_value=json.dumps({
        "route": "program_manager_team",
        "reason": "This concerns delivery milestones, which is program management's remit.",
    }))

    agent = RoutingAgent("RoutingAgent", ROUTES, llm)
    result = agent.run("Figure out what quarter this ships in and what it depends on.")

    assert result.output["route"] == "program_manager_team"
    assert result.metadata["fallback_used"] is False
    llm.generate.assert_called_once()
    assert llm.generate.call_args.kwargs.get("response_hint") == "routing"


def test_routing_agent_falls_back_on_invalid_llm_response():
    llm = LLMService(offline_mode=True)
    llm.generate = MagicMock(return_value="not valid json at all")

    agent = RoutingAgent("RoutingAgent", ROUTES, llm)
    result = agent.run("Create engineering tasks for the connector.")

    assert result.output["route"] == "development_engineer_team"  # keyword fallback
    assert result.metadata["fallback_used"] is True


def test_routing_agent_falls_back_on_unknown_route_name():
    llm = LLMService(offline_mode=True)
    llm.generate = MagicMock(return_value=json.dumps({"route": "marketing_team", "reason": "n/a"}))

    agent = RoutingAgent("RoutingAgent", ROUTES, llm)
    result = agent.run("Generate user stories for the pilot.")

    assert result.output["route"] == "product_manager_team"
    assert result.metadata["fallback_used"] is True


def test_action_planning_agent_uses_llms_actual_plan():
    """The LLM's plan has a DIFFERENT shape (5 tasks, not the old
    hardcoded 3) - this only passes if the agent is actually parsing
    and returning the LLM's output rather than a fixed list.
    """
    llm_tasks = [
        {"id": "T1", "objective": "Draft user stories", "expected_output": "Stories", "recommended_team": "product_manager_team"},
        {"id": "T2", "objective": "Draft acceptance criteria", "expected_output": "Criteria", "recommended_team": "product_manager_team"},
        {"id": "T3", "objective": "Define features", "expected_output": "Feature list", "recommended_team": "program_manager_team"},
        {"id": "T4", "objective": "Scope engineering work", "expected_output": "Task list", "recommended_team": "development_engineer_team"},
        {"id": "T5", "objective": "Define rollout plan", "expected_output": "Rollout plan", "recommended_team": "program_manager_team"},
    ]
    llm = LLMService(offline_mode=True)
    llm.generate = MagicMock(return_value=json.dumps(llm_tasks))

    agent = ActionPlanningAgent("ActionPlanningAgent", llm)
    result = agent.run("Plan the Email Router delivery in more granular detail.")

    assert result.output == llm_tasks
    assert len(result.output) == 5
    assert result.metadata["fallback_used"] is False


def test_action_planning_agent_falls_back_on_malformed_plan():
    llm = LLMService(offline_mode=True)
    llm.generate = MagicMock(return_value="I think you should do three things but not as JSON.")

    agent = ActionPlanningAgent("ActionPlanningAgent", llm)
    result = agent.run("Plan the Email Router delivery.")

    assert len(result.output) == 3
    assert result.metadata["fallback_used"] is True


def test_action_planning_agent_falls_back_on_invalid_team_name():
    llm = LLMService(offline_mode=True)
    bad_tasks = [{"id": "T1", "objective": "x", "expected_output": "y", "recommended_team": "marketing_team"}]
    llm.generate = MagicMock(return_value=json.dumps(bad_tasks))

    agent = ActionPlanningAgent("ActionPlanningAgent", llm)
    result = agent.run("Plan it.")

    assert result.metadata["fallback_used"] is True
    assert len(result.output) == 3


def test_offline_mode_response_hint_avoids_keyword_collision():
    """Regression test for the bug where a knowledge base containing
    the word "scores" (e.g. "confidence scores" in the product spec)
    made every KnowledgeAugmentedPromptAgent silently return the
    evaluation-JSON canned response instead of its real artifact.
    """
    llm = LLMService(offline_mode=True)
    knowledge_with_scores = "The system stores routing decisions, confidence scores, and timestamps in an audit log."

    pm_agent = KnowledgeAugmentedPromptAgent(
        "ProductManagerAgent", knowledge_with_scores, "Generate user stories.", llm, response_hint="user_stories"
    )
    pgm_agent = KnowledgeAugmentedPromptAgent(
        "ProgramManagerAgent", knowledge_with_scores, "Define features.", llm, response_hint="product_features"
    )
    eng_agent = KnowledgeAugmentedPromptAgent(
        "DevelopmentEngineerAgent", knowledge_with_scores, "Create engineering tasks.", llm, response_hint="engineering_tasks"
    )

    pm_output = pm_agent.run("Create stories.").output
    pgm_output = pgm_agent.run("Create features.").output
    eng_output = eng_agent.run("Create tasks.").output

    assert "As a" in pm_output
    assert "Feature" in pgm_output
    assert "Implement" in eng_output
    # None of the three should be the evaluation-agent's canned JSON.
    assert "\"passed\"" not in pm_output
    assert "\"passed\"" not in pgm_output
    assert "\"passed\"" not in eng_output
    # And they must be distinct from each other - the original bug made
    # all three identical.
    assert len({pm_output, pgm_output, eng_output}) == 3


def test_live_mode_generate_calls_openai_compatible_client():
    """End-to-end check of the non-offline code path: with a fake
    OpenAI-compatible client injected, LLMService.generate should call
    it with the configured model and return its content - proving the
    live-mode wiring itself is correct, independent of network access.
    """
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Live LLM response."))]
    )

    with patch("workflow_agents.llm_service.build_client", return_value=fake_client):
        llm = LLMService(config=LLMConfig(api_key="fake-key", base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b"))
        assert llm.offline_mode is False

        output = llm.generate("Say hello.", response_hint="user_stories")

    assert output == "Live LLM response."
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-oss-120b"
    assert call_kwargs["messages"][-1] == {"role": "user", "content": "Say hello."}
