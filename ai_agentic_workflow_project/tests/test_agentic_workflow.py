from agentic_workflow import run_agentic_workflow


def test_agentic_workflow_offline():
    result = run_agentic_workflow(offline_mode=True)
    assert result["project"] == "Email Router"
    assert "user_stories" in result["work_products"]
    assert "product_features" in result["work_products"]
    assert "engineering_tasks" in result["work_products"]

    # Regression test: these three used to all come back identical (the
    # evaluation agent's canned JSON), because the product spec's
    # knowledge-base text contains the word "scores", which hijacked
    # offline mode's keyword-based response picker for every artifact.
    artifacts = {
        key: result["work_products"][key]["artifact"]
        for key in ("user_stories", "product_features", "engineering_tasks")
    }
    assert len(set(artifacts.values())) == 3, f"Work products are not distinct: {artifacts}"
    assert "As a" in artifacts["user_stories"]
    assert "Feature" in artifacts["product_features"]
    assert "Implement" in artifacts["engineering_tasks"]

    # Each artifact was routed to the team its evaluator is scoped for.
    assert result["work_products"]["user_stories"]["assigned_route"] == "product_manager_team"
    assert result["work_products"]["product_features"]["assigned_route"] == "program_manager_team"
    assert result["work_products"]["engineering_tasks"]["assigned_route"] == "development_engineer_team"
