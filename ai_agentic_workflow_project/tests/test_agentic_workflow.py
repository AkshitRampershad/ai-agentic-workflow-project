from agentic_workflow import run_agentic_workflow


def test_agentic_workflow_offline():
    result = run_agentic_workflow(offline_mode=True)
    assert result["project"] == "Email Router"
    assert "user_stories" in result["work_products"]
    assert "product_features" in result["work_products"]
    assert "engineering_tasks" in result["work_products"]
