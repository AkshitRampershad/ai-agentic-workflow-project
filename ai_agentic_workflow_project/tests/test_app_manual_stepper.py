"""
Verifies the Streamlit app's manual step-through mode: each click of
"Run this step" should fire exactly one LLM call, and 10 clicks should
complete the same workflow the "run everything at once" button runs in
one go - proving the pacing feature (added so a rate-limited free tier
can be stepped through manually) actually does one call per click, not
a batch.

Uses Streamlit's AppTest with a mocked OpenAI-compatible client, since
this sandbox has no route to a live LLM API.
"""

import json
import os
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

os.environ.setdefault("GROQ_API_KEY", "fake-test-key-for-manual-stepper-check")


def _fake_create(model, temperature, messages):
    prompt = messages[-1]["content"]
    if '"route":' in prompt:
        content = json.dumps({"route": "development_engineer_team", "reason": "mocked"})
    elif "list of objects" in prompt:
        content = json.dumps([
            {"id": "T1", "objective": "a", "expected_output": "b", "recommended_team": "product_manager_team"},
            {"id": "T2", "objective": "a", "expected_output": "b", "recommended_team": "program_manager_team"},
            {"id": "T3", "objective": "a", "expected_output": "b", "recommended_team": "development_engineer_team"},
        ])
    elif "score from 1-10" in prompt:
        content = json.dumps({"score": 9, "passed": True, "feedback": "mocked"})
    else:
        content = "mocked artifact text"
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_manual_stepper_makes_exactly_one_call_per_click():
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _fake_create

    with patch("workflow_agents.config.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = fake_client

        at = AppTest.from_file("../app.py")
        at.run()
        at.radio[0].set_value("Step through manually (one LLM call per click)").run()

        clicks = 0
        for _ in range(15):  # generous upper bound
            step_buttons = [b for b in at.button if b.label.startswith("Run this step")]
            if not step_buttons:
                break
            step_buttons[0].click().run()
            clicks += 1

        assert not at.exception
        assert clicks == 10, f"expected exactly 10 manual steps (1 plan + 3 tasks x 3), got {clicks}"
        assert fake_client.chat.completions.create.call_count == 10, (
            "each manual step click should fire exactly one LLM call"
        )

        expander_labels = " | ".join(e.label for e in at.expander)
        assert "Development Engineer team" in expander_labels
        assert "All steps complete" in "\n".join(s.value for s in at.success)
