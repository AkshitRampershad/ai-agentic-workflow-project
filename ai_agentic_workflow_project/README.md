# AI-Powered Agentic Workflow for Project Management

Pilot project: **Email Router**

A reusable multi-agent workflow for technical project management. It takes a raw product specification and turns it into structured user stories, product features, and engineering tasks — using a planning agent to break the goal into sub-tasks, a routing agent to send each sub-task to the right specialist agent, and evaluation agents to score every artifact the workflow produces.

## How It Works
1. **`ActionPlanningAgent`** breaks the high-level goal ("turn this spec into a project plan") into ordered sub-tasks: generate user stories, define features, create engineering tasks.
2. **`RoutingAgent`** inspects each sub-task and routes it to the product manager, program manager, or development engineer "team".
3. **`KnowledgeAugmentedPromptAgent`** generates the actual artifact for that team, using the product spec as fixed domain knowledge injected into the prompt.
4. An **`EvaluationAgent`** scores each artifact against team-specific criteria (e.g. "uses Agile user-story format," "features map to product goals," "tasks are technically actionable") and returns a score, pass/fail, and feedback.
5. The full plan — action plan, generated artifacts, and evaluation scores — is written to `outputs/email_router_project_plan.json`.

## Agent Library
The workflow is built on seven reusable agent classes in `workflow_agents/base_agents.py`, all sharing one `LLMService` wrapper:

| Agent | Role |
| --- | --- |
| `DirectPromptAgent` | Sends a prompt straight to the LLM, no augmentation |
| `AugmentedPromptAgent` | Adds an explicit instruction/persona around the prompt |
| `KnowledgeAugmentedPromptAgent` | Injects fixed domain knowledge (the product spec) into the prompt |
| `RAGKnowledgePromptAgent` | Retrieves the most relevant documents from an in-memory corpus before answering |
| `EvaluationAgent` | Scores an artifact against a configurable list of criteria |
| `RoutingAgent` | Routes a task to the appropriate specialist team based on keywords |
| `ActionPlanningAgent` | Decomposes a high-level goal into ordered, assignable sub-tasks |

The main pipeline (`agentic_workflow.py`) wires up `ActionPlanningAgent`, `RoutingAgent`, `KnowledgeAugmentedPromptAgent`, and `EvaluationAgent`. `DirectPromptAgent`, `AugmentedPromptAgent`, and `RAGKnowledgePromptAgent` are demonstrated independently via `run_tests.py` and the test suite.

An **offline mode** in `LLMService` returns deterministic, hand-written responses for each agent type — useful for running tests and demos without hitting the LLM API or incurring cost.

## Project Structure
```text
ai_agentic_workflow_project/
  workflow_agents/
    base_agents.py         # The seven agent classes
    config.py               # LLM client / provider configuration
    llm_service.py           # Shared LLM wrapper with offline mode
  agentic_workflow.py        # Main orchestration pipeline
  app.py                     # Streamlit demo (workflow run + agent playground)
  Product-Spec-Email-Router.txt   # Pilot product spec used as input
  run_tests.py                # Exercises all seven agents, writes outputs/ for review
  tests/                      # Pytest test suite
  outputs/                    # Generated workflow and test outputs
  requirements.txt
```

## Setup
```bash
cd ai_agentic_workflow_project
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Configuration
This project defaults to [Groq](https://console.groq.com/keys) (OpenAI-compatible chat completions, free tier available) - the original Vocareum gateway this was built for is only reachable with an active Udacity course subscription, which most people running this afterward won't have. Create a `.env` file inside `ai_agentic_workflow_project/` (see `.env.example`):
```bash
GROQ_API_KEY=gsk-your-groq-key-here
```
An OpenAI key or a Vocareum key both work too (see `.env.example` for the alternative env vars) - whichever of `GROQ_API_KEY`, `VOC_API_KEY`, `OPENAI_API_KEY` is set first, in that order, is used. Leave all three unset to run everything in offline mode using deterministic mock responses (no API calls, no cost).

### What makes this a real agentic workflow, not just prompt templates
Two agents used to only look LLM-driven:
- **`RoutingAgent`** used to route purely by keyword-matching the task text - it never called the LLM at all. It now asks the LLM to choose a team from the configured route descriptions and returns its reasoning; a keyword heuristic is kept only as a fallback if the LLM's response can't be parsed into one of the configured routes.
- **`ActionPlanningAgent`** used to call the LLM, then discard its response and return the same hardcoded 3-task list every time. It now parses and returns the LLM's actual task breakdown (which can vary in count and content with the input), falling back to a fixed 3-task list only if that response is malformed.

Both fallback paths are real safety nets, not silent failures: each result's `metadata["fallback_used"]` says whether the LLM's decision was used or the deterministic fallback kicked in, so a malformed response degrades gracefully instead of crashing the workflow.

Along the way, a real bug in offline mode was also fixed: `LLMService`'s canned-response picker used to substring-match the *entire* prompt, including the injected product-spec knowledge base - which contains the word "scores" (`"...confidence scores..."`). That word matched the evaluation-response branch *before* any of the artifact-specific branches got a chance to match, so every `KnowledgeAugmentedPromptAgent` (user stories, product features, engineering tasks) silently returned the same canned evaluation JSON instead of its real artifact. `generate()` now takes an explicit `response_hint` so each agent picks its own canned response deterministically instead of relying on prompt-text sniffing.

## Usage

Run the main workflow (reads `Product-Spec-Email-Router.txt`, writes the full project plan to `outputs/email_router_project_plan.json`):
```bash
python agentic_workflow.py
```

Exercise all seven agents individually and write per-agent outputs to `outputs/` for review:
```bash
python run_tests.py
```

Run the automated test suite:
```bash
pytest -q
```

### Streamlit demo
```bash
streamlit run app.py
```
Two views: **Run the Workflow** runs the full pipeline on the Email Router spec (or one you paste in) and shows the action plan, each routing decision (with its LLM-given reason and whether a fallback kicked in), the generated artifacts, and their evaluation scores. **Agent Playground** runs any of the seven agent classes individually with your own input.

A full run makes ~10 sequential LLM calls (1 planning call, then route + generate + evaluate for each of the plan's tasks), which can trip a free-tier rate limit. Transient errors (rate limits, timeouts, 5xx) are retried automatically with backoff, honoring the provider's `Retry-After` hint. If you'd rather pace the calls yourself, switch **Run the Workflow**'s mode to *"Step through manually"* - each click of "Run this step" fires exactly one LLM call and shows its result immediately, so you control the timing between requests.

Add your Groq key as a Streamlit secret before running (never commit this file - it's gitignored):
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your real key
```
Without a key configured, the app runs in offline mode (deterministic mock responses, clearly labeled in the UI) rather than failing.

### Deploying to Streamlit Community Cloud
1. Push this repo to GitHub (already done if you're reading this from the deployed app's source).
2. At [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo, with **`ai_agentic_workflow_project/app.py`** as the main file path (note the subdirectory - the Streamlit app isn't at the repo root).
3. In the app's **Settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```
4. Deploy. The workflow runs in live LLM mode immediately; omit the secret to run in offline mode instead.
