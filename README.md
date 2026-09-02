# AI-Powered Agentic Workflow for Project Management

Pilot project: **Email Router**

This repository implements a reusable multi-agent workflow for technical project management. It converts a product specification into structured user stories, product features, and engineering tasks using agent planning, routing, knowledge augmentation, and evaluation.

## Project Structure

```text
workflow_agents/
  base_agents.py        # Seven reusable agent classes
  config.py             # Vocareum/OpenAI configuration
  llm_service.py        # Shared LLM wrapper with offline mode
agentic_workflow.py     # Main orchestration workflow
Product-Spec-Email-Router.txt
run_tests.py            # Generates terminal-style submission outputs
tests/                  # Pytest validation tests
outputs/                # Generated test and workflow outputs
```

## Implemented Agents

1. `DirectPromptAgent`
2. `AugmentedPromptAgent`
3. `KnowledgeAugmentedPromptAgent`
4. `RAGKnowledgePromptAgent`
5. `EvaluationAgent`
6. `RoutingAgent`
7. `ActionPlanningAgent`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Vocareum API Configuration

Create a `.env` file:

```bash
VOC_API_KEY=voc-your-key-here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
OPENAI_MODEL=gpt-4o-mini
```

The project is configured for Vocareum routing through:

```python
base_url="https://openai.vocareum.com/v1"
```

## Run the Main Workflow

```bash
python agentic_workflow.py
```

The workflow output is written to:

```text
outputs/email_router_project_plan.json
```

## Generate Submission Outputs

```bash
python run_tests.py
```

This writes text outputs for each agent and the full workflow into the `outputs/` folder.

## Run Automated Tests

```bash
pytest -q
```

## GitHub Push Commands

```bash
git init
git add .
git commit -m "Complete AI-powered agentic workflow project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
