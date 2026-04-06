---
title: Email Triage
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 📧 Email Triage — OpenEnv Environment

A real-world OpenEnv-compliant reinforcement learning environment where AI agents learn to triage emails like a professional executive assistant. 

## Motivation
Email overload is one of the most relatable and pervasive productivity challenges in the modern workplace. 
This environment models the real-world administrative task of triaging an inbox (distinguishing spam, identifying critical production incidents, sorting newsletters, and archiving notifications) rather than a toy puzzle or game. Agents that perform well in this environment demonstrate cognitive capabilities necessary for genuine autonomous executive assistants.

## Action Space
In each step, the agent outputs a JSON payload with its classification.
- **`label`** *(string)*: `spam`, `inbox`, `urgent`, `archive`, or `delete`
- **`priority`** *(string)*: `high`, `medium`, or `low`
- **`category`** *(string)*: `spam`, `work`, `personal`, `newsletter`, `notification`, or `social`

## Observation Space
The environment returns detailed JSON metadata representing the current email and episode state:
- `email_id` *(string)*: Unique identifier for the email
- `subject` *(string)*: Email subject line
- `sender` *(string)*: Sender email address
- `body` *(string)*: Full email body text
- `timestamp` *(string)*: When the email arrived
- `step` *(int)*: Current step number
- `total_emails` *(int)*: Total emails to process
- `emails_remaining` *(int)*: Count of remaining emails
- `reward` *(float)*: Normalized reward for the last step [0.0, 1.0].
- `cumulative_reward` *(float)*: Total accumulated reward
- `done` *(boolean)*: True if the episode is finished

## Tasks & Expected Difficulty
1. **Spam Detection (easy)**: Binary classification of emails as either `spam` or `inbox`. Scored purely on labeling accuracy.
2. **Priority Triage (medium)**: Assign the correct `label` AND `priority` to each email. Severe reward penalties are applied if an urgent email is given low priority.
3. **Full Email Triage (hard)**: Complete triage. Predict `label`, `priority`, AND `category`. Deep semantic nuance is required to distinguish notifications, automated emails, newsletters, and critical situations.

## Setup and Usage Instructions

**1. Install Dependencies**
```bash
pip install openenv-core openai requests uvicorn fastapi pydantic
# Run validate if desired
openenv validate
```

**2. Run the environment server locally (FastAPI)**
```bash
uvicorn server.app:app --port 7860
```
*Alternatively, simply deploy directly via the provided Dockerfile or access the Hugging Face Space URL.*

**3. Run Inference Pipeline**
To evaluate an LLM proxy agent against the hosted or local environment:
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
export ENV_BASE_URL="http://localhost:7860" # Local server endpoint

python inference.py
```

## Baseline Scores
Using a standard keyword heuristics approach (Rule-based Baseline Agent):
- **Task 1 (Spam Detection)**: 1.000 
- **Task 2 (Priority Triage)**: 0.925
- **Task 3 (Full Email Triage)**: 0.872

Using an LLM (e.g., `gpt-4o-mini`):
- Expect scores well above `0.95`, demonstrating near-perfect human-level sorting capability.

## Endpoints
- `GET /health` — liveness check
- `POST /reset` — start episode
- `POST /step` — submit action
- `GET /state` — episode metadata
- `GET /tasks` — list all tasks
- `POST /grader` — grade completed episode
- `GET /baseline` — run baseline agent
- `GET /docs` — interactive API docs
