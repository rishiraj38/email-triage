# 📧 Email Triage — OpenEnv Environment

> A real-world RL environment where AI agents learn to triage emails like a professional executive assistant.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Why Email Triage?

The average knowledge worker spends **2.6 hours per day on email**. An AI agent that can triage an inbox — filtering spam, flagging urgent issues, and correctly categorizing everything else — has immediate, measurable real-world value.

This environment tests whether an agent can:
- Distinguish **spam and phishing** from legitimate email
- Identify **genuinely urgent** situations (server outages, client crises, CEO mandates)
- Correctly **categorize** emails by type and **prioritize** them appropriately
- Avoid costly mistakes: **missing urgent emails** or **false-positive spam** on important messages

---

## Environment Description

Each episode consists of **20 diverse, realistic emails** shuffled into a random order. The agent reads them one at a time and must classify each one. Emails span:

| Type | Count | Examples |
|------|-------|---------|
| Spam / Phishing | 4 | Lottery scam, Nigerian prince, fake PayPal alert |
| Urgent Work | 5 | Production DB down, CEO mandatory meeting, security breach |
| Normal Work | 4 | Meeting notes, PR review, colleague question |
| Newsletters | 3 | Tech digest, flash sale, weekly product update |
| Notifications | 3 | GitHub PR approved, bank statement, HR reminder |
| Personal | 1 | Family email |

---

## Action Space

All three fields are present in every action. Which fields are **scored** depends on the task.

| Field | Type | Valid Values |
|-------|------|-------------|
| `label` | string | `spam` `inbox` `urgent` `archive` `delete` |
| `priority` | string | `high` `medium` `low` |
| `category` | string | `spam` `work` `personal` `newsletter` `notification` `social` |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | string | Unique email identifier |
| `subject` | string | Email subject line |
| `sender` | string | Sender address |
| `body` | string | Full email body |
| `timestamp` | string | When the email arrived |
| `step` | int | Current step number (0-indexed) |
| `total_emails` | int | Total emails in the episode |
| `emails_remaining` | int | Emails still to classify |
| `reward` | float | Reward earned on the last action |
| `cumulative_reward` | float | Total reward so far |
| `done` | bool | True when all emails are classified |

---

## Tasks

### Task 1 — Spam Detection `[easy]`

**Objective:** Classify each email as `spam` or `inbox`.

**Scoring (per email):**
```
Correct:   +1.0
Incorrect:  0.0
```

**Why it's easy:** Spam emails in the dataset have clear signals (suspicious senders, lottery language, phishing URLs).

**Expected baseline score:** ~0.85

---

### Task 2 — Priority Triage `[medium]`

**Objective:** Assign the correct `label` (spam / inbox / urgent / archive) **and** `priority` (high / medium / low).

**Scoring (per email, max 1.0):**
```
label correct (exact):    0.50
label close (inbox/urgent confusion): 0.25
priority correct (exact): 0.50
priority close (adjacent level):      0.25

Penalty: −0.30 if an URGENT email is assigned LOW priority
```

**Why it's medium:** Requires understanding context and stakes, not just pattern matching. Missing an urgent email is costly.

**Expected baseline score:** ~0.62

---

### Task 3 — Full Email Triage `[hard]`

**Objective:** Correctly assign `label`, `priority`, **and** `category` for every email.

**Scoring (per email, max 1.0):**
```
label correct:    0.35
priority correct: 0.35
category correct: 0.30

Penalty: −0.30 if URGENT email not given HIGH priority
Penalty: −0.20 if a LEGITIMATE email is marked SPAM (false positive)
```

**Why it's hard:** Requires distinguishing notification vs newsletter vs work emails, handling edge cases (is a GitHub PR notification "work" or "notification"?), and avoiding false spam on legitimate-but-suspicious-sounding emails.

**Expected baseline score:** ~0.44

---

## Reward Function Design

The reward function provides **dense per-step signal** — not just a binary win/loss at the end of the episode.

Key design decisions:

1. **Partial credit:** Adjacent priority levels (e.g., high vs medium) earn 0.5× instead of 0. This gives meaningful gradient signal even when the answer isn't perfect.

2. **Asymmetric penalties:** Missing an urgent email is penalized more severely than a wrong category, reflecting real-world stakes (missing a server outage is catastrophic; misclassifying a newsletter is minor).

3. **False positive penalty:** Marking legitimate email as spam is explicitly penalized, discouraging agents that learn to call everything spam.

4. **Normalized to [0, 1]:** Every per-email reward is clamped to [0.0, 1.0]. Episode score = mean per-email reward.

---

## Setup & Usage

### Option 1: Docker (recommended)

```bash
# Clone the repo
git clone https://huggingface.co/spaces/your-username/email-triage
cd email-triage

# Build
docker build -t email-triage:latest -f server/Dockerfile .

# Run
docker run -d -p 7860:7860 email-triage:latest

# Verify
curl http://localhost:7860/health
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run the LLM baseline

```bash
export OPENAI_API_KEY=sk-...
python baseline.py
```

### Use the Python client

```python
from client import EmailTriageClient

with EmailTriageClient("http://localhost:7860") as client:
    # Task 1: spam detection
    obs = client.reset(task_id=1)

    while not obs["done"]:
        # Your policy here
        if "lottery" in obs["subject"].lower():
            action = {"label": "spam"}
        else:
            action = {"label": "inbox"}

        result = client.step(**action)
        obs = result["observation"]
        print(f"  reward={result['reward']:.3f} | {result['info']['feedback']}")

    grade = client.grader()
    print(f"\nFinal score: {grade['score']:.4f}")
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server liveness check |
| `/reset` | POST | Start new episode. Body: `{"task_id": 1}` |
| `/step` | POST | Submit action. Body: `{"label":"...","priority":"...","category":"..."}` |
| `/state` | GET | Current episode metadata |
| `/tasks` | GET | All tasks and action schemas |
| `/grader` | POST | Grade completed episode (call after done==true) |
| `/baseline` | GET | Run built-in rule-based baseline, returns all 3 scores |
| `/docs` | GET | Interactive OpenAPI documentation |

---

## Baseline Scores

| Task | Agent | Score |
|------|-------|-------|
| Spam Detection (easy) | Rule-based | ~0.80 |
| Spam Detection (easy) | gpt-4o-mini | ~0.90 |
| Priority Triage (medium) | Rule-based | ~0.55 |
| Priority Triage (medium) | gpt-4o-mini | ~0.62 |
| Full Email Triage (hard) | Rule-based | ~0.38 |
| Full Email Triage (hard) | gpt-4o-mini | ~0.44 |

A well-trained RL agent that learns from reward signals should significantly exceed these baselines — particularly on the hard task.

---

## Project Structure

```
email_triage/
├── emails.py           ← 20 emails with ground truth labels
├── models.py           ← Pydantic models (Action, Observation, State, Reward)
├── client.py           ← Python HTTP client
├── baseline.py         ← LLM baseline script (OpenAI API)
├── openenv.yaml        ← OpenEnv spec manifest
├── requirements.txt
├── README.md
└── server/
    ├── __init__.py
    ├── environment.py  ← Core RL environment logic
    ├── grader.py       ← Episode scoring
    ├── app.py          ← FastAPI server (all endpoints)
    └── Dockerfile
```

---

## Deploy to Hugging Face Spaces

```bash
# Create a new Space on huggingface.co (Docker type)
# Then push:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/your-username/email-triage
git push origin main
```

The Space will automatically build the Docker image and start the server.
Your environment will be live at: `https://your-username-email-triage.hf.space`

---

## License

MIT
