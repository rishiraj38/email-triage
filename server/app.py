"""
Email Triage OpenEnv — FastAPI Server

Implements the full OpenEnv HTTP spec:
    POST /reset     → start new episode
    POST /step      → submit action for current email
    GET  /state     → current episode metadata
    GET  /health    → health check
    GET  /tasks     → list tasks and action schemas   (required by hackathon)
    POST /grader    → grade completed episode          (required by hackathon)
    GET  /baseline  → run rule-based baseline agent   (required by hackathon)
    GET  /docs      → auto-generated OpenAPI docs
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Make sure root directory is on sys.path so we can import models/emails ──
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailAction, ResetRequest
from server.environment import EmailTriageEnvironment
from server.grader import grade_episode

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Triage — OpenEnv Environment",
    description=(
        "A real-world RL environment where agents learn to triage emails. "
        "3 tasks from spam detection (easy) to full triage (hard). "
        "Built for the OpenEnv hackathon by Meta × Hugging Face."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (thread-safe enough for demo/judging)
_env = EmailTriageEnvironment()


# ─────────────────────────────────────────────────────────────────────────────
# OpenEnv core endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — must return 200 for hackathon validation."""
    return {"status": "healthy", "environment": "email-triage", "version": "1.0.0"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    Returns the first email observation.

    Body (optional): {"task_id": 1}
      task_id=1 → Spam Detection (easy)
      task_id=2 → Priority Triage (medium)
      task_id=3 → Full Email Triage (hard)
    """
    try:
        obs = _env.reset(task_id=request.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/step")
def step(action: EmailAction):
    """
    Submit a classification action for the current email.

    Body:
      {
        "label":    "spam | inbox | urgent | archive | delete",
        "priority": "high | medium | low",
        "category": "spam | work | personal | newsletter | notification | social"
      }

    Returns StepResult: (observation, reward, done, info)
    """
    if _env.done and _env.episode_id is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    if _env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is complete. Call POST /reset to start a new episode.",
        )
    try:
        result = _env.step(action)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """
    Return current episode metadata (no email content).
    Includes episode_id, task_id, step_count, score, done, etc.
    """
    return _env.state().model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Hackathon-required extra endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def tasks():
    """
    List all available tasks and the full action schema for each.
    Required by hackathon spec.
    """
    return {
        "tasks": [
            {
                "id": 1,
                "name": "Spam Detection",
                "description": (
                    "Classify each of the 20 emails as spam or legitimate (inbox). "
                    "Focus on identifying unsolicited, malicious, or phishing emails."
                ),
                "difficulty": "easy",
                "num_emails": 20,
                "scoring": "1.0 per correct spam/not-spam call. Score = avg across episode.",
                "action_schema": {
                    "label": {
                        "type": "string",
                        "required": True,
                        "values": ["spam", "inbox"],
                        "description": "spam = unwanted/malicious | inbox = legitimate",
                    },
                    "priority": {
                        "type": "string",
                        "required": False,
                        "note": "Not scored in Task 1 — any value accepted",
                    },
                    "category": {
                        "type": "string",
                        "required": False,
                        "note": "Not scored in Task 1 — any value accepted",
                    },
                },
            },
            {
                "id": 2,
                "name": "Priority Triage",
                "description": (
                    "Assign the correct label AND priority to each email. "
                    "Critical: failing to mark urgent emails as high-priority incurs a penalty."
                ),
                "difficulty": "medium",
                "num_emails": 20,
                "scoring": (
                    "0.5 × label_score + 0.5 × priority_score per email. "
                    "Penalty of −0.30 if an urgent email gets low priority."
                ),
                "action_schema": {
                    "label": {
                        "type": "string",
                        "required": True,
                        "values": ["spam", "inbox", "urgent", "archive"],
                        "description": (
                            "spam=junk | inbox=needs attention | "
                            "urgent=act immediately | archive=save but no action"
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "required": True,
                        "values": ["high", "medium", "low"],
                        "description": "high=act today | medium=act this week | low=whenever",
                    },
                    "category": {
                        "type": "string",
                        "required": False,
                        "note": "Not scored in Task 2",
                    },
                },
            },
            {
                "id": 3,
                "name": "Full Email Triage",
                "description": (
                    "Complete triage: assign label, priority, AND category correctly. "
                    "Penalized for missed urgent emails and false-positive spam classifications."
                ),
                "difficulty": "hard",
                "num_emails": 20,
                "scoring": (
                    "0.35 × label + 0.35 × priority + 0.30 × category per email. "
                    "Penalty: −0.30 for missed urgent, −0.20 for false spam."
                ),
                "action_schema": {
                    "label": {
                        "type": "string",
                        "required": True,
                        "values": ["spam", "inbox", "urgent", "archive", "delete"],
                    },
                    "priority": {
                        "type": "string",
                        "required": True,
                        "values": ["high", "medium", "low"],
                    },
                    "category": {
                        "type": "string",
                        "required": True,
                        "values": [
                            "spam", "work", "personal",
                            "newsletter", "notification", "social",
                        ],
                    },
                },
            },
        ]
    }


@app.post("/grader")
def grader():
    """
    Grade the most recently completed episode.
    Must be called AFTER the episode is done (obs.done == True).
    Required by hackathon spec.

    Returns:
      score          : float [0.0, 1.0] — primary metric
      task_id        : int
      task_name      : str
      total_emails   : int
      details        : accuracy breakdown per dimension
      per_email      : per-email scores and feedback
    """
    if not _env.done:
        raise HTTPException(
            status_code=400,
            detail=(
                "Episode is not yet complete. "
                f"Emails remaining: {len(_env.emails) - _env.current_index}. "
                "Keep calling POST /step until obs.done == true, then call /grader."
            ),
        )
    if not _env.results:
        raise HTTPException(
            status_code=400,
            detail="No results to grade. Run a full episode first (POST /reset then POST /step until done).",
        )

    return grade_episode(_env.results, _env.task_id)


@app.get("/baseline")
def baseline():
    """
    Run a built-in rule-based baseline agent against all 3 tasks and return scores.

    The rule-based agent uses keyword heuristics — no LLM required.
    For the LLM baseline (gpt-4o-mini), run: python baseline.py

    Required by hackathon spec.
    """
    from server.environment import EmailTriageEnvironment
    from server.grader import grade_episode
    from models import EmailAction

    # Keywords used by the rule-based baseline agent
    SPAM_SIGNALS     = ["lottery", "won $", "million", "bank details", "processing fee",
                        "nigeria", "transfer", "verify immediately", "paypa1", "quick-cash",
                        "winners.tk", "biz", ".ng", "act now", "limited spots"]
    URGENT_SIGNALS   = ["critical", "p0", "production", "down", "urgent", "mandatory",
                        "security breach", "incident", "immediately", "emergency", "asap"]
    ARCHIVE_SIGNALS  = ["unsubscribe", "newsletter", "digest", "promotional", "flash sale",
                        "statement is ready", "pull request", "approved your"]
    WORK_SIGNALS     = ["company.com", "ops.", "manager", "hr@", "dev.", "github"]
    PERSONAL_SIGNALS = ["gmail.com", "yahoo.com", "hotmail.com", "mom", "dad", "friend"]

    def rule_based_agent(obs: dict, task_id: int) -> dict:
        subject = obs["subject"].lower()
        sender  = obs["sender"].lower()
        body    = obs["body"].lower()
        text    = subject + " " + sender + " " + body

        # Determine label
        if any(s in text for s in SPAM_SIGNALS):
            label    = "spam"
            priority = "low"
            category = "spam"
        elif any(s in text for s in URGENT_SIGNALS):
            label    = "urgent"
            priority = "high"
            category = "work"
        elif any(s in text for s in ARCHIVE_SIGNALS):
            label    = "archive"
            priority = "low"
            category = "newsletter" if "unsubscribe" in text else "notification"
        else:
            label = "inbox"
            if any(s in text for s in PERSONAL_SIGNALS):
                priority = "low"
                category = "personal"
            elif any(s in text for s in WORK_SIGNALS):
                priority = "medium"
                category = "work"
            else:
                priority = "medium"
                category = "work"

        return {"label": label, "priority": priority, "category": category}

    baseline_env = EmailTriageEnvironment()
    all_scores = {}

    for task_id in [1, 2, 3]:
        obs = baseline_env.reset(task_id=task_id).model_dump()

        while not obs.get("done", False):
            action_dict = rule_based_agent(obs, task_id)
            action = EmailAction(**action_dict)
            result = baseline_env.step(action).model_dump()
            obs = result["observation"]

        grade = grade_episode(baseline_env.results, task_id)
        all_scores[f"task_{task_id}"] = {
            "score":      grade["score"],
            "task_name":  grade["task_name"],
            "difficulty": ["easy", "medium", "hard"][task_id - 1],
            "details":    grade["details"],
        }

    avg = sum(v["score"] for v in all_scores.values()) / 3

    return {
        "baseline_agent": "rule-based (keyword heuristics)",
        "note": "For LLM baseline (gpt-4o-mini), run: python baseline.py",
        "average_score": round(avg, 4),
        "tasks": all_scores,
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
