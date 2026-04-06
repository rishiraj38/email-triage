"""
Email Triage — LLM Baseline Inference Script

Uses the OpenAI API (gpt-4o-mini) to run an LLM agent against all 3 tasks
and report reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_BASE_URL=http://localhost:7860   # optional, defaults to localhost
    python baseline.py

Output:
    Prints a JSON baseline report to stdout.
    The /baseline endpoint on the server runs a rule-based agent instead.

Requirements:
    pip install openai requests
"""

import argparse
import json
import os
import sys

import requests

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL       = os.getenv("ENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL          = "gpt-4o-mini"       # fast and cheap — good for baseline

TASK_SYSTEM_PROMPTS = {
    1: """You are an expert email spam filter.
Read the email and classify it as either spam or legitimate.

Rules:
- "spam" = unsolicited, malicious, phishing, scam, or promotional junk
- "inbox" = any legitimate email from a real person or service

Respond ONLY with valid JSON (no markdown, no explanation):
{"label": "spam"} or {"label": "inbox"}""",

    2: """You are a professional executive assistant doing email triage.
Read the email and assign a label and priority.

Labels:
- "spam"    = junk, phishing, scam
- "inbox"   = legitimate email that needs attention
- "urgent"  = needs immediate action today (outages, crises, mandatory meetings)
- "archive" = save for reference but no action needed (newsletters, notifications, auto-emails)

Priority:
- "high"   = act today, cannot wait
- "medium" = act this week
- "low"    = whenever convenient

Respond ONLY with valid JSON (no markdown, no explanation):
{"label": "...", "priority": "..."}""",

    3: """You are a world-class professional email triager.
Read the email carefully and assign a label, priority, and category.

Labels: spam | inbox | urgent | archive | delete
Priorities: high | medium | low
Categories: spam | work | personal | newsletter | notification | social

Definitions:
- urgent = needs immediate action (production incidents, CEO mandates, client crises, security breaches)
- inbox  = needs attention but not immediately critical
- archive = useful to keep but no action (bank statements, GitHub notifications, newsletters)
- notification = automated system message (GitHub, bank, HR system)
- social = team events, celebrations, informal colleague emails

Respond ONLY with valid JSON (no markdown, no explanation):
{"label": "...", "priority": "...", "category": "..."}""",
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs: dict, task_id: int) -> dict:
    """Ask the LLM to classify the current email. Returns action dict."""

    email_text = (
        f"From: {obs['sender']}\n"
        f"Subject: {obs['subject']}\n"
        f"Date: {obs['timestamp']}\n\n"
        f"{obs['body']}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TASK_SYSTEM_PROMPTS[task_id]},
                {"role": "user", "content": f"Classify this email:\n\n{email_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,          # deterministic output for reproducibility
            max_tokens=100,
        )
        action_dict = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"    [LLM error: {e}] — using fallback", file=sys.stderr)
        action_dict = {}

    # Apply defaults for missing fields (always send all 3 fields)
    return {
        "label":    action_dict.get("label",    "inbox"),
        "priority": action_dict.get("priority", "medium"),
        "category": action_dict.get("category", "work"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task_id: int, llm_client: OpenAI, verbose: bool = True) -> dict:
    """Run one full episode of a task. Returns grader result dict."""

    task_names = {1: "Spam Detection (easy)", 2: "Priority Triage (medium)", 3: "Full Email Triage (hard)"}

    if verbose:
        print(f"\n  ── Task {task_id}: {task_names[task_id]} ─────────────────")

    # Reset episode
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    step = 0
    while not obs.get("done", False):
        # Get LLM action
        action = get_llm_action(llm_client, obs, task_id)

        # Submit action
        resp = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        obs = result["observation"]
        step += 1

        if verbose and step % 5 == 0:
            print(
                f"    step {step:2d}/20 | "
                f"cumulative_reward={obs.get('cumulative_reward', 0):.3f}",
                flush=True,
            )

    # Grade the episode
    resp = requests.post(f"{BASE_URL}/grader", timeout=30)
    resp.raise_for_status()
    grade = resp.json()

    if verbose:
        d = grade.get("details", {})
        print(f"  Score:        {grade['score']:.4f}")
        print(f"  Label acc:    {d.get('label_accuracy', 'N/A')}")
        print(f"  Priority acc: {d.get('priority_accuracy', 'N/A')}")
        print(f"  Category acc: {d.get('category_accuracy', 'N/A')}")
        if d.get("missed_urgent_count", 0) > 0:
            print(f"  ⚠ Missed urgent: {d['missed_urgent_count']}")
        if d.get("false_spam_count", 0) > 0:
            print(f"  ⚠ False spam:    {d['false_spam_count']}")

    return grade


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(mode: str = "print") -> dict:
    # Validate setup
    if not OPENAI_API_KEY:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "       Export it with: export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check server is up
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=10)
        h.raise_for_status()
    except Exception as e:
        print(
            f"ERROR: Cannot reach environment at {BASE_URL}\n"
            f"       Start it with: uvicorn server.app:app --port 7860\n"
            f"       Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    llm_client = OpenAI(api_key=OPENAI_API_KEY)
    verbose = mode == "print"

    if verbose:
        print("=" * 60)
        print("  Email Triage — LLM Baseline Evaluation")
        print(f"  Model: {MODEL}")
        print(f"  Environment: {BASE_URL}")
        print("=" * 60)

    task_scores = {}

    for task_id in [1, 2, 3]:
        try:
            grade = run_task(task_id, llm_client, verbose=verbose)
            task_scores[f"task_{task_id}"] = {
                "score":       grade["score"],
                "task_name":   grade["task_name"],
                "difficulty":  ["easy", "medium", "hard"][task_id - 1],
                "details":     grade.get("details", {}),
            }
        except Exception as e:
            task_scores[f"task_{task_id}"] = {
                "score": 0.0,
                "error": str(e),
                "difficulty": ["easy", "medium", "hard"][task_id - 1],
            }
            if verbose:
                print(f"  Task {task_id} FAILED: {e}", file=sys.stderr)

    avg_score = sum(v["score"] for v in task_scores.values()) / len(task_scores)

    output = {
        "baseline_model":  MODEL,
        "environment_url": BASE_URL,
        "average_score":   round(avg_score, 4),
        "tasks":           task_scores,
    }

    if mode == "api":
        # Machine-readable output for /baseline endpoint subprocess mode
        print(json.dumps(output))
    else:
        print("\n" + "=" * 60)
        print("  BASELINE RESULTS SUMMARY")
        print("=" * 60)
        for k, v in task_scores.items():
            print(f"  {v.get('task_name', k):25s} [{v.get('difficulty','?'):6s}] → {v['score']:.4f}")
        print(f"\n  Average Score: {avg_score:.4f}")
        print("=" * 60)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM baseline agent against Email Triage OpenEnv"
    )
    parser.add_argument(
        "--mode",
        default="print",
        choices=["print", "api"],
        help="print = human-readable output | api = JSON only (for /baseline endpoint)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
