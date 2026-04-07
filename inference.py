import json
import os
import sys

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

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

def get_llm_action(client: OpenAI, obs: dict, task_id: int) -> dict:
    email_text = (
        f"From: {obs['sender']}\n"
        f"Subject: {obs['subject']}\n"
        f"Date: {obs['timestamp']}\n\n"
        f"{obs['body']}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": TASK_SYSTEM_PROMPTS[task_id]},
                {"role": "user", "content": f"Classify this email:\n\n{email_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=100,
        )
        action_dict = json.loads(response.choices[0].message.content)
    except Exception as e:
        # Fallback
        action_dict = {}

    return {
        "label":    action_dict.get("label",    "inbox"),
        "priority": action_dict.get("priority", "medium"),
        "category": action_dict.get("category", "work"),
    }

def run_task(task_id: int, llm_client: OpenAI):
    task_names = {1: "spam-detection", 2: "priority-triage", 3: "full-email-triage"}
    task_name = task_names[task_id]
    
    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}", flush=True)
    
    try:
        resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"Failed to reset environment for task {task_id}: {e}", file=sys.stderr)
        return

    step = 0
    rewards = []
    success = False

    while not obs.get("done", False):
        try:
            action = get_llm_action(llm_client, obs, task_id)
            resp = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            obs = result["observation"]
            reward = result["reward"]
            done = obs.get("done", False)
            error = "null"
        except Exception as e:
            action = {}
            reward = 0.0
            done = True
            error = str(e).replace('\n', ' ')
            
        step += 1
        rewards.append(reward)
        action_str = json.dumps(action).replace('"', "'")
        
        done_str = "true" if done else "false"
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error}", flush=True)
        
    try:
        resp = requests.post(f"{ENV_BASE_URL}/grader", timeout=30)
        resp.raise_for_status()
        grade = resp.json()
        score = grade["score"]
        success = True
    except Exception:
        score = 0.0
        success = False
        
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success_str = "true" if success else "false"
    
    print(f"[END] success={success_str} steps={step} score={score:.2f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable must be set.", file=sys.stderr)
        sys.exit(1)

    try:
        llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        h = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        h.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}\nError: {e}", file=sys.stderr)
        sys.exit(1)

    for tid in [1, 2, 3]:
        run_task(tid, llm_client)
