"""
Episode grader for the Email Triage OpenEnv environment.

Called after an episode is complete (all 20 emails processed).
Returns a normalized score [0.0, 1.0] plus detailed breakdown.
"""

from typing import Any, Dict, List


def grade_episode(results: List[dict], task_id: int) -> Dict[str, Any]:
    """
    Grade a completed episode.

    Args:
        results : list of per-step result dicts stored by the environment
        task_id : 1 (easy), 2 (medium), or 3 (hard)

    Returns:
        dict with keys:
            score          float [0.0, 1.0] — primary metric
            task_id        int
            task_name      str
            total_emails   int
            details        dict — accuracy breakdown
            per_email      list — per-email score and feedback
    """
    if not results:
        return {
            "score": 0.0,
            "task_id": task_id,
            "task_name": _task_name(task_id),
            "total_emails": 0,
            "details": {},
            "per_email": [],
        }

    n = len(results)

    # ── Overall score ──────────────────────────────────────────────
    avg_reward = sum(r["reward"] for r in results) / n
    score = round(max(0.0, min(1.0, avg_reward)), 4)

    # ── Per-dimension accuracy ─────────────────────────────────────
    correct_labels = sum(
        1 for r in results if r["action"]["label"] == r["ground_truth"]["label"]
    )
    correct_priorities = sum(
        1 for r in results if r["action"]["priority"] == r["ground_truth"]["priority"]
    )
    correct_categories = sum(
        1 for r in results if r["action"]["category"] == r["ground_truth"]["category"]
    )

    # ── High-stakes mistake tracking ──────────────────────────────
    missed_urgent = [
        {
            "email_id": r["email_id"],
            "subject":  r.get("subject", ""),
            "action_label":    r["action"]["label"],
            "action_priority": r["action"]["priority"],
        }
        for r in results
        if r["ground_truth"]["label"] == "urgent"
        and r["action"]["label"] != "urgent"
    ]

    false_spam = [
        {
            "email_id": r["email_id"],
            "subject":  r.get("subject", ""),
            "true_label": r["ground_truth"]["label"],
        }
        for r in results
        if r["ground_truth"]["label"] != "spam"
        and r["action"]["label"] == "spam"
    ]

    # ── Spam detection specific metrics (Task 1) ───────────────────
    true_spam_caught = sum(
        1 for r in results
        if r["ground_truth"]["label"] == "spam" and r["action"]["label"] == "spam"
    )
    total_spam = sum(1 for r in results if r["ground_truth"]["label"] == "spam")

    # ── Build output ───────────────────────────────────────────────
    details: Dict[str, Any] = {
        "label_accuracy":    round(correct_labels    / n, 4),
        "priority_accuracy": round(correct_priorities / n, 4),
        "category_accuracy": round(correct_categories / n, 4),
        "spam_recall":       round(true_spam_caught / total_spam, 4) if total_spam else None,
        "missed_urgent_count": len(missed_urgent),
        "false_spam_count":    len(false_spam),
    }

    if missed_urgent:
        details["missed_urgent_emails"] = missed_urgent
    if false_spam:
        details["false_spam_emails"] = false_spam

    per_email = [
        {
            "email_id":    r["email_id"],
            "reward":      round(r["reward"], 4),
            "feedback":    r["feedback"],
            "penalties":   r.get("penalties", []),
            "action":      r["action"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
    ]

    return {
        "score":        score,
        "task_id":      task_id,
        "task_name":    _task_name(task_id),
        "total_emails": n,
        "details":      details,
        "per_email":    per_email,
    }


def _task_name(task_id: int) -> str:
    return {1: "Spam Detection", 2: "Priority Triage", 3: "Full Email Triage"}.get(
        task_id, "Unknown"
    )
