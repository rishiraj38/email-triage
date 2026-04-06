"""
Email Triage Environment — core RL environment logic.

Implements the OpenEnv interface:
    reset(task_id)  → EmailObservation
    step(action)    → StepResult
    state()         → EmailState

Three tasks of increasing difficulty:
    Task 1 (easy)   — Spam Detection       : label only (spam | inbox)
    Task 2 (medium) — Priority Triage      : label + priority
    Task 3 (hard)   — Full Email Triage    : label + priority + category
"""

import random
import uuid
from typing import Dict, List, Optional

from emails import EMAILS
from models import (
    EmailAction,
    EmailObservation,
    EmailReward,
    EmailState,
    StepResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_META = {
    1: {"name": "Spam Detection",    "difficulty": "easy"},
    2: {"name": "Priority Triage",   "difficulty": "medium"},
    3: {"name": "Full Email Triage", "difficulty": "hard"},
}

# Numeric mapping for priority comparison (to give partial credit)
PRIORITY_RANK = {"high": 2, "medium": 1, "low": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    Stateful RL environment for email triage.

    An episode consists of triaging all 20 emails in the dataset (shuffled).
    The agent receives one email per step and must classify it according to
    the active task's requirements.

    Usage:
        env = EmailTriageEnvironment()
        obs = env.reset(task_id=1)
        while not obs.done:
            action = EmailAction(label="spam")
            result = env.step(action)
            obs = result.observation
        score = env.get_episode_score()
    """

    def __init__(self) -> None:
        self.episode_id: Optional[str] = None
        self.task_id: int = 1
        self.emails: List[dict] = []
        self.current_index: int = 0
        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = True
        self.results: List[dict] = []   # history — used by grader

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API  (OpenEnv spec)
    # ──────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 1) -> EmailObservation:
        """Start a new episode. Shuffles emails and returns the first one."""
        if task_id not in TASK_META:
            raise ValueError(f"task_id must be 1, 2, or 3 — got {task_id}")

        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.emails = EMAILS.copy()
        random.shuffle(self.emails)
        self.current_index = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.results = []

        return self._make_observation(reward=0.0)

    def step(self, action: EmailAction) -> StepResult:
        """
        Submit a classification action for the current email.
        Returns the next email as an observation, plus the reward earned.
        """
        if self.done:
            raise ValueError(
                "Episode is already done. Call reset() to start a new episode."
            )
        if self.current_index >= len(self.emails):
            raise ValueError("No more emails — episode should have ended.")

        current_email = self.emails[self.current_index]

        # Score the action
        reward_info = self._calculate_reward(action, current_email)

        # Record result for grader
        self.results.append(
            {
                "email_id":     current_email["id"],
                "subject":      current_email["subject"],
                "action":       action.model_dump(),
                "ground_truth": current_email["ground_truth"],
                "reward":       reward_info.value,
                "feedback":     reward_info.feedback,
                "penalties":    reward_info.penalties,
            }
        )

        # Advance state
        self.cumulative_reward += reward_info.value
        self.step_count += 1
        self.current_index += 1

        if self.current_index >= len(self.emails):
            self.done = True

        obs = self._make_observation(reward=reward_info.value)

        return StepResult(
            observation=obs,
            reward=reward_info.value,
            done=self.done,
            info={
                "label_score":    reward_info.label_score,
                "priority_score": reward_info.priority_score,
                "category_score": reward_info.category_score,
                "feedback":       reward_info.feedback,
                "penalties":      reward_info.penalties,
                "ground_truth":   current_email["ground_truth"],
            },
        )

    def state(self) -> EmailState:
        """Return current episode metadata without email content."""
        n = len(self.emails) if self.emails else len(EMAILS)
        score = self.cumulative_reward / self.step_count if self.step_count > 0 else 0.0
        meta = TASK_META.get(self.task_id, {"name": "Unknown", "difficulty": "unknown"})

        return EmailState(
            episode_id=self.episode_id or "",
            task_id=self.task_id,
            task_name=meta["name"],
            task_difficulty=meta["difficulty"],
            step_count=self.step_count,
            total_emails=n,
            cumulative_reward=round(self.cumulative_reward, 4),
            score=round(max(0.0, min(1.0, score)), 4),
            done=self.done,
        )

    def get_episode_score(self) -> float:
        """
        Final normalized score for the completed episode [0.0, 1.0].
        Called by the grader after the episode is done.
        """
        if not self.results:
            return 0.0
        avg = sum(r["reward"] for r in self.results) / len(self.results)
        return round(max(0.0, min(1.0, avg)), 4)

    # ──────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────

    def _make_observation(self, reward: float) -> EmailObservation:
        """Build an observation from current environment state."""
        # Terminal observation — no email content
        if self.done or self.current_index >= len(self.emails):
            return EmailObservation(
                email_id="EPISODE_DONE",
                subject="",
                sender="",
                body="All emails have been processed. Call /reset to start a new episode.",
                timestamp="",
                step=self.step_count,
                total_emails=len(self.emails),
                emails_remaining=0,
                reward=reward,
                cumulative_reward=round(self.cumulative_reward, 4),
                done=True,
            )

        email = self.emails[self.current_index]
        return EmailObservation(
            email_id=email["id"],
            subject=email["subject"],
            sender=email["sender"],
            body=email["body"],
            timestamp=email["timestamp"],
            step=self.step_count,
            total_emails=len(self.emails),
            emails_remaining=len(self.emails) - self.current_index,
            reward=reward,
            cumulative_reward=round(self.cumulative_reward, 4),
            done=False,
        )

    def _calculate_reward(self, action: EmailAction, email: dict) -> EmailReward:
        """
        Score an action against ground truth for the current task.

        Reward structure (max 1.0 per email):
          Task 1: 1.0 for correct spam/not-spam classification
          Task 2: 0.5 label + 0.5 priority, penalty for missed urgents
          Task 3: 0.35 label + 0.35 priority + 0.30 category,
                  penalty for missed urgents and false-positive spam
        """
        gt = email["ground_truth"]
        label_score    = 0.0
        priority_score = 0.0
        category_score = 0.0
        feedback_parts: List[str] = []
        penalties: List[str] = []

        # ── TASK 1: Spam Detection ─────────────────────────────────
        if self.task_id == 1:
            pred_spam = action.label == "spam"
            true_spam = gt["label"] == "spam"

            if pred_spam == true_spam:
                label_score = 1.0
                feedback_parts.append("✓ Correct spam classification")
            else:
                label_score = 0.0
                if true_spam:
                    feedback_parts.append(
                        f"✗ Missed spam (got '{action.label}', should be 'spam')"
                    )
                else:
                    feedback_parts.append(
                        f"✗ False positive — legitimate email labelled spam (got 'spam', should be 'inbox')"
                    )

            total = label_score

        # ── TASK 2: Priority Triage ────────────────────────────────
        elif self.task_id == 2:
            # Label (0.5 weight)
            if action.label == gt["label"]:
                label_score = 1.0
                feedback_parts.append("✓ Correct label")
            elif action.label in ("inbox", "urgent") and gt["label"] in ("inbox", "urgent"):
                # Partial: agent identified email as important, just confused urgency level
                label_score = 0.5
                feedback_parts.append(
                    f"~ Close label (got '{action.label}', expected '{gt['label']}')"
                )
            else:
                label_score = 0.0
                feedback_parts.append(
                    f"✗ Wrong label (got '{action.label}', expected '{gt['label']}')"
                )

            # Priority (0.5 weight) — partial credit for adjacent levels
            pred_p = PRIORITY_RANK.get(action.priority, 1)
            true_p = PRIORITY_RANK.get(gt["priority"], 1)

            if pred_p == true_p:
                priority_score = 1.0
                feedback_parts.append("✓ Correct priority")
            elif abs(pred_p - true_p) == 1:
                priority_score = 0.5
                feedback_parts.append(
                    f"~ Close priority (got '{action.priority}', expected '{gt['priority']}')"
                )
            else:
                priority_score = 0.0
                feedback_parts.append(
                    f"✗ Wrong priority (got '{action.priority}', expected '{gt['priority']}')"
                )

            # Critical penalty: urgent email assigned low priority
            if gt["label"] == "urgent" and action.priority == "low":
                penalties.append(
                    "MISSED_URGENT: Urgent email assigned low priority (−0.30)"
                )

            penalty = 0.30 if penalties else 0.0
            total = (label_score * 0.5 + priority_score * 0.5) - penalty

        # ── TASK 3: Full Email Triage ──────────────────────────────
        elif self.task_id == 3:
            # Label (0.35 weight)
            if action.label == gt["label"]:
                label_score = 1.0
                feedback_parts.append("✓ Correct label")
            elif action.label in ("inbox", "urgent") and gt["label"] in ("inbox", "urgent"):
                label_score = 0.5
                feedback_parts.append(
                    f"~ Close label (got '{action.label}', expected '{gt['label']}')"
                )
            else:
                label_score = 0.0
                feedback_parts.append(
                    f"✗ Wrong label (got '{action.label}', expected '{gt['label']}')"
                )

            # Priority (0.35 weight)
            pred_p = PRIORITY_RANK.get(action.priority, 1)
            true_p = PRIORITY_RANK.get(gt["priority"], 1)

            if pred_p == true_p:
                priority_score = 1.0
                feedback_parts.append("✓ Correct priority")
            elif abs(pred_p - true_p) == 1:
                priority_score = 0.5
                feedback_parts.append(
                    f"~ Close priority (got '{action.priority}', expected '{gt['priority']}')"
                )
            else:
                priority_score = 0.0
                feedback_parts.append(
                    f"✗ Wrong priority (got '{action.priority}', expected '{gt['priority']}')"
                )

            # Category (0.30 weight)
            if action.category == gt["category"]:
                category_score = 1.0
                feedback_parts.append("✓ Correct category")
            else:
                category_score = 0.0
                feedback_parts.append(
                    f"✗ Wrong category (got '{action.category}', expected '{gt['category']}')"
                )

            # Penalty: urgent email not flagged as high priority
            if gt["label"] == "urgent" and action.priority != "high":
                penalties.append(
                    "MISSED_URGENT: Critical email not assigned high priority (−0.30)"
                )
            # Penalty: legitimate email falsely marked as spam
            if gt["label"] != "spam" and action.label == "spam":
                penalties.append(
                    "FALSE_SPAM: Legitimate email marked as spam (−0.20)"
                )

            penalty = sum(
                0.30 if "MISSED_URGENT" in p else 0.20 for p in penalties
            )
            total = (
                label_score    * 0.35
                + priority_score * 0.35
                + category_score * 0.30
            ) - penalty

        else:
            total = 0.0

        total = round(max(0.0, min(1.0, total)), 4)

        return EmailReward(
            value=total,
            label_score=label_score,
            priority_score=priority_score,
            category_score=category_score,
            feedback=" | ".join(feedback_parts),
            penalties=penalties,
        )
