"""
OpenEnv-compliant typed Pydantic models for the Email Triage environment.

Three core model types required by the OpenEnv spec:
  - EmailAction      : what the agent sends each step
  - EmailObservation : what the agent receives each step
  - EmailState       : episode-level metadata

Plus helpers:
  - EmailReward      : detailed reward breakdown per step
  - StepResult       : full result returned by step()
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# ACTION  — what the agent sends on every step
# ─────────────────────────────────────────────────────────────────────────────

VALID_LABELS     = {"spam", "inbox", "urgent", "archive", "delete"}
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_CATEGORIES = {"spam", "work", "personal", "newsletter", "notification", "social"}


class EmailAction(BaseModel):
    """
    The action an agent takes after reading one email.

    Task 1 (easy)   → only `label` is scored  (spam | inbox)
    Task 2 (medium) → `label` + `priority` are scored
    Task 3 (hard)   → all three fields are scored
    """

    label: str = Field(
        ...,
        description=(
            "How to file this email. "
            "spam=junk/malicious | inbox=needs attention | "
            "urgent=act immediately | archive=save but no action | delete=discard"
        ),
        examples=["spam", "inbox", "urgent", "archive", "delete"],
    )
    priority: str = Field(
        default="medium",
        description="How time-sensitive is this email. high=act today | medium=act this week | low=whenever",
        examples=["high", "medium", "low"],
    )
    category: str = Field(
        default="work",
        description=(
            "What type of email this is. "
            "spam | work | personal | newsletter | notification | social"
        ),
        examples=["work", "personal", "newsletter", "notification", "social", "spam"],
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got '{v}'")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {VALID_PRIORITIES}, got '{v}'")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}, got '{v}'")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION  — what the agent receives each step
# ─────────────────────────────────────────────────────────────────────────────

class EmailObservation(BaseModel):
    """
    Everything the agent can see after a reset() or step() call.
    Contains the current email to classify plus episode progress info.
    """

    # Current email content
    email_id: str         = Field(..., description="Unique identifier for this email")
    subject: str          = Field(..., description="Email subject line")
    sender: str           = Field(..., description="Sender email address")
    body: str             = Field(..., description="Full email body text")
    timestamp: str        = Field(..., description="When the email arrived (YYYY-MM-DD HH:MM:SS)")

    # Episode progress
    step: int             = Field(..., description="Current step number (0-indexed)")
    total_emails: int     = Field(..., description="Total number of emails in this episode")
    emails_remaining: int = Field(..., description="Emails still to be processed (including current)")

    # Reward signal
    reward: float         = Field(0.0, description="Reward for the LAST action (0.0 on first obs)")
    cumulative_reward: float = Field(0.0, description="Total reward accumulated this episode")

    # Terminal flag
    done: bool            = Field(False, description="True when all emails have been processed")

    # Optional extra info
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra debugging info")


# ─────────────────────────────────────────────────────────────────────────────
# REWARD  — detailed breakdown of how a step was scored
# ─────────────────────────────────────────────────────────────────────────────

class EmailReward(BaseModel):
    """
    Detailed reward breakdown for a single step.
    Returned in StepResult.info — gives the agent rich learning signal.
    """

    value: float          = Field(..., description="Total reward for this step [0.0, 1.0]")
    label_score: float    = Field(0.0, description="Score for the label field [0.0, 1.0]")
    priority_score: float = Field(0.0, description="Score for the priority field [0.0, 1.0]")
    category_score: float = Field(0.0, description="Score for the category field [0.0, 1.0]")
    feedback: str         = Field("", description="Human-readable explanation of scoring")
    penalties: List[str]  = Field(default_factory=list, description="Any penalties applied and why")


# ─────────────────────────────────────────────────────────────────────────────
# STATE  — episode-level metadata
# ─────────────────────────────────────────────────────────────────────────────

class EmailState(BaseModel):
    """
    Episode metadata returned by state().
    Gives an overview of current episode progress without email content.
    """

    episode_id: str       = Field(..., description="Unique ID for this episode (UUID)")
    task_id: int          = Field(..., description="Which task is running (1, 2, or 3)")
    task_name: str        = Field(..., description="Human-readable task name")
    task_difficulty: str  = Field(..., description="easy | medium | hard")
    step_count: int       = Field(..., description="Steps taken so far")
    total_emails: int     = Field(..., description="Total emails in this episode")
    cumulative_reward: float = Field(..., description="Total reward accumulated so far")
    score: float          = Field(..., description="Normalized score so far [0.0, 1.0]")
    done: bool            = Field(..., description="True when episode is complete")


# ─────────────────────────────────────────────────────────────────────────────
# STEP RESULT  — full response from step()
# ─────────────────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Complete result returned by the /step endpoint.
    Matches the OpenEnv spec: (observation, reward, done, info).
    """

    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# RESET REQUEST
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""

    task_id: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Which task to run. 1=Spam Detection (easy), 2=Priority Triage (medium), 3=Full Triage (hard)",
    )
