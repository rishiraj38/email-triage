"""
Email Triage OpenEnv — Python Client

Provides a clean Python interface over the HTTP API.

Usage:
    from client import EmailTriageClient

    # Basic usage
    with EmailTriageClient("http://localhost:7860") as client:
        obs = client.reset(task_id=1)
        while not obs["done"]:
            action = {"label": "spam" if "lottery" in obs["subject"].lower() else "inbox"}
            result = client.step(**action)
            obs = result["observation"]
        score = client.grader()
        print(f"Score: {score['score']}")

    # Run a full episode with a custom policy
    def my_policy(obs):
        if "urgent" in obs["subject"].lower():
            return {"label": "urgent", "priority": "high", "category": "work"}
        return {"label": "inbox", "priority": "medium", "category": "work"}

    with EmailTriageClient("http://localhost:7860") as client:
        result = client.run_episode(task_id=2, policy_fn=my_policy)
        print(result)
"""

import requests
from typing import Callable, Dict, Optional


class EmailTriageClient:
    """
    HTTP client for the Email Triage OpenEnv environment.

    Wraps all REST endpoints with clean Python methods.
    Compatible with both local Docker runs and HF Spaces deployment.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ──────────────────────────────────────────────────────────────
    # Core OpenEnv API
    # ──────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check if the server is alive."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: int = 1) -> dict:
        """
        Start a new episode.

        Args:
            task_id: 1 (easy/spam), 2 (medium/priority), 3 (hard/full)

        Returns:
            EmailObservation dict — the first email to classify.
        """
        resp = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return resp.json()

    def step(
        self,
        label: str,
        priority: str = "medium",
        category: str = "work",
    ) -> dict:
        """
        Submit your classification for the current email.

        Args:
            label    : spam | inbox | urgent | archive | delete
            priority : high | medium | low
            category : spam | work | personal | newsletter | notification | social

        Returns:
            StepResult dict: {observation, reward, done, info}
        """
        action = {"label": label, "priority": priority, "category": category}
        resp = self.session.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        """Return current episode metadata (no email content)."""
        resp = self.session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> dict:
        """List all tasks and their action schemas."""
        resp = self.session.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def grader(self) -> dict:
        """
        Grade the completed episode. Call after obs.done == True.

        Returns:
            {score, task_id, task_name, total_emails, details, per_email}
        """
        resp = self.session.post(f"{self.base_url}/grader")
        resp.raise_for_status()
        return resp.json()

    def baseline(self) -> dict:
        """Trigger the built-in rule-based baseline agent against all 3 tasks."""
        resp = self.session.get(f"{self.base_url}/baseline")
        resp.raise_for_status()
        return resp.json()

    # ──────────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────────

    def run_episode(
        self,
        task_id: int,
        policy_fn: Callable[[dict], Dict[str, str]],
        verbose: bool = False,
    ) -> dict:
        """
        Run a complete episode with a custom policy function.

        Args:
            task_id   : which task to run (1, 2, or 3)
            policy_fn : callable(obs: dict) -> dict with keys label, priority, category
            verbose   : if True, print step-by-step progress

        Returns:
            Grader result dict with final score.

        Example:
            def my_policy(obs):
                if "CRITICAL" in obs["subject"]:
                    return {"label": "urgent", "priority": "high", "category": "work"}
                return {"label": "inbox", "priority": "medium", "category": "work"}

            score = client.run_episode(task_id=2, policy_fn=my_policy, verbose=True)
        """
        obs = self.reset(task_id=task_id)
        total_reward = 0.0

        while not obs.get("done", False):
            action = policy_fn(obs)
            result = self.step(
                label=action.get("label", "inbox"),
                priority=action.get("priority", "medium"),
                category=action.get("category", "work"),
            )
            obs = result["observation"]
            total_reward += result["reward"]

            if verbose:
                step = obs["step"]
                remaining = obs["emails_remaining"]
                print(
                    f"  Step {step:2d} | reward={result['reward']:.3f} | "
                    f"cumulative={obs['cumulative_reward']:.3f} | "
                    f"remaining={remaining}"
                )

        grade = self.grader()
        if verbose:
            print(f"\n  Final Score: {grade['score']:.4f}")
        return grade

    # ──────────────────────────────────────────────────────────────
    # Context manager support
    # ──────────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    def close(self):
        self.session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo — runs when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    BASE_URL = "http://localhost:7860"

    print("=" * 60)
    print("  Email Triage Client — Quick Demo")
    print("=" * 60)

    with EmailTriageClient(BASE_URL) as client:
        # Health check
        h = client.health()
        print(f"\n✓ Server healthy: {h}")

        # Simple rule-based policy
        def demo_policy(obs: dict) -> dict:
            subject = obs["subject"].lower()
            body    = obs["body"].lower()
            text    = subject + " " + body

            if any(w in text for w in ["won", "lottery", "million", "bank details", "nigeria"]):
                return {"label": "spam", "priority": "low", "category": "spam"}
            if any(w in text for w in ["critical", "urgent", "p0", "immediately", "mandatory"]):
                return {"label": "urgent", "priority": "high", "category": "work"}
            if any(w in text for w in ["unsubscribe", "newsletter", "flash sale", "digest"]):
                return {"label": "archive", "priority": "low", "category": "newsletter"}
            return {"label": "inbox", "priority": "medium", "category": "work"}

        for task_id in [1, 2, 3]:
            print(f"\n── Task {task_id} ─────────────────")
            result = client.run_episode(task_id=task_id, policy_fn=demo_policy, verbose=False)
            print(f"  Score:    {result['score']:.4f}")
            print(f"  Task:     {result['task_name']}")
            details = result.get("details", {})
            print(f"  Label acc:    {details.get('label_accuracy', 'N/A')}")
            print(f"  Priority acc: {details.get('priority_accuracy', 'N/A')}")
            print(f"  Category acc: {details.get('category_accuracy', 'N/A')}")
