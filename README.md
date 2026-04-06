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

A real-world RL environment where AI agents learn to triage emails like a professional executive assistant.

3 tasks: Spam Detection (easy) → Priority Triage (medium) → Full Email Triage (hard).

## Endpoints
- `GET /health` — liveness check
- `POST /reset` — start episode
- `POST /step` — submit action
- `GET /state` — episode metadata
- `GET /tasks` — list all tasks
- `POST /grader` — grade completed episode
- `GET /baseline` — run baseline agent
- `GET /docs` — interactive API docs
