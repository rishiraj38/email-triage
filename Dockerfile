# ─────────────────────────────────────────────────────────────────────────────
# Email Triage — OpenEnv Environment
# Dockerfile for Hugging Face Spaces + local Docker deployment
#
# Build:  docker build -t email-triage:latest .
# Run:    docker run -d -p 7860:7860 email-triage:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Environment variables ─────────────────────────────────────────────────────
# HF Spaces uses port 7860 by default.
# Override with: docker run -e PORT=8000 ...
ENV PORT=7860
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start server ─────────────────────────────────────────────────────────────
CMD uvicorn server.app:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers 1 \
    --log-level info
