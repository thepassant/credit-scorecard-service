# syntax=docker/dockerfile:1

# ---- build: resolve the locked runtime dependencies into a virtualenv ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- runtime: slim image, non-root, only the venv and the model ----
FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/logs && chown app /app/logs
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY models/champion.json /app/models/champion.json

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CS_MODEL_PATH=/app/models/champion.json \
    CS_PREDICTION_LOG=/app/logs/predictions.db

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
CMD ["uvicorn", "credit_scorecard_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
