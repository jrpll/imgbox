# syntax=docker/dockerfile:1
#
# Clean-install image for the imgbox web app: builds the React/Vite SPA, then
# serves it together with the FastAPI + SD3 backend from a single CUDA process.
# Mirrors the "single command launch" flow (npm run build -> python app.py),
# packaged so it runs on a from-scratch machine. See docs/clean-install-test.md.

# ── Stage 1: build the SPA ──────────────────────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /app/frontend

# Install from the lockfile first so this layer caches unless deps change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build. Intentionally NOT setting VITE_API_URL: the SPA is served same-origin
# by FastAPI, so api.js falls back to relative URLs (its '' branch).
COPY frontend/ ./
RUN npm run build
# → /app/frontend/dist

# ── Stage 2: CUDA runtime that serves SPA + API ─────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS app
ENV DEBIAN_FRONTEND=noninteractive

# Runtime libs needed by opencv / insightface / Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv provisions Python 3.12 (per server/.python-version) and installs deps.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app/server

# Dependency layer (cached unless pyproject/uv.lock change). --frozen installs
# exactly what's locked — the same resolution that already works on your machine.
COPY server/pyproject.toml server/uv.lock server/.python-version ./
RUN uv sync --frozen --no-install-project

# App source + the SPA from stage 1. app.py resolves ../frontend/dist relative to
# itself (server/app.py -> /app/frontend/dist), so match that path or the mount
# silently won't fire.
COPY server/ ./
COPY --from=frontend /app/frontend/dist /app/frontend/dist
RUN uv sync --frozen

# Cold by default: the model cache is ephemeral, so the first inference downloads
# the weights just like a new user. Mount a volume here to warm it (see compose).
ENV HF_HOME=/root/.cache/huggingface
# Stream stdout/stderr (HF download + tqdm progress) live instead of block-buffering.
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Bind 0.0.0.0 — app.py's __main__ uses 127.0.0.1, which is unreachable from
# outside the container, so the published port wouldn't work.
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
