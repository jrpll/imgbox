#!/usr/bin/env bash
cd frontend
npm install
npm run build
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  . "$HOME/.local/bin/env"
fi
cd ../server
uv sync
