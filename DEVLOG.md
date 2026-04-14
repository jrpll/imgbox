# Devlog

## 2026-04-14 — Overhaul: Flask → FastAPI + Tauri

### Goal
Make the app distributable as a native installer (.deb, .exe, .dmg) while keeping remote GPU server support.

### Backend: Flask → FastAPI (`server/app.py`)
- Replaced Flask with FastAPI + Uvicorn
- Routes `/generate` and `/edit` are now async, ML work runs in a thread pool to avoid blocking the event loop
- Added `/health` endpoint returning `{"status": "ready"}` once the model is loaded
- Model loading moved into a FastAPI `lifespan` startup handler
- `UPLOAD_FOLDER` uses `tempfile` for predictable paths when running as a sidecar
- `finedits.py` untouched
- Dependencies managed with `uv` (`pyproject.toml` replaces `requirement.txt`)

### Frontend (`client/src/`)
- Added `lib/api.js`: centralised fetch wrapper that reads server URL from Tauri store (local = `127.0.0.1:8080`, remote = user-configured). Falls back to `VITE_API_URL` or `localhost:8080` in browser dev mode.
- `ImageEditor.jsx`: replaced two hardcoded `fetch('http://35.208.53.156:8080/...')` calls with `apiPost()`
- `App.jsx`: checks for HuggingFace token on launch, shows setup screen if not set
- `Setup.jsx`: one-time setup modal for entering the HF token

### Desktop shell (`src-tauri/`)
- Tauri v2 scaffolded around the existing React + Vite frontend
- `src/sidecar.rs`: starts/stops the FastAPI server as a child process, stores handle in Tauri state
- `src/lib.rs`: `ServerConfig` struct (mode, remote_url, hf_token) persisted via `tauri-plugin-store`. Two Tauri commands exposed to frontend: `get_server_config`, `set_server_config`. Sidecar auto-starts on launch if mode is `local`.
- Token passed to sidecar as `HUGGING_FACE_TOKEN` env var, read by `app.py` via `os.getenv`
- Sidecar binary placeholder at `src-tauri/binaries/imgbox-server-x86_64-unknown-linux-gnu` (real binary will be built with PyInstaller)

### Two modes
| | Local | Remote |
|---|---|---|
| Server | Tauri starts FastAPI as sidecar | User runs FastAPI manually on GPU machine |
| API URL | `127.0.0.1:8080` | Configurable in settings |
| HF token | Stored in Tauri store, injected as env var | Managed on remote machine |

### Remaining
See `README.md` todo list.
