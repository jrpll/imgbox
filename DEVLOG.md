# Devlog

## 2026-04-15 — Dev environment fixes + image drag-and-drop

### Dev environment
- `src-tauri/tauri.conf.json`: fixed `beforeDevCommand` path from `cd ../client` to `cd client` — Tauri v2 CLI runs the command from the workspace root, not from `src-tauri/`
- `client/vite.config.js`: added `server: { port: 5173, strictPort: true }` — prevents Vite from silently switching to port 5174 when a stale process holds 5173, which caused blank pages on every other launch

### Drag-and-drop image input (`client/src/components/ImageEditor.jsx`)
- Added drag-and-drop support to the left image box
- On Linux, Tauri intercepts OS-level file drops before they reach the WebView, so the HTML `onDrop` event never fires. Fixed by listening to `tauri://drag-drop` instead
- Added `read_file` Tauri command (`src-tauri/src/lib.rs`) that reads a file path and returns its bytes — used to convert the dropped file path into a `File` object in the frontend
- Visual highlight (blue border) on drag-enter/leave kept via HTML drag events, which do fire in the WebView

### SVG import fix (`client/src/components/ImageEditor.jsx`)
- WebKitGTK (Tauri's WebView on Linux) rejects ES module imports served with `Content-Type: image/svg+xml`, causing React to fail to mount and the window to show blank
- Fixed by switching from `import boxIcon from '../assets/box.svg?url'` to `import boxIconRaw from '../assets/box.svg?raw'` — `?raw` makes Vite return the SVG content as a JS string, no MIME type issue
- Rendered inline via `dangerouslySetInnerHTML` with the SVG width/height overridden to 40px

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
