# Devlog

## 2026-05-09 — Mode contract simplified, model swap registry, remove-background mode

### Mode contract (frontend)
- Replaced `forwardRef` + `useImperativeHandle` + `onCanRunChange` plumbing with a config-object contract: each mode under `frontend/src/components/modes/` exports `{ label, initialState, Inputs, submit, canSubmit }`
- `ImageEditor.jsx` owns `image`, `result`, `isLoading`, `isEditingSlider`, and an opaque `modeState`; derives `canRun = !isLoading && cfg.canSubmit(...)` and wires the shared Run/Reset to `cfg.submit` / `cfg.initialState`
- No more refs reaching into modes, no more effect-based readiness propagation
- Mode change clears image, mode state, and result for a fresh slate

### Model swap registry (`server/model_registry.py`)
- New `ModelRegistry` singleton — holds at most one model on GPU. `acquire(name)` full-unloads the current model (`gc.collect()` + `torch.cuda.empty_cache()`) and loads the new one; same-name calls are a no-op fast path. Thread-safe via a lock.
- Loaders (`_load_edit`, `_load_remove_background`) hardcoded inside the class; adding a mode is one edit in one file
- `app.py` boots with `registry.acquire('edit')` in `lifespan`; each handler calls `registry.acquire(...)` inside `_run` (off the event loop)

### Eager mode swap (`POST /mode`)
- New `/mode` endpoint takes `name` and acquires the corresponding model in the threadpool
- `handleModeChange` in `ImageEditor.jsx` fires `POST /mode` fire-and-forget on dropdown change, so the model is warm by the time the user clicks Run
- Lazy fallback preserved: if a swap is in flight when Run fires, the registry's lock serializes the request behind it

### Remove-background mode
- New `server/background_remover.py`: `BackgroundRemover` wraps RMBG-2.0 with `@torch.no_grad()` inference and aspect-preserving resize (scale longer side to 1024, run inference, mask resize back to original dimensions)
- New `POST /remove-background` route — thin handler: read upload → PIL → call `BackgroundRemover` via `run_in_threadpool` → PNG response (alpha preserved)
- `RemoveBackground.jsx` upgraded from stub to full mode: submit posts to `/remove-background`, `canSubmit` returns true once an image is uploaded
- URL/key naming aligned to hyphens everywhere (`'remove-background'` mode key, `/remove-background` route)

### Docs
- `CLAUDE.md`: `client/` → `frontend/` everywhere, architecture tree updated, "Mode contract" section added
- `README.md`: fixed broken image path (`client/src/assets/imgbox.jpeg` → `frontend/...`), closed dangling code fence that was swallowing the Todo list, updated sidecar launch command with `HUGGING_FACE_TOKEN` env var and proper uvicorn invocation

## 2026-05-09 — Frontend refactor + UI polish (`frontend/`)

### Drag-and-drop
- Replaced the Tauri-specific `tauri://drag-drop` listener (needed only for WebKitGTK on Linux) with standard HTML `onDrop` — the app is not using Tauri on Linux, so native browser drag-and-drop is sufficient.

### Mode selector
- Added a dropdown menu next to the app name to switch between modes ("edit", "remove background")
- Mode state lives in `ImageEditor.jsx`; switching resets result and canRun

### Component refactor
- `ImageEditor.jsx` is now a persistent shell: header, image upload zone, result card, shared Reset/Run buttons, and mode state
- Mode-specific input components extracted to `src/components/modes/Edit.jsx` and `modes/RemoveBackground.jsx`
- Mode components expose `submit` and `reset` imperatively via `forwardRef` + `useImperativeHandle`, and signal readiness via an `onCanRunChange` callback — keeps the buttons in the shell without lifting mode-specific state up

### Styles
- Deleted unused `App.css` (Vite boilerplate)
- Added `.input-textarea` and `.input-clear-btn` component classes to `index.css` via Tailwind `@apply` — single place to change shared input styles across all mode files

## 2026-04-20 — UI redesign: two-panel card layout

### Layout (`client/src/components/ImageEditor.jsx`, `client/src/App.jsx`)
- Replaced the old layout (two full-size image boxes side by side, inputs below) with a two-panel card layout that fills the viewport
- Left card (Input): image upload thumbnail, source and target description textareas, structure-preservation slider, Reset / Run buttons
- Right card (Result): fills remaining space; Download button appears in the card header once a result exists
- Both cards are `rounded-xl border border-gray-200` floating on a white padded background — no hard dividing lines
- Slider moved inside the Input card (below the textareas); fixed rendering by removing `appearance-none` which was stripping native browser styles
- Image thumbnail given 12 px breathing room from the box edges (`max-w/max-h calc(100% - 24px)`)
- Delete / clear buttons unified to the same small size (`p-0.5`, `X size={12}`) across image upload and text fields
- Processing spinner made inline (row layout) next to "Processing…" label instead of stacked above it

## 2026-04-20 — Three.js WebGPU spinner

### Loading state (`client/src/components/ImageEditor.jsx`)
- Replaced the plain "Processing..." text in the result box with an inline spinner + "processing..." label
- Spinner sits to the left of the label; both are styled `text-gray-400 text-3xl` to match the "resultado" placeholder

### Spinner (`client/src/spinners/spinner.js`, `client/src/components/ThreeSpinner.jsx`)
- Added `three` as a dependency
- `spinner.js`: Three.js WebGPU/TSL `Points` object drawing a hypotrochoid (spirograph) curve with a travelling particle trail. Uses `NormalBlending` so it renders correctly on the white background. Particle color matches Tailwind gray-400.
- `ThreeSpinner.jsx`: React component that mounts a `WebGPURenderer` onto a `<canvas>` ref, sets up an orthographic camera and animation loop, and cleans up on unmount. Three.js falls back to WebGL2 automatically when WebGPU is unavailable.

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
