# imgbox

AI-powered image editing desktop app. User uploads an image, describes what it looks like and what they want it to become — a fine-tuned Stable Diffusion 3 pipeline produces the edit. A slider lets the user adjust how much of the original structure is preserved.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Desktop shell | Tauri v2 (Rust) |
| Backend | FastAPI + Uvicorn (Python) |
| ML | Stable Diffusion 3 Medium via HuggingFace Diffusers |
| Loading spinner | Three.js WebGPU/TSL (falls back to WebGL2) |

---

## How to run in dev

**Backend** (requires CUDA GPU):
```bash
cd server && uv run uvicorn app:app --reload --port 8080
```

**Frontend** (separate terminal):
```bash
cd frontend && npm run dev
# → http://localhost:5173
```

The Tauri app bundles both, but for dev you run them separately and hit localhost:5173 in a browser.

---

## Architecture

```
frontend/src/
├── App.jsx                  # Routes between Setup and ImageEditor
├── components/
│   ├── ImageEditor.jsx      # Two-card shell: header + mode dropdown,
│   │                          shared image upload, mode dispatch, result viewer
│   ├── Setup.jsx            # First-run HF token entry
│   ├── ThreeSpinner.jsx     # WebGPU canvas spinner (shown while isLoading)
│   └── modes/
│       ├── Edit.jsx         # 'edit' mode: text1 + text2 + structure slider
│       └── RemoveBackground.jsx  # 'remove background' mode (stub)
├── lib/api.js               # Fetch wrapper — reads server URL from Tauri store
│                            # or VITE_API_URL env, falls back to 127.0.0.1:8080
└── spinners/spinner.js      # Three.js TSL hypotrochoid Points object

server/
├── app.py                   # FastAPI routes: /generate, /edit, /health
└── finedits.py              # FINEdits class — the full ML pipeline

src-tauri/src/
├── lib.rs                   # Tauri commands + ServerConfig store
└── sidecar.rs               # Spawns/kills the FastAPI server as a child process
```

### Mode contract

Each file under `components/modes/` exports a config object:

```js
{
  label: string,                      // shown in the mode dropdown
  initialState: object,               // mode's input state (e.g. { text1, text2, slider })
  Inputs: ({ state, setState, result, onResult, onEditingSlider }) => JSX,
  submit: async ({ image, state }) => string,   // returns a result blob URL
  canSubmit: ({ image, state }) => boolean,
}
```

`ImageEditor.jsx` owns `image`, `result`, `isLoading`, `isEditingSlider`, and `modeState` (opaque per-mode). It renders the mode's `Inputs`, computes `canRun = !isLoading && cfg.canSubmit(...)`, and wires the shared Run/Reset buttons to `cfg.submit` and `cfg.initialState`. Modes do not call refs and do not plumb readiness up via callbacks — derive from state.

To add a mode: drop a new file under `components/modes/`, register it in the `MODES` map at the top of `ImageEditor.jsx`.

---

## ML pipeline (finedits.py)

`FINEdits` wraps SD3 Medium and implements the FINEdits algorithm:

1. **`ft_invert(image, text1)`** — called on `/generate`
   - Encodes image to VAE latent `z0`
   - Fine-tunes the SD3 transformer on `z0` for 100 steps (8-bit Adam, batch size 10)
   - Inverts `z0` through 50 diffusion steps, storing the full trajectory `zts_ref`

2. **`edit(text2, neg_prompt, num_skipped_steps)`** — called on both `/generate` (after invert) and `/edit` (slider)
   - `neg_prompt` = `text1` minus `text2` concepts
   - Re-denoises from `zts_ref[num_skipped_steps]` with CFG=7
   - `num_skipped_steps = slider × 50`: higher slider → start later in trajectory → more original structure preserved
   - Decodes final latent → JPEG

The fine-tuned transformer state persists in memory between `/generate` and subsequent `/edit` calls — don't reinitialize it between slider adjustments.

---

## API endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /generate` | multipart: `image`, `text1`, `text2` | JPEG blob |
| `POST /edit` | multipart: `slider` (0–1), `text1`, `text2` | JPEG blob |
| `GET /health` | — | `{"status": "ready" \| "loading" \| "no_cuda"}` |

---

## Tauri commands (exposed to frontend via `invoke`)

| Command | Purpose |
|---|---|
| `get_server_config` | Returns `{ mode, remote_url, hf_token }` from persistent store |
| `set_server_config` | Saves config; auto-starts/stops sidecar |
| `read_file` | Reads file bytes by path — used for drag-drop on Linux |

**Two modes:** `local` (sidecar on user's machine, needs GPU) and `remote` (user-hosted FastAPI, URL configurable).

---

## Key quirks to remember

- **Linux drag-drop**: WebKitGTK intercepts file drops before HTML events fire. Must listen to `tauri://drag-drop` Tauri event instead of `onDrop`.
- **SVG import**: WebKitGTK rejects ES module imports with `Content-Type: image/svg+xml`. Use `?raw` import + `dangerouslySetInnerHTML`.
- **Vite port**: configured to `strictPort: true` on 5173. Kill stale processes before starting dev.
- **Backend port**: always start uvicorn with `--port 8080`. Default is 8000, which the frontend won't find.
- **No CUDA → 503**: server returns 503 if `torch.cuda.is_available()` is False.

---

## Open TODOs (from README)

- Settings UI: mode switch (local/remote) + remote URL input
- Model loading screen: poll `/health` on startup
- PyInstaller: bundle FastAPI server for local distribution
- GitHub Actions: build .deb / .AppImage / .exe / .dmg on release
- Masking: `do_masking` exists in FINEdits but isn't wired to the frontend
