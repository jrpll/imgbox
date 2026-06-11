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

---

## How to run in dev

**Backend** (requires a CUDA GPU or Apple Silicon — device auto-detected in `server/device.py`):
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
│   └── modes/
│       ├── Edit.jsx         # 'edit' mode: text1 + text2 + structure slider
│       └── RemoveBackground.jsx  # 'remove background' mode (stub)
└── lib/api.js               # Fetch wrapper — reads server URL from Tauri store
                             # or VITE_API_URL env, falls back to 127.0.0.1:8080

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
  submit: async ({ image, state }) => ({ blob, state }),   // result image bytes + new mode state
  canSubmit: ({ image, state }) => boolean,
  restoreState?: (state) => state,    // optional: massage persisted state on rehydrate
}
```

`ImageEditor.jsx` owns `image`, `result`, `isLoading`, `isEditingSlider`, and `modeState` (opaque per-mode). It renders the mode's `Inputs`, computes `canRun = !isLoading && cfg.canSubmit(...)`, and wires the shared Run/Reset buttons to `cfg.submit` and `cfg.initialState`. Modes do not call refs and do not plumb readiness up via callbacks — derive from state.

Per-mode last state (`modeState`, image blob, result blob) is persisted to IndexedDB (`lib/persist.js`, db `imgbox` / store `modes`) on every successful submit and rehydrated when the mode is opened. If rehydration needs adjusting — e.g. resetting multi-step progress because the server's in-memory model state doesn't survive restarts — the mode exports `restoreState`. The `edit` mode uses this to snap back to step 1 with `trained: false`, since the slider depends on the fine-tuned transformer + `zts_ref` that live only in server RAM.

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
| `GET /health` | — | `{"status": "ready" \| "loading", "device": "cuda" \| "mps"}`; 503 if CPU-only |

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
- **Device selection**: `server/device.py` is the single source of truth (`DEVICE` resolves cuda → mps → cpu; `empty_cache()` dispatches per backend). Never hardcode `"cuda"` / `.cuda()` in server code. `/health` returns 503 when only CPU is available.
- **MPS caveats**: bitsandbytes `Adam8bit` is CUDA-only — `ft_invert` falls back to `torch.optim.Adam` on MPS (full fp32 optimizer states, so edit mode realistically needs a 32 GB+ Mac). `onnxruntime-gpu` has no macOS wheels; pyproject splits onnxruntime by platform.

---

## Open TODOs (from README)

- Settings UI: mode switch (local/remote) + remote URL input
- Model loading screen: poll `/health` on startup
- PyInstaller: bundle FastAPI server for local distribution
- GitHub Actions: build .deb / .AppImage / .exe / .dmg on release
- Masking: `do_masking` exists in FINEdits but isn't wired to the frontend
