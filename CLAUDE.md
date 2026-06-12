# imgbox

AI-powered image editing app. The core mode: user uploads an image, describes what it looks like and what they want it to become — a fine-tuned Stable Diffusion 3 pipeline produces the edit, with a slider to adjust how much of the original structure is preserved. Additional modes: Flux2 Klein generation, face-identity database, background removal.

**Development strategy: webapp first.** The app is developed and tested as a plain web app (FastAPI + Vite in the browser); the Tauri shell is a thin wrapper to be finished later. The frontend currently talks to the FastAPI server directly (including HF token via `POST /config`) and only invokes `get_server_config` from Tauri. Don't invest in sidecar/Tauri plumbing unless asked.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Desktop shell | Tauri v2 (Rust) — thin wrapper, low priority for now |
| Backend | FastAPI + Uvicorn (Python, managed with uv) |
| ML | SD3 Medium + Flux2 Klein via HuggingFace Diffusers; insightface/ONNX for identity |

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

Vite proxies `/identity` to `127.0.0.1:8080` in dev (see `vite.config.js`). The server also serves `frontend/dist` at `/` if it exists, so a built frontend works same-origin without Vite.

Python deps live in `server/pyproject.toml` (+ `uv.lock`). `server/requirement.txt` is stale — do not use or update it.

---

## Architecture

```
frontend/src/
├── App.jsx                  # Renders ImageEditor
├── components/
│   ├── ImageEditor.jsx      # The shell: header + mode dropdown + settings panel,
│   │                          Run/Reset + progress bar, result card,
│   │                          identity database browser, lightbox/overlays
│   ├── ImageDropZone.jsx    # Reusable upload zone (drag-drop incl. directory traversal,
│   │                          HEIC conversion, thumbnails) — modes render it themselves
│   ├── AdvancedSettings.jsx # Collapsible disclosure for non-prompt tuning fields
│   └── modes/
│       ├── Edit.jsx             # 'edit': two-step FINEdits flow (train+invert, then slider edits)
│       ├── Flux2Klein.jsx       # 'flux2klein': prompt + optional reference images
│       ├── Identity.jsx         # 'identity': batch face embedding into the database
│       └── RemoveBackground.jsx # 'remove background': image in, PNG out
└── lib/
    ├── api.js               # Fetch wrapper — base URL from Tauri store / VITE_API_URL,
    │                          falls back to 127.0.0.1:8080 (5173 → 8080 special-case)
    ├── persist.js           # IndexedDB wrapper (db 'imgbox', store 'modes')
    └── i18n.js              # ENG/FR/ES strings + LangContext; t('key') falls back to the key

server/
├── app.py                   # All FastAPI routes + SPA static mount
├── model_registry.py        # Single-slot GPU model cache (see below)
├── finedits.py              # FINEdits class — SD3 edit pipeline
├── flux2klein_vp.py         # Vendored/modified diffusers Flux2 Klein pipeline
├── identity.py              # Face detection + embedding model (insightface/ONNX)
├── identity_store.py        # LanceDB table + crop/original image files on disk
├── background_remover.py    # Background removal model
├── progress.py              # Global progress tracker → /progress SSE
└── device.py                # DEVICE resolution (cuda → mps → cpu) + empty_cache()

src-tauri/src/
├── lib.rs                   # Tauri commands + ServerConfig store
└── sidecar.rs               # Spawns/kills the FastAPI server (not the current focus)
```

### Mode contract

Each file under `components/modes/` exports a config object. `ImageEditor.jsx` owns the shared chrome and state — `images`, Run/Reset buttons, progress bar, result card, persistence. Modes render their own inputs, including the image drop zone:

```js
{
  label: string,            // i18n key shown in the mode dropdown
  initialState: object,     // mode's input state, opaque to the shell
  Inputs: ({ state, setState, images, setImages, onZoom }) => JSX,
  submit: async ({ images, state }) => ({ blob?, state, meta? }),
  canSubmit: ({ images, state }) => boolean,
  totalSteps?: number,      // multi-step modes: shell renders step arrows
  getStepLabel?: (state, t) => string,
  Result?: ({ result, meta, onZoom }) => JSX,  // custom result card (identity grid)
  restoreState?: (state) => state,  // massage persisted state on rehydrate
}
```

Notes:
- The shell owns `images` (so persistence/rehydration/Reset stay uniform) but modes decide where and whether to show the upload UI by rendering `<ImageDropZone images={images} onChange={setImages} multi? directory? onZoom={onZoom} />` inside their `Inputs`. Edit mode renders it only in step 1; `multi` allows unlimited images. Folders can be drag-dropped onto any multi zone (the drop handler traverses directories). A browser file dialog can't offer files and folders at once (platform limitation, even native on Linux), so `directory` (identity mode) makes a click on the zone split it into two icon halves — image icon (left) opens the image picker, folder icon (right) opens the folder picker. Without `directory`, click opens the image picker directly.
- `images` is always an array (single-image modes use `images[0]`).
- `submit` returns `blob` (result image) and/or `meta` (arbitrary JSON, rendered by `Result`).
- Modes do not use refs and do not plumb readiness up via callbacks — derive from state.
- Multi-step modes keep `step` in their state; the shell's arrows write `state.step` directly.

Per-mode last state (`modeState`, image blobs, result blob, meta) is persisted to IndexedDB on every successful submit and rehydrated when the mode is opened. If rehydration needs adjusting — e.g. resetting multi-step progress because the server's in-memory model state doesn't survive restarts — the mode exports `restoreState`. The `edit` mode uses this to snap back to step 1 with `trained: false`, since the slider depends on the fine-tuned transformer + `zts_ref` that live only in server RAM.

To add a mode: drop a new file under `components/modes/`, register it in the `MODES` map at the top of `ImageEditor.jsx`.

### Model registry (server)

`model_registry.py` holds **one model on the GPU at a time** ('edit', 'flux2klein', 'identity', 'remove-background'). `registry.acquire(name)` unloads the current model and loads the requested one. Consequences:

- Switching modes evicts the FINEdits instance, destroying the fine-tuned transformer + `zts_ref` — a subsequent `/edit` hits a fresh instance and 500s. The frontend's `restoreState` mitigates across restarts, but not across mode switches within a session.
- The lock only guards load/swap, not use — concurrent ML requests are unsafe (known limitation, single-user app).

---

## ML pipeline (finedits.py)

`FINEdits` wraps SD3 Medium and implements the FINEdits algorithm:

1. **`ft_invert(img, prompt, num_train_steps, num_inversion_steps)`** — called on `/generate`
   - Encodes image to VAE latent `z0`
   - Fine-tunes the SD3 transformer on `z0` (`num_train_steps`, 8-bit Adam on CUDA, batch size 10)
   - Inverts `z0` through `num_inversion_steps` diffusion steps, storing the full trajectory `zts_ref`

2. **`edit(prompt, neg_prompt, num_skipped_steps)`** — called on both `/generate` (after invert) and `/edit` (slider)
   - `neg_prompt` defaults to `text1` minus `text2` concepts
   - Re-denoises from `zts_ref[num_skipped_steps]` with CFG=7
   - Server maps the slider: `num_skipped_steps = (1 - slider) × num_inversion_steps`. More skipped steps → start later in the trajectory → more original structure preserved. The UI shows preservation as `(1 - slider) × 100`%, consistent with this.
   - Decodes final latent → JPEG

The fine-tuned transformer state persists in memory between `/generate` and subsequent `/edit` calls — don't reinitialize it between slider adjustments (and see the model-registry eviction caveat above).

---

## API endpoints (app.py)

| Endpoint | Input | Output |
|---|---|---|
| `POST /generate` | multipart: `image`, `text1`, `text2`, `num_train_steps`, `num_inversion_steps` | JPEG blob |
| `POST /edit` | multipart: `slider` (0–1), `text1`, `text2`, `neg_prompt?` | JPEG blob |
| `POST /flux2klein` | multipart: `images[]?`, `prompt`, `num_inference_steps?`, `diffusion_coefficient?`, `seed?`, `width?`, `height?` | JPEG blob |
| `POST /identity` | multipart: `images[]`, `caption?` | `{ processed, faces_found, skipped, ids }` |
| `GET /identity/list` | — | array of rows (id, source_filename, created_at, gender, caption, …) |
| `GET /identity/crop/{id}` | — | JPEG face crop |
| `GET /identity/original/{id}` | — | original image file |
| `DELETE /identity/{id}` | — | `{ ok }` |
| `GET /identity/match/{id}?k=` | — | k nearest-neighbor rows |
| `POST /remove-background` | multipart: `image` | PNG blob |
| `POST /config` | form: `hf_token` | `{ ok }` — writes `server/.env` |
| `GET /progress` | — | SSE stream `{ message, progress, remaining }` (single global tracker) |
| `GET /health` | — | `{"status": "ready" \| "loading", "device": "cuda" \| "mps"}`; 503 if CPU-only |

Result images get imgbox EXIF tags (Software + DateTime) via `_save_with_exif`.

---

## Tauri (low priority for now)

Commands exposed via `invoke`: `get_server_config` (the only one the frontend currently calls), `set_server_config`, `read_file`. Two intended modes: `local` (sidecar, needs GPU) and `remote` (user-hosted FastAPI). The sidecar lifecycle, token-to-sidecar flow, and `read_file`-based drag-drop are all unfinished — known and deferred until the webapp is solid.

---

## Key quirks to remember

- **Linux drag-drop (Tauri only)**: WebKitGTK intercepts file drops before HTML events fire. The fix is to listen to the `tauri://drag-drop` event instead of `onDrop` — not yet implemented; HTML drag-drop works fine in the browser.
- **SVG import**: WebKitGTK rejects ES module imports with `Content-Type: image/svg+xml`. Use `?raw` import + `dangerouslySetInnerHTML`.
- **Relative `/identity/...` image URLs** in the frontend rely on the Vite dev proxy (or same-origin SPA serving). They bypass `lib/api.js`'s base URL, so they break in remote-server mode.
- **Vite port**: configured to `strictPort: true` on 5173. Kill stale processes before starting dev.
- **Backend port**: always start uvicorn with `--port 8080`. Default is 8000, which the frontend won't find.
- **Device selection**: `server/device.py` is the single source of truth (`DEVICE` resolves cuda → mps → cpu; `empty_cache()` dispatches per backend). Never hardcode `"cuda"` / `.cuda()` in server code. `/health` returns 503 when only CPU is available.
- **MPS caveats**: bitsandbytes `Adam8bit` is CUDA-only — `ft_invert` falls back to `torch.optim.Adam` on MPS (full fp32 optimizer states, so edit mode realistically needs a 32 GB+ Mac). `onnxruntime-gpu` has no macOS wheels; pyproject splits onnxruntime by platform.
- **HEIC uploads**: converted client-side via dynamic `heic2any` import in `ImageEditor.jsx`; server also registers `pillow_heif`.

---

## Open TODOs

- Frontend error states: failed requests currently only `console.error` — the result card just goes blank
- Guard `/edit` when no inversion state exists (return 4xx instead of 500); validate `slider` and step counts
- Lock or 409 concurrent ML requests (model registry races)
- Restrict CORS (currently `*` on an unauthenticated local server) + sanitize `id_` in identity_store filters/globs
- Model loading screen: poll `/health` on startup
- Masking: `do_masking` exists in FINEdits but isn't wired to the frontend
- Later (Tauri phase): settings UI for local/remote, sidecar lifecycle (crash detect, exit cleanup), CSP, scope or drop `read_file`, packaging (PyInstaller, CI builds)
