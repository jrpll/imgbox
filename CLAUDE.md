# imgbox

AI-powered image editing app. The core mode: user uploads an image, describes what it looks like and what they want it to become — a fine-tuned Stable Diffusion 3 pipeline produces the edit, with a slider to adjust how much of the original structure is preserved. Additional modes: Flux2 Klein generation, SDEdit (paint on a photo, renoise, denoise), face-identity database, background removal.

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
│   ├── ImageEditor.jsx      # The shell: header + mode dropdown (task stickers) + settings
│   │                          panel, stepper, Run/Reset·Cancel + progress bar, result card
│   │                          (error strip, controls under the image, history strip),
│   │                          send-to picker, lightbox
│   ├── ImageDropZone.jsx    # Reusable upload zone (drag-drop incl. directory traversal, paste,
│   │                          HEIC conversion, thumbnails) — modes render it themselves
│   ├── AdvancedSettings.jsx # Collapsible disclosure for non-prompt tuning fields; remembers
│   │                          whether it was left open (localStorage)
│   ├── ModelsPanel.jsx      # Settings panel list: which models are on disk, download + progress
│   ├── HistoryStrip.jsx     # Thumbnails of a mode's past results, under the result card
│   └── modes/
│       ├── Edit.jsx             # 'edit': two-step FINEdits flow (train+invert, then slider edits)
│       ├── Flux2Klein.jsx       # 'flux2klein': prompt + optional reference images
│       ├── Identity.jsx         # 'identity': batch face embedding; its Result is the faces
│       │                          table (search, view / match / delete) and their overlays
│       ├── RemoveBackground.jsx # 'remove background': image in, PNG out
│       ├── SDEdit.jsx           # 'sdedit': its Result is a paint canvas — pipette a colour off the
│       │                          photo, brush it on; Run noises the painted photo and denoises it
│       └── Sam3.jsx             # 'sam3': its Result is the canvas — click points on the image at
│                                  full size, SAM3 cuts them out live; Run is only for the text prompt
└── lib/
    ├── api.js               # Fetch wrapper — base URL from Tauri store / VITE_API_URL,
    │                          falls back to 127.0.0.1:8080 (5173 → 8080 special-case)
    ├── persist.js           # IndexedDB wrapper (db 'imgbox', store 'modes'); updateState merges atomically
    └── i18n.js              # ENG/FR/ES strings + LangContext; t('key') falls back to the key

server/
├── app.py                   # All FastAPI routes + SPA static mount
├── model_registry.py        # Single-slot GPU model cache (see below)
├── finedits.py              # FINEdits class — SD3 edit pipeline
├── flux2klein_vp.py         # Vendored/modified diffusers Flux2 Klein pipeline (VP-SDE sampler;
│                              `init_image` + `strength` start it part-way, for SDEdit)
├── identity.py              # Face detection + embedding model (insightface/ONNX)
├── identity_store.py        # LanceDB table + crop/original image files on disk
├── background_remover.py    # Background removal model
├── sam3_segmenter.py        # SAM3 interactive point segmentation (external package, see below)
├── model_store.py           # Catalog of what the modes need on disk: status + download with progress
├── jobs.py                  # Job queue: one worker thread runs all ML work in order (see below)
├── progress.py              # Global progress tracker → /progress SSE; its ticks are where a cancel lands
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
  tasks: string[],          // 'edit' | 'generate' | 'cutout' | 'faces': the stickers beside the label
  model: string,            // registry name of the model it runs on: the header shows whether it is loaded
  initialState: object,     // mode's input state, opaque to the shell
  Inputs: ({ state, setState, images, setImages, onZoom, onRun, onError }) => JSX,
  submit: async ({ images, state, run }) => ({ blob?, state, meta? }),  // run(path, formData) → Response
  canSubmit: ({ images, state }) => boolean,
  hideRun?: boolean,        // live modes: no Run button, Inputs pushes results via onResult
  steps?: [{ label, enabled?: (state) => boolean }],  // multi-step modes: shell renders the stepper
  Result?: ({ result, meta, onZoom, onSend, state, setState, images, loaded, onResult }) => JSX,
                            // custom result card (flux grid, faces table, SAM3 / SDEdit canvas)
  ResultControls?: ({ state, setState, onRun }) => JSX | null,  // under the result (Edit's slider)
  multi?: boolean,          // takes several images: one sent here is added to them, not swapped in
  restoreState?: (state, health) => state,  // massage persisted state on rehydrate; health = GET /health
  loadFirst?: boolean,      // the model is loaded explicitly: Run reads Load model until then
                            //   (Sam3, whose clicks need it there). Others load on their first run.
}
```

Notes:
- The shell owns `images` (so persistence/rehydration/Reset stay uniform) but modes decide where and whether to show the upload UI by rendering `<ImageDropZone images={images} onChange={setImages} multi? directory? onZoom={onZoom} />` inside their `Inputs`. Edit mode renders it only in step 1; `multi` allows unlimited images. Folders can be drag-dropped onto any multi zone (the drop handler traverses directories). A browser file dialog can't offer files and folders at once (platform limitation, even native on Linux), so `directory` (identity mode) makes a click on the zone split it into two icon halves — image icon (left) opens the image picker, folder icon (right) opens the folder picker. Without `directory`, click opens the image picker directly.
- `images` is always an array (single-image modes use `images[0]`).
- `submit` returns `blob` (result image) and/or `meta` (arbitrary JSON, rendered by `Result`). It makes its request through `run(path, formData)`, not `apiPost`: the shell starts it as a server-side job (`?job=1`), stores the job id with the mode's persisted state, and resolves `run` with the response the route would have given. After a reload the shell calls `submit` again with a `run` that only waits on that stored job — so `submit` must do nothing with side effects before `run`. While the job runs the Reset button reads Cancel (`DELETE /jobs/{id}`); a queued job shows Queued. The `state` it returns is applied only if it is a new object (Edit moving on to step 2): returned as given, it is where the run started, and the inputs may have moved on since (SDEdit strokes painted while it ran).
- Chaining: the result header's send button (and `onSend(blob)` from a custom `Result` — each flux image, each faces row) opens a picker of modes; the picked mode opens with that image as its input (`sentRef`, consumed by the hydration effect), and its `restoreState` gets an empty health. A cutout sent to FlowID carries its mask as transparency (see `/generate`).
- Modes do not use refs and do not plumb readiness up via callbacks — derive from state.
- `onRun()` runs the mode as the Run button would (Edit's `ResultControls` calls it when the preservation slider is released). `onError(message)` reports a failed request the mode makes itself (Sam3's live clicks); failed submits and model loads are caught by the shell. Either way the message lands in a strip above the result, and the previous result stays up — the shell no longer blanks it when a run starts.
- Under the result image sits a strip for what is judged against it: the before/after toggle (one image in / one image out, no custom `Result`; swaps the result for `images[0]`) and the mode's `ResultControls`. It collapses (`empty:hidden`) when there is neither, so `ResultControls` returns null rather than an empty wrapper.
- Multi-step modes keep `step` in their state; the shell's stepper writes `state.step` directly, and a step with `enabled` can't be opened until it returns true (Edit's step 2 needs `trained`).
- A mode whose output updates continuously rather than on demand sets `hideRun: true`: the shell
  drops the Run button (and the Cmd+Enter binding) and hands `Inputs` an `onResult(blob | null)`
  that writes straight into the shared result card. `submit` is then never called, and nothing is
  persisted on submit either — such a mode gets only the debounced `modeState` save.

Per-mode last state (`modeState`, image blobs, result blob, meta) is persisted to IndexedDB on every successful submit and rehydrated when the mode is opened. If rehydration needs adjusting — e.g. resetting multi-step progress because the server's in-memory model state doesn't survive restarts — the mode exports `restoreState`. The `edit` mode uses this to snap back to step 1 with `trained: false`, since the slider depends on the fine-tuned transformer + `zts_ref` that live only in server RAM — unless `/health` reports `trained`, i.e. the server still holds them.

The last 12 results of a mode (a blob, or a `meta.images` batch) are kept apart from that state, under the IndexedDB key `<mode>:history`, so the debounced `modeState` save doesn't rewrite them. `HistoryStrip` shows them under the result once there are two; Reset clears them.

A run belongs to its mode, not to the screen: `running` maps mode → job id, and you can open other modes while one trains (the dropdown shows a pulsing dot). `execute` saves everything a run produces under the run's own mode before showing any of it, and shows it only if that mode is the open one (`modeRef`); otherwise the result — or the error — is simply there when the mode is next hydrated. `endedRef` keeps a hydration that read IndexedDB just before a run ended from painting that stale read over it. Reopening a mode whose run this page is still waiting on does not pick the job up a second time.

Changing mode resets everything that belongs to a mode (`modeState`, `images`, result, history…) in the same update as `setMode`, in `handleModeChange` — never in an effect a render later: a mode's components read fields only their own state has, and throw on another's.

To add a mode: drop a new file under `components/modes/`, register it in the `MODES` map at the top of `ImageEditor.jsx`.

### Models on disk (server + settings panel)

`model_store.py` lists what each mode's loader fetches on first use — hub repo + the `allow`/`ignore` patterns that reproduce `from_pretrained`'s selection for the loader's arguments (checked against the hub listings and a populated cache), or insightface's zip. `GET /models` reports each as `ready` (all its weight `files` resolve in the HF cache — or `SAM3_CKPT` exists), `missing`, `downloading` (`done`/`total` bytes) or `error`; it reads only the disk, so it works offline. `POST /models/{name}/download` fetches in a background thread — not a job, it isn't GPU work. Hub progress is measured, not reported by the library: planned bytes from the hub listing, against the planned files already cached plus the repo's `*.incomplete` blobs. Change a loader's arguments in `model_registry.py` (a variant, a component passed as None) and the patterns here must follow.

### Model registry (server)

`model_registry.py` holds **one model on the GPU at a time** ('edit', 'flux2klein', 'identity', 'remove-background', 'sam3'). `registry.acquire(name)` unloads the current model and loads the requested one. Consequences:

- Switching modes evicts the FINEdits instance, destroying the fine-tuned transformer + `zts_ref` — a subsequent `/edit` hits a fresh instance and 500s. The frontend's `restoreState` asks `/health` (`trained`) every time the mode is opened, so the slider step is only offered while that state exists.
- The lock only guards load/swap, not use. What keeps two requests off the GPU at once is `jobs.py`: every ML route queues its work there, and a single worker thread runs the queue in order.

### Job queue (server)

`jobs.respond(fn, detach)` queues `fn` (which returns the finished `Response` or JSON). Without `?job=1` the route waits for it — the old synchronous API, still used by SAM3's live clicks and `/load`. With `?job=1` it answers `{job_id}` at once and the result waits at `GET /jobs/{id}/result` (the last 16 detached jobs are kept in RAM, so a result can be fetched again after a reload; a server restart loses them → 404). `DELETE /jobs/{id}` cancels: a queued job never starts, the running one raises `progress.Cancelled` at its next `tracker.set` — so a step in progress finishes, and a model load can't be interrupted. `ft_invert` drops the previous inversion as its first act, so a cancelled training leaves `trained: false` rather than a stale `zts_ref`.

---

## ML pipeline (finedits.py)

`FINEdits` wraps SD3 Medium and implements the FINEdits algorithm:

1. **`ft_invert(img, prompt, num_train_steps, num_inversion_steps)`** — called on `/generate`
   - Encodes image to VAE latent `z0`
   - Fine-tunes the SD3 transformer on `z0` (`num_train_steps`, 8-bit Adam on CUDA, batch size 10)
   - Inverts `z0` through `num_inversion_steps` diffusion steps, storing the full trajectory `zts_ref`

2. **`edit(prompt, neg_prompt, num_skipped_steps)`** — called on both `/generate` (after invert) and `/edit` (slider)
   - `neg_prompt` defaults to `text1` minus `text2` concepts
   - Re-denoises from `zts_ref[num_skipped_steps]` with CFG=7 — or with cond passes only (`cfg=1`) when the UI's CFG toggle is off. The toggle also covers `ft_invert`'s inversion: CFG 7 against the empty prompt when on, cond-only when off. Training is cond-only either way.
   - Server maps the slider: `num_skipped_steps = (1 - slider) × num_inversion_steps`. More skipped steps → start later in the trajectory → more original structure preserved. The UI shows preservation as `(1 - slider) × 100`%, consistent with this.
   - Decodes final latent → JPEG

The fine-tuned transformer state persists in memory between `/generate` and subsequent `/edit` calls — don't reinitialize it between slider adjustments (and see the model-registry eviction caveat above).

---

## API endpoints (app.py)

| Endpoint | Input | Output |
|---|---|---|
| `POST /generate` | multipart: `image` (EXIF-rotated, then resized to ~1 MP at its own aspect ratio, sides multiples of 16), `text1`, `text2`, `num_train_steps`, `num_inversion_steps`, `cfg?` | JPEG blob |
| `POST /edit` | multipart: `slider` (0–1), `text1`, `text2`, `neg_prompt?`, `cfg?` | JPEG blob |
| `POST /sdedit` | multipart: `image` (the photo with the strokes painted on — the client composes it; resized to ~1 MP like `/generate`), `prompt?`, `strength` (0–1, the noise level), `num_inference_steps?`, `seed?` | JPEG blob |
| `POST /flux2klein` | multipart: `images[]?`, `prompt`, `num_inference_steps?`, `diffusion_coefficient?`, `num_images_per_prompt?`, `seed?`, `width?`, `height?` | `{ images: [base64 JPEG, …] }` |
| `POST /identity` | multipart: `images[]`, `caption?` | `{ processed, faces_found, skipped, ids }` |
| `GET /identity/list` | — | array of rows (id, source_filename, created_at, gender, caption, …) |
| `GET /identity/crop/{id}` | — | JPEG face crop |
| `GET /identity/original/{id}` | — | original image file |
| `DELETE /identity/{id}` | — | `{ ok }` |
| `GET /identity/match/{id}?k=` | — | k nearest-neighbor rows |
| `POST /remove-background` | multipart: `image` | PNG blob |
| `POST /load/{name}` | — | `{ ok }` — `registry.acquire(name)`; the shell's Load model button for modes with `model` |
| `POST /sam3` | multipart: `image`, `points` (JSON `[[x,y],…]`, normalized), `labels` (JSON `[1,0,…]`), `preserve?`, `additive?` (each point segmented alone; label 1 adds its mask, 0 subtracts it, in click order) | RGBA PNG — the selection cut out (or kept, with `preserve`) |
| `POST /config` | form: `hf_token` | `{ ok }` — writes `server/.env` |
| `GET /progress` | — | SSE stream `{ message, progress, remaining, job }` (single global tracker; `job` = id of the running job) |
| `GET /jobs/{id}/result` | — | waits for the job, then answers as its route would have; 409 if cancelled, 500 if it failed, 404 if unknown |
| `DELETE /jobs/{id}` | — | `{ ok }` — cancel |
| `GET /models` | — | `[{ name, label, modes, status, size, done, total, error }]` — see Models on disk |
| `POST /models/{name}/download` | — | `{ ok }` — starts the download; poll `GET /models` |
| `GET /health` | — | `{"status": "ready" \| "loading", "device": "cuda" \| "mps", "model": <registry name> \| null, "trained": bool}` (`trained`: the loaded FINEdits instance holds an inversion); 503 if CPU-only |

Every ML route takes `?job=1` (see Job queue). A failure of the work comes back as `500 {"detail": "<ExceptionType>: <message>"}`, which `lib/api.js` turns into the `Error` message the UI shows. Flux prompt embeddings are cached per prompt (`_flux_embed`), so a repeated prompt doesn't swap the text encoder back in. `/identity` stores every face of an image: the best one under the image's sha256 (as before), the others as `<sha>_<n>`, all sharing one original file.

Result images get imgbox EXIF tags (Software + DateTime) via `_save_with_exif`.

---

## SDEdit (modes/SDEdit.jsx, `/sdedit`)

Paint on the photo, then noise it and denoise it with FLUX.2 klein base — the registry's
`flux2klein`, so going between this mode and FLUX.2 [klein] reloads nothing.

- **Server**: no pipeline of its own. `Flux2KleinVPSDEPipeline.__call__` takes `init_image` +
  `strength`: the image is encoded and noised to VP time `strength` (`alpha·x0 + sigma·noise`,
  the marginal the sampler expects there), and the timesteps start at `1000 · strength`
  instead of 1000. `strength` is thus SDEdit's t0 on the VP schedule (0.6 ≈ 86% noise in the
  model's own OT time). The route runs `round(steps · strength)` steps — the share of a full
  schedule below t0 — as an ODE (`diffusion_norm=0`, like `/flux2klein`), at CFG 4, or with
  cond passes only when the prompt is empty. The whole image is re-generated: there is no mask.
- **Client**: the result card is the canvas — the photo under an SVG of the strokes, like
  SAM3's eraser but without zoom. The pipette reads a pixel of the displayed photo through a
  1px canvas; strokes are `{ color, r, pts }`, normalized like SAM3's. `submit` paints them
  onto the photo in a canvas and uploads that JPEG: composed in the browser so the stroke
  colours are the ones the pipette read off the same decode (EXIF rotation, colour profile).
- Strokes live in `state.strokes[<name>:<size> of the image]`: they survive a reload or a
  mode switch, and are never shown on another image — nothing has to clear them.
- A run's result replaces the canvas until a tool is picked up again (or the Result toggle);
  the drop zone clears the result when the image changes.

---

## Tauri (low priority for now)

Commands exposed via `invoke`: `get_server_config` (the only one the frontend currently calls), `set_server_config`, `read_file`. Two intended modes: `local` (sidecar, needs GPU) and `remote` (user-hosted FastAPI). The sidecar lifecycle, token-to-sidecar flow, and `read_file`-based drag-drop are all unfinished — known and deferred until the webapp is solid.

---

## SAM3 (sam3_segmenter.py)

The `sam3` package is **not vendored and not a pip dependency** — `sam3_segmenter.py` puts
`SAM3_PKG` on `sys.path` and loads the checkpoint at `SAM3_CKPT` (both env-overridable;
they default to `/home_pers/jripoll/flux2-sde-sampling`). If the checkpoint is missing the
builder falls back to downloading `facebook/sam3` from HuggingFace. Its Python deps
(`eva-decord`, `einops`, `ftfy`, `iopath`, `pycocotools`, `setuptools<81` for `pkg_resources`)
*are* in `server/pyproject.toml`, since sam3 imports them at module scope.

`model.inst_interactive_predictor` is the point-prompt head, but it has **no backbone of its
own** — `build_tracker` is called with `with_backbone=False`, so `predictor.set_image()` raises
`AttributeError: 'NoneType' object has no attribute 'forward_image'`. The working path is
`Sam3Processor.set_image(img)` (runs the detector's shared backbone and prepares
`sam2_backbone_out`) followed by `model.predict_inst(state, point_coords=…, point_labels=…)`.

The per-image `inference_state` is cached on the segmenter, keyed by an md5 of the pixel
buffer, so successive clicks on the same image only re-run the mask decoder. Mode switching
evicts the instance and drops that cache (it is rebuilt on the next call — unlike FINEdits,
nothing is lost but time).

Clicks need no Run step: each one POSTs to `/sam3` (synchronously — see Job queue) and the
returned cutout becomes the shell's `result`. The mode's `Result` is the canvas: the source
image fitted to the result card (a size container; the frame's width is
`min(100cqw, 100cqh × ratio)`, so it hugs the image and clicks map to [0, 1]), with `result`
stacked over a 50%-opacity blue wash — opaque wherever it was kept — as the selection. Its
toolbar (eraser, undo, clear, brush) lives there too, plus a Result toggle that shows the bare
cutout. `Inputs` keeps the drop zone, the toggles, the text prompt (the one thing Run is for)
and the effect that makes the request. Since the overlay is whatever `result` is, anything
that swaps the image must clear it: the drop zone's `onChange` does, and the shell does not
restore a result next to an image that was sent in.

---

## Key quirks to remember

- **Linux drag-drop (Tauri only)**: WebKitGTK intercepts file drops before HTML events fire. The fix is to listen to the `tauri://drag-drop` event instead of `onDrop` — not yet implemented; HTML drag-drop works fine in the browser.
- **SVG import**: WebKitGTK rejects ES module imports with `Content-Type: image/svg+xml`. Use `?raw` import + `dangerouslySetInnerHTML`.
- **Relative `/identity/...` image URLs** in the frontend rely on the Vite dev proxy (or same-origin SPA serving). They bypass `lib/api.js`'s base URL, so they break in remote-server mode.
- **Vite port**: configured to `strictPort: true` on 5173. Kill stale processes before starting dev.
- **Backend port**: always start uvicorn with `--port 8080`. Default is 8000, which the frontend won't find.
- **Device selection**: `server/device.py` is the single source of truth (`DEVICE` resolves cuda → mps → cpu; `empty_cache()` dispatches per backend). Never hardcode `"cuda"` / `.cuda()` in server code. `/health` returns 503 when only CPU is available.
- **MPS caveats**: `DTYPE` is bf16 on every backend (`device.py`) — fp32 on MPS OOMs unified memory. bitsandbytes `Adam8bit` is CUDA-only, so on MPS `ft_invert` freezes the transformer and trains a **rank-2 peft LoRA** with plain Adam (full-param Adam's fp32 optimizer state crashes the host). The LoRA fits memory but can't memorize the source image like full fine-tuning, so **edit quality on MPS is degraded**, not just slower. flux2klein's `step_sde` autocast is fp16 on CUDA but bf16 on MPS (fp16 there overflows the VP-SDE math to NaN), and `model_registry` loads flux2klein with `enable_model_cpu_offload` on MPS (no `.to()`) to keep the idle Qwen3 text encoder off the GPU during the denoise loop. `onnxruntime-gpu` has no macOS wheels; pyproject splits onnxruntime by platform. See DEVLOG 2026-06-17 for the full history.
- **HEIC uploads**: converted client-side via dynamic `heic2any` import in `ImageEditor.jsx`; server also registers `pillow_heif`.

---

## Open TODOs

- Guard `/edit` when no inversion state exists (return 4xx instead of 500); validate `slider` and step counts
- Restrict CORS (currently `*` on an unauthenticated local server) + sanitize `id_` in identity_store filters/globs
- Model loading screen: poll `/health` on startup
- Masking: user masks are wired (transparent pixels of the `/generate` upload = the area `edit()` may change, the rest is pasted back from the inversion at every step); the attention-derived `do_masking` path still isn't
- Later (Tauri phase): settings UI for local/remote, sidecar lifecycle (crash detect, exit cleanup), CSP, scope or drop `read_file`, packaging (PyInstaller, CI builds)
