# Clean-install test (Docker + GPU)

Emulate what a **brand-new user on a fresh machine** experiences: a from-scratch
install of the web app, served as one process, doing a **real GPU inference** —
including the first-run model download.

This packages the same "single command launch" flow the README describes
(`npm run build` → serve from FastAPI), so the SPA and API run together on one
origin at <http://localhost:8080>.

## What it proves (and what it doesn't)

- ✅ Dependencies install from scratch (npm lockfile + uv lockfile) on a clean image
- ✅ The SPA builds and is served by the backend (same-origin — no CORS / URL config)
- ✅ HF token entry persists and a real edit runs end-to-end on the GPU
- ❌ It does **not** test the Tauri desktop bundle (out of scope here)
- ❌ A successful token *save* doesn't prove the token is *valid* — that's only
  exercised when the gated model actually downloads

## Prerequisites (on the host)

1. An NVIDIA GPU + driver.
2. The **NVIDIA Container Toolkit** installed and wired into Docker.
3. A HuggingFace account that has **accepted the Stable Diffusion 3 Medium
   license** — the repo is gated, so downloads 403 without it even with a valid
   token.
4. Your HF token handy (`hf_...`).

## Run it (cold = new user)

Build the image once:

```bash
docker compose build
```

Then run it attached to your terminal:

```bash
docker compose run --rm --service-ports app
```

Use `docker compose run`, **not** `docker compose up`. `up` pipes output through
its log aggregator (the `app-1 |` line prefix), which mangles tqdm's in-place
download/progress bars into a garbled pile-up. `run` attaches your terminal
directly to the container, so progress renders cleanly — just like running
`uv run python app.py` locally. `--service-ports` keeps 8080 published; `--rm`
removes the one-off container on exit. (It still updates a little slower than a
native terminal — that's the container TTY, not a hang.)

Then open <http://localhost:8080> in your normal browser — you'll see the real UI.

Walkthrough, as a new user would:

1. Open **Settings** → paste your HF token → press Enter (saves to the server's `.env`).
2. Upload an image, fill in the prompts, hit **Run**.
3. The **first** inference downloads the model — slow (multi-GB), and part of the
   experience. Watch the progress bar / server logs.

Stop with `Ctrl-C`, then `docker compose down`.

## Cold vs. warm

Default is **cold**: no cache mounted, so each fresh container re-downloads the
weights — the honest new-user path. Do this at least once.

While iterating on the *harness itself*, re-downloading 15+ GB every run is
painful. Opt into a **warm** run by uncommenting the `volumes` block in
`docker-compose.yml` — but that's a *returning-user* run, not a new-user one.
Reset to cold with:

```bash
docker compose down && docker volume rm imgbox_hf-cache
```

(Volume name is `<project>_hf-cache`; the project defaults to the repo directory
name, `imgbox`.)

## Known rough edges to verify on your hardware

- **CUDA/cuDNN base image** (`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`) is a
  starting point. It must match your host driver, and `onnxruntime-gpu` expects a
  compatible cuDNN. If the identity/ORT steps fail, bump the base tag.
- **VRAM**: SD3 Medium plus the fine-tune step (8-bit Adam via bitsandbytes) needs
  a real chunk of GPU memory.
- **First-run latency** is dominated by the model download, not compute.
- Plain `docker run` equivalent of the GPU flag is `--gpus all`.
