import base64
from contextlib import asynccontextmanager
import datetime
from functools import lru_cache
import io
import json
import os
import tempfile

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm


def _save_with_exif(image: Image.Image, fmt: str, **save_kwargs) -> io.BytesIO:
    """Save image with imgbox EXIF: Software + DateTime + DateTimeOriginal=now."""
    now = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    exif = Image.Exif()
    exif[0x0131] = "imgbox"   # Software (0th IFD)
    exif[0x0132] = now        # DateTime (0th IFD)
    exif.get_ifd(0x8769)[0x9003] = now  # DateTimeOriginal (Exif sub-IFD)
    buf = io.BytesIO()
    image.save(buf, fmt, exif=exif.tobytes(), **save_kwargs)
    buf.seek(0)
    return buf

register_heif_opener()

from device import DEVICE, DTYPE, empty_cache
from model_registry import ModelRegistry
from progress import tracker
import identity_store
import jobs
import model_store
import sam3_segmenter

registry = ModelRegistry()


def _fit_1mp(image: Image.Image) -> Image.Image:
    """~1 MP at the source aspect ratio, sides in multiples of 16 (SD3: VAE /8, patch 2)."""
    scale = (1024 * 1024 / (image.width * image.height)) ** 0.5
    return image.resize(tuple(max(16, round(d * scale / 16) * 16) for d in image.size))


def _image_response(image: Image.Image, fmt: str) -> Response:
    """Whole bytes rather than a stream: a finished job's result can be fetched more than once."""
    return Response(_save_with_exif(image, fmt).getvalue(), media_type=f"image/{fmt.lower()}")


@lru_cache(maxsize=8)
def _flux_embed(prompt: str) -> torch.Tensor:
    """Cached, so a prompt seen before doesn't swap the text encoder back onto the GPU."""
    p_encoder = registry.acquire('flux2klein_pe')
    with torch.no_grad():
        return p_encoder.encode_prompt(prompt)[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), ".env")

def _load_env_file():
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

_load_env_file()

@app.post("/config")
async def set_config(hf_token: str = Form(...)):
    os.environ["HUGGING_FACE_TOKEN"] = hf_token
    with open(_CONFIG_FILE, "w") as f:
        f.write(f"HUGGING_FACE_TOKEN={hf_token}\n")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    if DEVICE == "cpu":
        raise HTTPException(status_code=503, detail={"status": "no_accelerator", "device": "cpu"})
    return {
        "status": "ready" if registry.current() else "loading",
        "device": DEVICE,
        "model": registry.current(),
        "trained": hasattr(registry.current_model(), "zts_ref"),
    }


# ---------------------------------------------------------------------------
# Models on disk (the settings panel's list)
# ---------------------------------------------------------------------------
@app.get("/models")
async def models():
    return model_store.listing()


@app.post("/models/{name}/download")
async def models_download(name: str):
    if name not in model_store.MODELS:
        raise HTTPException(status_code=404, detail="unknown model")
    model_store.download(name)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Progress (SSE)
# ---------------------------------------------------------------------------
@app.get("/progress")
async def progress():
    async def stream():
        queue = tracker.subscribe()
        try:
            while True:
                snap = await queue.get()
                payload = {
                    "message": snap.message,
                    "progress": snap.current / snap.total,
                    "remaining": snap.remaining,
                    "job": jobs.current(),
                }
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            tracker.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Jobs — every ML route below queues one (jobs.py). With ?job=1 it answers with the
# job's id instead of waiting for the result, which is then fetched here.
# ---------------------------------------------------------------------------
@app.get("/jobs/{id_}/result")
async def job_result(id_: str):
    """Waits for the job, then answers as its route would have: 409 if cancelled, 500 if it failed."""
    return await jobs.get(id_).wait()


@app.delete("/jobs/{id_}")
async def job_cancel(id_: str):
    jobs.cancel(id_)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    text1: str = Form(""),
    text2: str = Form(""),
    num_train_steps: int = Form(100),
    num_inversion_steps: int = Form(50),
    cfg: bool = Form(True),
    job: bool = False,
):
    contents = await image.read()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "img.jpg")[1])
    tmp.write(contents)
    tmp.close()

    upload = ImageOps.exif_transpose(Image.open(tmp.name))
    pil_image = _fit_1mp(upload.convert("RGB"))
    # Transparency marks the area to edit: a SAM3 or background-remover cutout sent on from
    # the result card still has the whole picture in its RGB. Everything opaque is kept.
    alpha = upload.convert("RGBA").getchannel("A") if upload.has_transparency_data else None
    if alpha is not None and alpha.getextrema()[0] < 255:
        alpha = alpha.resize((pil_image.width // 8, pil_image.height // 8))
    else:
        alpha = None
    os.unlink(tmp.name)

    neg_prompt = text1.replace(text2, "").strip(", ")

    print(f"generate: text1={text1!r}  text2={text2!r}  neg={neg_prompt!r}  train_steps={num_train_steps}  inv_steps={num_inversion_steps}  cfg={cfg}  masked={alpha is not None}")

    # cfg off: cond passes only, in the inversion and in the generation alike.
    guidance = 7 if cfg else 1

    def _run():
        image_editor = registry.acquire('edit')
        image_editor.ft_invert(
            img=pil_image,
            prompt=text1,
            num_train_steps=num_train_steps,
            fine_tune=True,
            num_inversion_steps=num_inversion_steps,
            cfg=guidance,
            mask=None if alpha is None else 1 - to_tensor(alpha).to(DEVICE, DTYPE),
        )
        return _image_response(image_editor.edit(prompt=text2, neg_prompt=neg_prompt, cfg=guidance), "JPEG")

    return await jobs.respond(_run, job)


@app.post("/edit")
async def edit(
    slider: float = Form(0.9),
    text1: str = Form(""),
    text2: str = Form(""),
    neg_prompt: str = Form(""),
    cfg: bool = Form(True),
    job: bool = False,
):
    if not neg_prompt:
        neg_prompt = text1.replace(text2, "").strip(", ")
    print(f"edit: slider={slider} neg={neg_prompt!r} cfg={cfg}")

    def _run():
        image_editor = registry.acquire('edit')
        num_inversion_steps = image_editor.num_inversion_steps
        num_skipped_steps = num_inversion_steps - int(slider * num_inversion_steps)
        edited = image_editor.edit(
            prompt=text2,
            neg_prompt=neg_prompt,
            num_skipped_steps=num_skipped_steps,
            cfg=7 if cfg else 1,
        )
        return _image_response(edited, "JPEG")

    return await jobs.respond(_run, job)

@app.post("/flux2klein")
async def flux2klein(
    images: list[UploadFile] | None = File(None),
    prompt: str = Form(""),
    num_inference_steps: int = Form(50),
    num_images_per_prompt: int = Form(1),
    width: int | None = Form(None),
    height: int | None = Form(None),
    seed: int | None = Form(None),
    job: bool = False,
):
    pil_images = []
    if images:
        for upload in images:
            contents = await upload.read()
            pil_images.append(Image.open(io.BytesIO(contents)).convert("RGB"))
    pil_image = pil_images if pil_images else None

    print(f"flux2klein: prompt={prompt!r}  images={len(pil_images)}  steps={num_inference_steps}  n={num_images_per_prompt}  size={width}x{height}")

    generator = None if seed is None else torch.Generator().manual_seed(seed)

    def _run():
        prompt_embeds = _flux_embed(prompt)
        negative_prompt_embeds = _flux_embed("")
        empty_cache()
        pipe = registry.acquire('flux2klein')
        t = tqdm(total=num_inference_steps, desc="Generating")
        def on_step(_pipe, step_index, _timestep, kwargs):
            t.update(1)
            tracker.set_from_tqdm(t)
            return kwargs
        try:
            edited = pipe(
                image=pil_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                diffusion_norm=0,
                num_images_per_prompt=num_images_per_prompt,
                callback_on_step_end=on_step,
                width=width,
                height=height,
                generator=generator,
            ).images
        finally:
            t.close()
        return {"images": [
            base64.b64encode(_save_with_exif(img, "JPEG").getvalue()).decode()
            for img in edited
        ]}

    return await jobs.respond(_run, job)

@app.post("/flux2klein-fast")
async def flux2klein_fast(
    images: list[UploadFile] | None = File(None),
    prompt: str = Form(""),
    num_images_per_prompt: int = Form(1),
    width: int | None = Form(None),
    height: int | None = Form(None),
    seed: int | None = Form(None),
    job: bool = False,
):
    pil_images = []
    if images:
        for upload in images:
            contents = await upload.read()
            pil_images.append(Image.open(io.BytesIO(contents)).convert("RGB"))
    pil_image = pil_images if pil_images else None

    num_inference_steps = 4

    print(f"flux2klein-fast: prompt={prompt!r}  images={len(pil_images)}  steps={num_inference_steps}  n={num_images_per_prompt}  size={width}x{height}")

    generator = None if seed is None else torch.Generator().manual_seed(seed)

    def _run():
        prompt_embeds = _flux_embed(prompt)
        negative_prompt_embeds = _flux_embed("")
        empty_cache()
        pipe = registry.acquire('flux2klein-fast')
        t = tqdm(total=num_inference_steps, desc="Generating")
        def on_step(_pipe, step_index, _timestep, kwargs):
            t.update(1)
            tracker.set_from_tqdm(t)
            return kwargs
        try:
            edited = pipe(
                image=pil_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0,
                num_images_per_prompt=num_images_per_prompt,
                callback_on_step_end=on_step,
                width=width,
                height=height,
                generator=generator,
            ).images
        finally:
            t.close()
        return {"images": [
            base64.b64encode(_save_with_exif(img, "JPEG").getvalue()).decode()
            for img in edited
        ]}

    return await jobs.respond(_run, job)

@app.post("/sdedit")
async def sdedit(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    strength: float = Form(0.6, gt=0, le=1),
    num_inference_steps: int = Form(50),
    seed: int | None = Form(None),
    job: bool = False,
):
    """SDEdit on FLUX.2 klein base: `image` (the photo, painted on) is noised to `strength`,
    then denoised from there. `strength` is the sampler's VP time, i.e. SDEdit's t0."""
    pil_image = _fit_1mp(Image.open(io.BytesIO(await image.read())).convert("RGB"))
    # The share of a full schedule that lies below `strength`. The sampler needs two steps to know its step size.
    steps = max(2, round(num_inference_steps * strength))

    print(f"sdedit: prompt={prompt!r}  strength={strength}  steps={steps}  size={pil_image.width}x{pil_image.height}")

    generator = None if seed is None else torch.Generator().manual_seed(seed)

    def _run():
        prompt_embeds = _flux_embed(prompt)
        negative_prompt_embeds = _flux_embed("")
        empty_cache()
        pipe = registry.acquire('flux2klein')
        t = tqdm(total=steps, desc="Generating")
        def on_step(_pipe, step_index, _timestep, kwargs):
            t.update(1)
            tracker.set_from_tqdm(t)
            return kwargs
        try:
            edited = pipe(
                init_image=pil_image,
                strength=strength,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=steps,
                # Without a prompt the guided and the unguided pass are the same one: skip CFG.
                guidance_scale=4.0 if prompt.strip() else 1.0,
                diffusion_norm=0,
                callback_on_step_end=on_step,
                width=pil_image.width,
                height=pil_image.height,
                generator=generator,
            ).images[0]
        finally:
            t.close()
        return _image_response(edited, "JPEG")

    return await jobs.respond(_run, job)

@app.post("/identity")
async def identity(images: list[UploadFile] = File(...), caption: str = Form(""), job: bool = False):
    # Read now: the uploads are closed along with the request, and a detached job outlives it.
    uploads = [(upload.filename or "img", await upload.read()) for upload in images]

    def _run():
        processed = 0
        faces_found = 0
        skipped = 0
        ids: list[str] = []
        model = registry.acquire("identity")
        for filename, contents in uploads:
            try:
                pil = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception:
                skipped += 1
                processed += 1
                tracker.set("Embedding", processed, len(uploads))
                continue
            results = model.embed(pil)
            if not results:
                skipped += 1
            for face_index, result in enumerate(results):
                id_, _was_new = identity_store.insert(
                    image_bytes=contents,
                    source_filename=filename,
                    model_name=model.name,
                    embed_result=result,
                    caption=caption,
                    face_index=face_index,
                )
                ids.append(id_)
                faces_found += 1
            processed += 1
            tracker.set("Embedding", processed, len(uploads))
        return {
            "processed": processed,
            "faces_found": faces_found,
            "skipped": skipped,
            "ids": ids,
        }

    return await jobs.respond(_run, job)


@app.get("/identity/list")
async def identity_list():
    return identity_store.list_all()


@app.get("/identity/crop/{id_}")
async def identity_crop(id_: str):
    p = identity_store.crop_path(id_)
    if not p:
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/identity/original/{id_}")
async def identity_original(id_: str):
    p = identity_store.original_path(id_)
    if not p:
        raise HTTPException(status_code=404)
    return FileResponse(p)


@app.delete("/identity/{id_}")
async def identity_delete(id_: str):
    if not identity_store.delete(id_):
        raise HTTPException(status_code=404)
    return {"ok": True}


@app.get("/identity/match/{id_}")
async def identity_match(id_: str, k: int = 2):
    return identity_store.find_similar(id_, k)


@app.post("/remove-background")
async def remove_background(image: UploadFile = File(...), job: bool = False):
    contents = await image.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "img.jpg")[1])
    tmp.write(contents)
    tmp.close()
    pil_image = Image.open(tmp.name).convert("RGB")
    def _run():
        model = registry.acquire('remove-background')
        return _image_response(model(pil_image), "PNG")

    return await jobs.respond(_run, job)

@app.post("/load/{name}")
async def load(name: str):
    """The Load model button of modes that load explicitly (see `model` in the mode config)."""
    def _run():
        registry.acquire(name)
        return {"ok": True}

    return await jobs.respond(_run)

@app.post("/sam3")
async def sam3(
    image: UploadFile = File(...),
    points: str = Form("[]"),
    labels: str = Form("[]"),
    text: str = Form(""),
    preserve: bool = Form(False),
    additive: bool = Form(False),
    erase: str = Form("[]"),
    job: bool = False,
):
    """Points come from clicks (one call per click, no Run step); text comes from Run.

    They are separate SAM3 heads, so a request carries one or the other, not both.
    """
    contents = await image.read()
    pts, lbls = json.loads(points), json.loads(labels)
    if len(pts) != len(lbls):
        raise HTTPException(status_code=400, detail="points and labels must be the same length")
    if not pts and not text.strip():
        raise HTTPException(status_code=400, detail="need either points or a text prompt")
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    def _run():
        segmenter = registry.acquire('sam3')
        if pts:
            mask = segmenter.mask(pil_image, pts, lbls, additive)
        else:
            mask = segmenter.mask_from_text(pil_image, text.strip())
        # Eraser strokes go last: whatever they cover is removed, whichever way preserve points.
        removed = (~mask if preserve else mask) | sam3_segmenter.erase_mask(pil_image.size, json.loads(erase))
        return _image_response(sam3_segmenter.cutout(pil_image, removed), "PNG")

    return await jobs.respond(_run, job)


# ---------------------------------------------------------------------------
# Frontend (built SPA) — mounted last so API routes take precedence
# ---------------------------------------------------------------------------
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
