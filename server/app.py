from contextlib import asynccontextmanager
import io
import json
import os
import tempfile

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from model_registry import ModelRegistry
from progress import tracker

registry = ModelRegistry()

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
    return {"status": "ready" if registry.current() else "loading"}


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
# Routes
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    text1: str = Form(""),
    text2: str = Form(""),
    num_train_steps: int = Form(100),
    num_inversion_steps: int = Form(50),
):
    contents = await image.read()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "img.jpg")[1])
    tmp.write(contents)
    tmp.close()

    pil_image = Image.open(tmp.name).resize((1024, 1024)).convert("RGB")
    os.unlink(tmp.name)

    neg_prompt = text1.replace(text2, "").strip(", ")

    print(f"generate: text1={text1!r}  text2={text2!r}  neg={neg_prompt!r}  train_steps={num_train_steps}  inv_steps={num_inversion_steps}")

    tracker.set("Loading", 0, 1)

    def _run():
        image_editor = registry.acquire('edit')
        image_editor.ft_invert(
            img=pil_image,
            prompt=text1,
            num_train_steps=num_train_steps,
            fine_tune=True,
            num_inversion_steps=num_inversion_steps,
        )
        return image_editor.edit(prompt=text2, neg_prompt=neg_prompt)

    try:
        edited = await run_in_threadpool(_run)
    finally:
        tracker.clear()

    buf = io.BytesIO()
    edited.save(buf, "JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/edit")
async def edit(
    slider: float = Form(0.9),
    text1: str = Form(""),
    text2: str = Form(""),
    neg_prompt: str = Form(""),
):
    if not neg_prompt:
        neg_prompt = text1.replace(text2, "").strip(", ")
    print(f"edit: slider={slider} neg={neg_prompt!r}")

    tracker.set("Loading", 0, 1)

    def _run():
        image_editor = registry.acquire('edit')
        num_inversion_steps = image_editor.num_inversion_steps
        num_skipped_steps = num_inversion_steps - int(slider * num_inversion_steps)
        return image_editor.edit(
            prompt=text2,
            neg_prompt=neg_prompt,
            num_skipped_steps=num_skipped_steps,
        )

    try:
        edited = await run_in_threadpool(_run)
    finally:
        tracker.clear()

    buf = io.BytesIO()
    edited.save(buf, "JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

@app.post("/flux2klein")
async def flux2klein(
    image: UploadFile | None = File(None),
    prompt: str = Form(""),
    num_inference_steps: int = Form(100),
    diffusion_coefficient: float = Form(3),
):
    pil_image = None
    if image is not None:
        contents = await image.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "img.jpg")[1])
        tmp.write(contents)
        tmp.close()
        pil_image = Image.open(tmp.name).convert("RGB")
        os.unlink(tmp.name)

    print(f"flux2klein: prompt={prompt!r}  steps={num_inference_steps}  diff_coef={diffusion_coefficient}")

    tracker.set("Generating", 0, num_inference_steps)

    def _run():
        generator = registry.acquire('flux2klein')
        def on_step(_pipe, step_index, _timestep, kwargs):
            tracker.set("Generating", step_index + 1, num_inference_steps)
            return kwargs
        return generator(
            image=pil_image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            diffusion_norm=diffusion_coefficient,
            callback_on_step_end=on_step,
        ).images[0]
    try:
        edited = await run_in_threadpool(_run)
    finally:
        tracker.clear()

    buf = io.BytesIO()
    edited.save(buf, "JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

@app.post("/remove-background")
async def remove_background(image: UploadFile = File(...)):
    contents = await image.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "img.jpg")[1])
    tmp.write(contents)
    tmp.close()
    pil_image = Image.open(tmp.name).convert("RGB")
    def _run():
        model = registry.acquire('remove-background')
        return model(pil_image)
    
    pil_image = await run_in_threadpool(_run)
    buf = io.BytesIO()
    pil_image.save(buf, "png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
