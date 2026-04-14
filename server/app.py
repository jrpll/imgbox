from contextlib import asynccontextmanager
import io
import os
import tempfile

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline

from finedits import FINEdits


# ---------------------------------------------------------------------------
# Globals — populated during startup
# ---------------------------------------------------------------------------
pipe = None
image_editor = None
_model_status = "loading"  # "loading" | "ready" | "no_cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, image_editor, _model_status

    if not torch.cuda.is_available():
        _model_status = "no_cuda"
        yield
        return

    token = os.getenv("HUGGING_FACE_TOKEN")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16,
        token=token,
    ).to("cuda")
    image_editor = FINEdits(pipe)
    _model_status = "ready"

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": _model_status}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    text1: str = Form(""),
    text2: str = Form(""),
):
    if _model_status != "ready":
        raise HTTPException(status_code=503, detail=_model_status)

    contents = await image.read()

    # Save to temp file (keeps a path available if needed by PIL)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "img.jpg")[1])
    tmp.write(contents)
    tmp.close()

    pil_image = Image.open(tmp.name).resize((1024, 1024)).convert("RGB")
    os.unlink(tmp.name)

    neg_prompt = text1.replace(text2, "").strip(", ")

    print(f"generate: text1={text1!r}  text2={text2!r}  neg={neg_prompt!r}")

    def _run():
        image_editor.ft_invert(
            img=pil_image,
            prompt=text1,
            num_train_steps=100,
            fine_tune=True,
            num_inversion_steps=50,
        )
        return image_editor.edit(prompt=text2, neg_prompt=neg_prompt)

    edited = await run_in_threadpool(_run)

    buf = io.BytesIO()
    edited.save(buf, "JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/edit")
async def edit(
    slider: float = Form(0.9),
    text1: str = Form(""),
    text2: str = Form(""),
):
    if _model_status != "ready":
        raise HTTPException(status_code=503, detail=_model_status)

    num_inversion_steps = image_editor.num_inversion_steps
    num_skipped_steps = num_inversion_steps - int(slider * num_inversion_steps)
    neg_prompt = text1.replace(text2, "").strip(", ")

    print(f"edit: slider={slider}  skipped={num_skipped_steps}  neg={neg_prompt!r}")

    def _run():
        return image_editor.edit(
            prompt=text2,
            neg_prompt=neg_prompt,
            num_skipped_steps=num_skipped_steps,
        )

    edited = await run_in_threadpool(_run)

    buf = io.BytesIO()
    edited.save(buf, "JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
