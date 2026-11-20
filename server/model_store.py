"""What the modes need on disk: whether it is there, and fetching it with progress.

The loaders in model_registry.py download on first use, silently. This is the same
set of files, fetched on demand from the settings panel instead.
"""
import os
import threading
import traceback
import zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, constants, snapshot_download, try_to_load_from_cache
from huggingface_hub.utils import filter_repo_objects

import sam3_segmenter

_PIPELINE = ["model_index.json", "scheduler/*", "tokenizer*/*", "transformer/*", "vae/*"]
_INSIGHTFACE = Path(os.path.expanduser("~/.insightface/models"))

# `allow` / `ignore` select what from_pretrained fetches for the loader's arguments, and no
# more (no fp16 duplicates, no single-file checkpoints). `files` are the weights among them:
# all of them on disk is what "downloaded" means. `gb` is only shown while that isn't so.
MODELS = {
    "sd3": {
        "label": "Stable Diffusion 3 Medium",
        "modes": ["edit"],
        "gb": 15.5,
        "repo": "stabilityai/stable-diffusion-3-medium-diffusers",
        "allow": _PIPELINE + ["text_encoder*/*"],
        "ignore": ["*fp16*"],
        "files": [
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "text_encoder_3/model-00001-of-00002.safetensors",
            "text_encoder_3/model-00002-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    },
    # Its text encoder also serves the fast mode, whose own repo is fetched without one.
    "flux2-klein-base": {
        "label": "FLUX.2 [klein] base 4B",
        "modes": ["flux2klein", "flux2klein-fast", "sdedit"],
        "gb": 16.0,
        "repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "allow": _PIPELINE + ["text_encoder/*"],
        "files": [
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    },
    "flux2-klein": {
        "label": "FLUX.2 [klein] 4B",
        "modes": ["flux2klein-fast"],
        "gb": 7.9,
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "allow": _PIPELINE,
        "files": [
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    },
    "rmbg": {
        "label": "RMBG-2.0",
        "modes": ["remove-background"],
        "gb": 0.9,
        "repo": "briaai/RMBG-2.0",
        "allow": ["config.json", "*.py", "model.safetensors"],
        "files": ["model.safetensors"],
    },
    # A checkpoint at SAM3_CKPT takes precedence over the hub's, as in sam3_segmenter.
    "sam3": {
        "label": "SAM 3",
        "modes": ["sam3"],
        "gb": 3.5,
        "repo": "facebook/sam3",
        "allow": ["config.json", "sam3.pt"],
        "files": ["sam3.pt"],
        "local": sam3_segmenter.SAM3_CKPT,
    },
    # Not on the hub: insightface's own release (its storage.BASE_REPO_URL), unzipped where it looks.
    "antelopev2": {
        "label": "antelopev2",
        "modes": ["identity"],
        "gb": 0.4,
        "zip": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        "files": ["glintr100.onnx", "scrfd_10g_bnkps.onnx"],
    },
}

# Downloads in flight or failed, by model name: {"done", "total", "plan", "error"}.
_state: dict[str, dict] = {}


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _on_disk(m: dict) -> tuple[bool, int]:
    """Whether the model's weights are all there, and how much they weigh."""
    if os.path.exists(m.get("local", "")):
        return True, os.path.getsize(m["local"])
    if "zip" in m:
        found = [p for f in m["files"] for p in (_INSIGHTFACE / "antelopev2").rglob(f)]
    else:
        found = [Path(p) for p in (try_to_load_from_cache(m["repo"], f) for f in m["files"]) if isinstance(p, str)]
    return len(found) >= len(m["files"]), sum(_size(p) for p in found)


def _done(m: dict, st: dict) -> int:
    """Bytes fetched so far: the planned files already in the cache, plus the partial ones."""
    if "zip" in m:
        return st["done"]
    blobs = Path(constants.HF_HUB_CACHE) / ("models--" + m["repo"].replace("/", "--")) / "blobs"
    partial = sum(_size(p) for p in blobs.glob("*.incomplete"))
    return partial + sum(size for f, size in st["plan"].items() if isinstance(try_to_load_from_cache(m["repo"], f), str))


def listing() -> list[dict]:
    rows = []
    for name, m in MODELS.items():
        ready, size = _on_disk(m)
        st = _state.get(name)
        downloading = st is not None and st["error"] is None
        rows.append({
            "name": name,
            "label": m["label"],
            "modes": m["modes"],
            "status": "downloading" if downloading else "ready" if ready else "error" if st else "missing",
            "size": size if ready else int(m["gb"] * 1e9),
            "done": _done(m, st) if downloading else 0,
            "total": st["total"] if downloading else 0,
            "error": st["error"] if st else None,
        })
    return rows


def download(name: str) -> None:
    """Start fetching a model in the background; listing() reports how far it is."""
    st = _state.get(name)
    if st is not None and st["error"] is None:
        return
    _state[name] = {"done": 0, "total": 0, "plan": {}, "error": None}
    threading.Thread(target=_fetch, args=(name,), daemon=True).start()


def _fetch(name: str) -> None:
    m, st = MODELS[name], _state[name]
    try:
        if "zip" in m:
            _fetch_zip(m, st)
        else:
            token = os.getenv("HUGGING_FACE_TOKEN")
            info = HfApi().model_info(m["repo"], files_metadata=True, token=token)
            sizes = {s.rfilename: s.size or 0 for s in info.siblings}
            picked = filter_repo_objects(sizes, allow_patterns=m["allow"], ignore_patterns=m.get("ignore"))
            st["plan"] = {f: sizes[f] for f in picked}
            st["total"] = sum(st["plan"].values())
            snapshot_download(m["repo"], allow_patterns=m["allow"], ignore_patterns=m.get("ignore"), token=token)
        del _state[name]
    except Exception as e:
        traceback.print_exc()
        st["error"] = f"{type(e).__name__}: {e}"


def _fetch_zip(m: dict, st: dict) -> None:
    _INSIGHTFACE.mkdir(parents=True, exist_ok=True)
    zip_path = _INSIGHTFACE / "antelopev2.zip"
    with requests.get(m["zip"], stream=True, timeout=30) as r:
        r.raise_for_status()
        st["total"] = int(r.headers.get("Content-Length", 0))
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                st["done"] += len(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_INSIGHTFACE / "antelopev2")
