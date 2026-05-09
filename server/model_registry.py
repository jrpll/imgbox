# server/model_registry.py
import gc, threading
from typing import Any
import torch
import os
from diffusers import StableDiffusion3Pipeline
from transformers import AutoModelForImageSegmentation

from finedits import FINEdits
from background_remover import BackgroundRemover

TOKEN = os.getenv("HUGGING_FACE_TOKEN")

class ModelRegistry:
    def __init__(self):
        self._loaders = {
            'edit': self._load_edit,
            'remove-background': self._load_remove_background,
        }
        self._current_name: str | None = None
        self._current_model: Any | None = None
        self._lock = threading.Lock()

    def acquire(self, name: str) -> Any:
        with self._lock:
            if self._current_name == name:
                return self._current_model
            self._unload()
            self._current_model = self._loaders[name]()
            self._current_name = name
            return self._current_model

    def current(self) -> str | None:
        return self._current_name

    def _unload(self) -> None:
        if self._current_model is None:
            return
        self._current_model = None
        self._current_name = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load_edit(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.bfloat16,
            token=TOKEN
        )
        pipe.enable_model_cpu_offload()
        image_editor = FINEdits(pipe)
        return image_editor

    def _load_remove_background(self):
        model = AutoModelForImageSegmentation.from_pretrained(
            'briaai/RMBG-2.0', 
            trust_remote_code=True,
            token=TOKEN
        ).eval().to("cuda")
        background_remover = BackgroundRemover(model)
        return background_remover
