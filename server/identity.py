import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _flatten_nested_pack(pack_name: str) -> None:
    """insightface's antelopev2 zip extracts into a nested subdir; flatten it."""
    root = Path(os.path.expanduser(f"~/.insightface/models/{pack_name}"))
    nested = root / pack_name
    if nested.is_dir() and any(nested.glob("*.onnx")) and not any(root.glob("*.onnx")):
        for f in nested.iterdir():
            shutil.move(str(f), str(root / f.name))
        nested.rmdir()


class IdentityModel:
    name = "antelopev2"
    embedding_dim = 512

    def __init__(self):
        from insightface.app import FaceAnalysis
        _flatten_nested_pack(self.name)
        if sys.platform == "darwin":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name=self.name,
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def embed(self, pil_image: Image.Image) -> dict | None:
        rgb = pil_image.convert("RGB")
        arr = np.array(rgb)[:, :, ::-1]  # BGR for insightface
        faces = self.app.get(arr)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)

        x1, y1, x2, y2 = best.bbox.tolist()
        w, h = rgb.size
        pad_x = (x2 - x1) * 0.25
        pad_y = (y2 - y1) * 0.25
        crop = rgb.crop((
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        ))

        return {
            "embedding": best.normed_embedding.astype(np.float32).tolist(),
            "bbox": [float(v) for v in best.bbox.tolist()],
            "det_score": float(best.det_score),
            "kps": best.kps.astype(np.float32).tolist(),
            "age": int(getattr(best, "age", -1)),
            "gender": int(getattr(best, "gender", -1)),
            "crop": crop,
            "width": w,
            "height": h,
        }
