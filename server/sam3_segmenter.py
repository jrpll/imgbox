import hashlib
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from device import DEVICE

# The sam3 package and its checkpoint live outside this repo.
SAM3_PKG = os.getenv("SAM3_PKG", "/home_pers/jripoll/flux2-sde-sampling")
SAM3_CKPT = os.getenv("SAM3_CKPT", "/home_pers/jripoll/flux2-sde-sampling/sam3/sam3.pt")


class Sam3Segmenter:
    """SAM3's interactive point predictor, with the image embedding cached across clicks."""

    def __init__(self):
        if SAM3_PKG not in sys.path:
            sys.path.insert(0, SAM3_PKG)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # The interactive predictor has no backbone of its own (build_tracker leaves it
        # None); it reads the detector's shared image features, so prompting goes through
        # Sam3Processor.set_image + model.predict_inst rather than predictor.set_image.
        self.model = build_sam3_image_model(
            checkpoint_path=SAM3_CKPT if os.path.exists(SAM3_CKPT) else None,
            device=DEVICE,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model, device=DEVICE)
        self._digest = None
        self._state = None

    def _image_state(self, image: Image.Image):
        """The backbone pass, cached so repeated prompts on one image only run the heads."""
        digest = hashlib.md5(image.tobytes()).hexdigest()
        if digest != self._digest:
            self._state = self.processor.set_image(image)
            self._digest = digest
        return self._state

    @torch.inference_mode()
    def mask(self, image: Image.Image, points, labels, additive=False) -> np.ndarray:
        """points: normalized (x, y) in [0, 1]. labels: 1 = select, 0 = exclude.

        additive: each point is segmented on its own and folded in click order — a select
        point adds its object to the mask so far, an exclude point removes its object.
        """
        if additive:
            out = np.zeros((image.height, image.width), dtype=bool)
            for p, l in zip(points, labels):
                m = self.mask(image, [p], [1])
                out = out | m if l else out & ~m
            return out
        state = self._image_state(image)
        scale = np.array(image.size, dtype=np.float32)
        masks, ious, _ = self.model.predict_inst(
            state,
            point_coords=np.array(points, dtype=np.float32) * scale,
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=True,
        )
        return masks[int(np.argmax(ious))].astype(bool)

    @torch.inference_mode()
    def mask_from_text(self, image: Image.Image, text: str) -> np.ndarray:
        """Union of every instance of `text` the grounding head finds — 'remove all X'.

        Empty when nothing clears the confidence threshold, which cuts nothing out.
        """
        state = self._image_state(image)
        # The vision backbone's FlashAttention-3 kernel returns bf16 whatever it is fed
        # (sam3/perflib/fa3.py), while every parameter is fp32 — so the grounding head's
        # LayerNorm sees bf16 activations against an fp32 weight and raises. SAM3's own
        # video predictor sidesteps this by running the model under bf16 autocast; the
        # point head needs no such thing, its decoder is fed through predict_inst.
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            state = self.processor.set_text_prompt(prompt=text, state=state)
        masks = state["masks"]
        if masks.shape[0] == 0:
            return np.zeros((image.height, image.width), dtype=bool)
        return masks.any(dim=0).squeeze(0).cpu().numpy()


def erase_mask(size, strokes) -> np.ndarray:
    """Eraser strokes rasterized: points normalized like the clicks, radius `r` a fraction of the width."""
    w, h = size
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        r = stroke["r"] * w
        pts = [(x * w, y * h) for x, y in stroke["pts"]]
        draw.line(pts, fill=255, width=round(2 * r))
        # PIL lines have flat ends and no joins; a disc at every point rounds them off.
        for x, y in pts:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.array(img) > 0


def cutout(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """The source image with the selected regions made transparent."""
    out = image.convert("RGB")
    out.putalpha(Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), "L"))
    return out
