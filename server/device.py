import torch


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
# bf16 is correct on both CUDA and MPS. The MPS "gibberish" was caused by the fp16
# autocast in flux2klein (now gated to CUDA), not by bf16 — and fp32 here doubles the
# resident footprint and OOMs Apple's unified memory, so keep bf16.
DTYPE = torch.bfloat16


def empty_cache() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
