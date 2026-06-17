import torch


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
# bfloat16 is native on NVIDIA but unreliable on Apple MPS (incomplete kernels →
# CPU fallback + garbage output), so only use it on CUDA. float32 has full MPS coverage.
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def empty_cache() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
