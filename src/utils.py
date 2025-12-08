from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return a torch.device, honoring an optional preferred device string."""
    if preferred is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


DEVICE = get_device()


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
