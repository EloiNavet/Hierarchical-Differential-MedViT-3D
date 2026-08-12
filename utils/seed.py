"""Reproducibility utilities: seeding and deterministic settings."""

import os
import random

import numpy as np
from monai.utils import set_determinism as monai_set_determinism

_MAX_UINT32 = 2**32


def normalize_seed(value) -> int | None:
    """Normalize SEED values from config/CLI to an int or None.

    Accepts integers, strings like 'none'/'false', or dict nodes with a 'value' key
    (to mirror the training script config format).
    """
    if value in (None, False):
        return None

    if isinstance(value, dict):
        if "value" in value:
            return normalize_seed(value["value"])
        raise ValueError(f"Invalid SEED configuration structure: {value!r}")

    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "none", "null", "false"}:
            return None

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SEED value {value!r}") from exc


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs and MONAI if available."""
    import torch

    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if monai_set_determinism is not None:
        monai_set_determinism(seed=seed)
