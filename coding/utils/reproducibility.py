"""Utilities for ensuring experiment reproducibility."""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np
import torch


DEFAULT_SEED: Final[int] = 314


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed all relevant random number generators.

    Parameters
    ----------
    seed:
        Seed value applied to Python's ``random`` module, NumPy, PyTorch (CPU and
        CUDA), and the ``PYTHONHASHSEED`` environment variable.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

