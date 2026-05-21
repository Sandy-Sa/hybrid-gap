"""Tests for ``coding.utils.reproducibility.set_global_seed``."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

import numpy as np  # noqa: E402

from coding.utils.reproducibility import DEFAULT_SEED, set_global_seed  # noqa: E402


def _sample() -> tuple[float, float]:
    return random.random(), float(np.random.random())


def test_seed_is_deterministic() -> None:
    set_global_seed(DEFAULT_SEED)
    first = _sample()

    set_global_seed(DEFAULT_SEED)
    second = _sample()

    assert first == second, "Seeding twice with the same value should yield the same draws"


def test_different_seeds_differ() -> None:
    set_global_seed(123)
    a = _sample()

    set_global_seed(456)
    b = _sample()

    assert a != b, "Different seeds should produce different draws"
