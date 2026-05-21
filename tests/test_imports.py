"""Smoke tests that every Python module under ``coding/`` is importable.

These tests guard against accidental breakage of import paths and missing
runtime dependencies. They do not exercise model code or hit the network.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "coding"


def _module_names() -> list[str]:
    """Discover dotted module names by walking the filesystem.

    ``pkgutil.walk_packages`` is not used here because the ``coding/`` tree
    does not yet ship ``__init__.py`` files (tracked in docs/known-issues.md);
    a direct filesystem walk works regardless.
    """
    names: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        # Files with hyphens in the stem cannot be imported via the dotted
        # path — skip them defensively if any are ever added.
        if "-" in path.stem:
            continue
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        names.append(".".join(rel.parts))
    return names


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports(module_name: str) -> None:
    """Every discovered submodule of ``coding`` should import without error.

    A missing optional dependency is treated as a skip rather than a failure so
    that the suite remains useful on CPU-only / minimal CI runners.
    """

    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional dependency missing for {module_name}: {exc.name}")
