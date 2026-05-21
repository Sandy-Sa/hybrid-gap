"""Path constants for the eval package, read from `params.yaml`.

Single-dataset by design — the active pipeline runs on the dataset configured
in `params.yaml` (`dataset_name`, `dataset_path`, `output_dir`, `code_type`).
All eval producers route through these constants so a `dataset_name` change in
`params.yaml` re-targets the whole stage.

`TAU_MACRO` / `TAU_MICRO` default to the values written into
`${dataset_path}optimal_thresholds.json` by the `evaluate_sft` DVC stage.
Override either or both by setting `eval.tau_macro` / `eval.tau_micro` in
`params.yaml` (for manual runs at non-optimal operating points). Resolution
is lazy: attribute access triggers the file read, so importing this module
before `evaluate_sft` has produced its output is still safe — failure only
surfaces if a caller actually consumes the thresholds.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_PARAMS = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())

DATASET_DIR = REPO_ROOT / _PARAMS["dataset_path"]
OUTPUT_DIR = REPO_ROOT / _PARAMS["output_dir"]
EVAL_DIR = OUTPUT_DIR / "eval"
LABELS_PATH = REPO_ROOT / _PARAMS["processed_data_path"]
FREQ_BANDS_PATH = REPO_ROOT / "data" / "freq_bands.json"
MDACE_LEAK_DIR = REPO_ROOT / "data" / "mdace_prompt_tuning"
THRESHOLDS_PATH = DATASET_DIR / "optimal_thresholds.json"

DATASET_NAME = _PARAMS["dataset_name"]
CODE_TYPE = _PARAMS["code_type"]


@functools.cache
def resolve_thresholds() -> tuple[float, float]:
    """Return ``(tau_macro, tau_micro)``.

    Resolution order, per-knob:

    1. ``params.yaml: eval.tau_macro`` / ``eval.tau_micro`` when set (non-null).
    2. ``${dataset_path}optimal_thresholds.json`` — keys ``macro`` / ``micro``,
       produced by the ``evaluate_sft`` DVC stage.

    Raises ``FileNotFoundError`` if a knob is needed from the file but the file
    is absent. Cached so the read happens once per process.
    """
    eval_block = _PARAMS.get("eval") or {}
    override_macro = eval_block.get("tau_macro")
    override_micro = eval_block.get("tau_micro")
    if override_macro is not None and override_micro is not None:
        return float(override_macro), float(override_micro)

    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"{THRESHOLDS_PATH} not present — run the `evaluate_sft` DVC stage "
            "first, or set `eval.tau_macro` and `eval.tau_micro` in params.yaml."
        )
    file_data = json.loads(THRESHOLDS_PATH.read_text())
    macro = override_macro if override_macro is not None else file_data["macro"]
    micro = override_micro if override_micro is not None else file_data["micro"]
    return float(macro), float(micro)


def __getattr__(name: str):
    """Lazy module attributes: `TAU_MACRO`, `TAU_MICRO`.

    Triggers `resolve_thresholds()` on first access so a missing
    `optimal_thresholds.json` only errors when something actually reads the
    thresholds (not at simple `import coding.eval.paths`).
    """
    if name == "TAU_MACRO":
        return resolve_thresholds()[0]
    if name == "TAU_MICRO":
        return resolve_thresholds()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
