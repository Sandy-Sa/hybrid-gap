"""Loaders for SFT predictions, LLM verdicts, gold labels, and class maps.

Extracted from the repo-root `evaluate.py` — these are the only five helpers
the eval harness needs. Underscore-prefixed names are kept as private aliases
so any external caller still importing them by the old name continues to work
for one release; new code should use the unprefixed names.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from coding.pipeline.io import load_predictions
from coding.utils.helpers import (
    _ID_CANDIDATES,
    cast_id_columns_to_str,
    get_id_columns,
    is_procedure_code,
)


def load_verified_codes(path: Path) -> tuple[str, dict]:
    """Accumulate verified-correct codes per note from a verification JSONL.

    Handles both one-line-per-note (run_verification) and one-line-per-
    (note, term) (run_verify_top_n) files: codes from every line are unioned
    into the per-note set.
    """
    acc: dict = {}
    id_col: str | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if id_col is None:
                id_col = next((c for c in _ID_CANDIDATES if c in r), None)
                if id_col is None:
                    raise ValueError(f"No known ID column found in {path.name}")
            content = r["content"]
            if isinstance(content, str):
                content = json.loads(content)
            acc.setdefault(str(r[id_col]), set()).update(
                v["code"] for v in content.get("verdicts", []) if v.get("correct")
            )
    return id_col, acc


def load_sft_codes(
    data_path: Path, threshold: float
) -> tuple[str, dict, pl.DataFrame]:
    """Load SFT predictions >= threshold and return (id_col, {eid: set[code]}, class_map)."""
    prediction = load_predictions(data_path)
    if prediction.height == 1:
        prediction = prediction.explode(pl.all())
    if "ids" in prediction.columns:
        prediction = prediction.rename({"ids": "event_id"})
    prediction = cast_id_columns_to_str(prediction)
    prediction = prediction.select(
        "event_id",
        pl.col("labels").cast(pl.List(pl.Int8)),
        "predictions",
    )
    class_map = pl.read_parquet(data_path / "class_map.parquet")

    sft_df = (
        prediction.with_columns(
            pl.int_ranges(pl.col("labels").list.len()).alias("value")
        )
        .explode(["labels", "predictions", "value"])
        .filter(pl.col("predictions") >= threshold)
        .join(class_map, on="value", how="left")
        .rename({"key": "code"})
    )
    id_cols = get_id_columns(sft_df)
    sft_grouped = sft_df.group_by(id_cols).agg(pl.col("code"))
    sft_lookup = {
        row[id_cols[0]]: set(row["code"]) for row in sft_grouped.iter_rows(named=True)
    }
    return id_cols[0], sft_lookup, class_map


def build_class_maps_from_parquet(
    class_map: pl.DataFrame, code_type: str = "ICD-9"
) -> tuple[dict, dict, dict]:
    """Build (combined, diag, proc) class maps from class_map.parquet.

    Uses the stable `value` column (set at training time) rather than
    re-deriving from labels with non-deterministic unique() ordering.
    Diag and proc sub-maps are re-indexed from 0 for one_hot_encode compatibility.
    """
    combined = dict(zip(class_map["key"].to_list(), class_map["value"].to_list()))

    diag_codes = [k for k in combined if not is_procedure_code(k, code_type)]
    proc_codes = [k for k in combined if is_procedure_code(k, code_type)]

    diag_codes.sort(key=lambda k: combined[k])
    proc_codes.sort(key=lambda k: combined[k])

    diag_map = {code: i for i, code in enumerate(diag_codes)}
    proc_map = {code: i for i, code in enumerate(proc_codes)}

    return combined, diag_map, proc_map


def build_ood_and_full_class_maps(
    labels: pl.DataFrame,
    label_config: dict,
    class_maps: tuple[dict, dict, dict],
) -> tuple[tuple[dict, dict, dict], tuple[dict, dict, dict]]:
    """Derive OOD and full (in-dist + OOD) class maps from labels.

    OOD codes are codes present in `labels` but absent from the in-dist
    `class_maps` (which were filtered to codes occurring >10 times in train).
    Diag/proc split comes from the label parquet's pre-split columns; each
    sub-map is re-indexed from 0 for `one_hot_encode` compatibility.
    """
    in_combined, in_diag, in_proc = class_maps

    label_diag_codes = set(
        labels[label_config["diag_column"]].explode().drop_nulls().to_list()
    )
    label_proc_codes = set(
        labels[label_config["proc_column"]].explode().drop_nulls().to_list()
    )

    ood_diag = sorted(label_diag_codes - set(in_diag))
    ood_proc = sorted(label_proc_codes - set(in_proc))
    ood_combined_codes = sorted(set(ood_diag) | set(ood_proc))

    ood_combined = {c: i for i, c in enumerate(ood_combined_codes)}
    ood_diag_map = {c: i for i, c in enumerate(ood_diag)}
    ood_proc_map = {c: i for i, c in enumerate(ood_proc)}

    full_combined_codes = sorted(set(in_combined) | set(ood_combined))
    full_diag_codes = sorted(set(in_diag) | set(ood_diag))
    full_proc_codes = sorted(set(in_proc) | set(ood_proc))

    full_combined = {c: i for i, c in enumerate(full_combined_codes)}
    full_diag = {c: i for i, c in enumerate(full_diag_codes)}
    full_proc = {c: i for i, c in enumerate(full_proc_codes)}

    return (
        (ood_combined, ood_diag_map, ood_proc_map),
        (full_combined, full_diag, full_proc),
    )


def load_labels(label_path: Path, label_config: dict) -> pl.DataFrame:
    """Load labels parquet and add a combined diag+proc `target` column."""
    labels = cast_id_columns_to_str(pl.read_parquet(label_path)).with_columns(
        pl.concat_list(
            [
                pl.col(label_config["diag_column"]).fill_null([]),
                pl.col(label_config["proc_column"]).fill_null([]),
            ]
        ).alias("target")
    )
    return labels


# Private aliases for callers still importing under the old underscore names.
_load_verified_codes = load_verified_codes
_load_sft_codes = load_sft_codes
_build_class_maps_from_parquet = build_class_maps_from_parquet
_build_ood_and_full_class_maps = build_ood_and_full_class_maps
_load_labels = load_labels
