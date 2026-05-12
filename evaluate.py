import json
import polars as pl
import re
from pathlib import Path
from jsonargparse import CLI
from coding.utils.helpers import (
    cast_id_columns_to_str,
    one_hot_encode,
    map_class,
    get_id_columns,
    is_procedure_code,
    _ID_CANDIDATES,
)
from coding.pipeline.io import load_predictions
import torch
from torchmetrics.functional.classification import (
    multilabel_f1_score,
    multilabel_precision,
    multilabel_recall,
)


# --- Polars expressions ---


def top_k_codes(col: str, k: int) -> pl.Expr:
    """Slice each inner list to top-k, then flatten and deduplicate."""
    return (
        pl.col(col)
        .list.eval(pl.element().list.head(k))
        .list.eval(pl.element().explode())
        .list.unique()
    )


def flatten_codes(col: str) -> pl.Expr:
    """Flatten nested code lists and deduplicate."""
    return pl.col(col).list.eval(pl.element().explode()).list.unique()


# --- Tensor / metrics ---


def codes_to_tensor(code_lists: list, class_map: dict) -> torch.Tensor:
    """Convert lists of codes to a one-hot encoded tensor.

    Codes not present in class_map are silently ignored — they fall outside
    the model's training vocabulary and cannot be predicted or evaluated.
    """
    num_classes = len(class_map)
    return torch.tensor(
        [
            (
                [0] * num_classes
                if codes is None
                else one_hot_encode([c for c in codes if c in class_map], class_map)
            )
            for codes in code_lists
        ]
    )


def compute_metrics(
    pred_tensor: torch.Tensor, target_tensor: torch.Tensor, num_labels: int
) -> dict:
    """Compute precision, recall, and F1 (micro + macro) for multilabel classification."""
    return {
        "precision_micro": multilabel_precision(
            pred_tensor, target_tensor, num_labels=num_labels, average="micro"
        ).item(),
        "recall_micro": multilabel_recall(
            pred_tensor, target_tensor, num_labels=num_labels, average="micro"
        ).item(),
        "f1_micro": multilabel_f1_score(
            pred_tensor, target_tensor, num_labels=num_labels, average="micro"
        ).item(),
        "precision_macro": multilabel_precision(
            pred_tensor, target_tensor, num_labels=num_labels, average="macro"
        ).item(),
        "recall_macro": multilabel_recall(
            pred_tensor, target_tensor, num_labels=num_labels, average="macro"
        ).item(),
        "f1_macro": multilabel_f1_score(
            pred_tensor, target_tensor, num_labels=num_labels, average="macro"
        ).item(),
    }


def print_metrics(label: str, metrics: dict):
    """Print micro and macro metrics with a label prefix."""
    print(
        f"{label} micro: P={metrics['precision_micro']:.4f}, R={metrics['recall_micro']:.4f}, F1={metrics['f1_micro']:.4f}"
    )
    print(
        f"{label} macro: P={metrics['precision_macro']:.4f}, R={metrics['recall_macro']:.4f}, F1={metrics['f1_macro']:.4f}"
    )


# --- Stats ---


def _length_stats(lengths: list) -> dict:
    s = pl.Series("x", lengths, dtype=pl.Float64)
    return {"mean": s.mean(), "std": s.std()}


def _token_stats(values: list) -> dict:
    s = pl.Series("x", [v for v in values if v is not None], dtype=pl.Float64)
    return {
        "mean": s.mean(),
        "std": s.std(),
        "median": s.median(),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
    }


# --- I/O ---


def _load_pred(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return cast_id_columns_to_str(pl.read_parquet(path))
    records = []
    id_cols: list[str] | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if id_cols is None:
                id_cols = [c for c in _ID_CANDIDATES if c in r]
            content = r.get("content", {})
            record = {c: r[c] for c in id_cols}
            record.update(
                {
                    "diag_codes": [content.get("diagnosis_codes") or []],
                    "proc_codes": [content.get("procedure_codes") or []],
                    "completion_tokens": r.get("completion_tokens"),
                    "total_tokens": r.get("total_tokens"),
                }
            )
            records.append(record)
    return cast_id_columns_to_str(pl.DataFrame(records))


def _write_metrics(path: Path, data: dict) -> None:
    """Write metrics as a flat JSON file (DVC-compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_verified_codes(path: Path) -> tuple[str, dict]:
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


def _load_sft_codes(
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


# --- Shared pipeline helpers ---


def _build_class_maps_from_parquet(
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


def _build_ood_and_full_class_maps(
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


def _load_labels(label_path: Path, label_config: dict) -> pl.DataFrame:
    """Load labels parquet, add combined target column, and build class maps."""
    labels = cast_id_columns_to_str(pl.read_parquet(label_path)).with_columns(
        pl.concat_list(
            [
                pl.col(label_config["diag_column"]).fill_null([]),
                pl.col(label_config["proc_column"]).fill_null([]),
            ]
        ).alias("target")
    )

    return labels


def _build_target_tensors(
    df: pl.DataFrame, label_config: dict, class_maps: tuple
) -> tuple:
    """Build (combined, diag, proc) target tensors from a joined dataframe."""
    class_map, diag_class_map, proc_class_map = class_maps
    return (
        codes_to_tensor(df["target"].to_list(), class_map),
        codes_to_tensor(df[label_config["diag_column"]].to_list(), diag_class_map),
        codes_to_tensor(df[label_config["proc_column"]].to_list(), proc_class_map),
    )


def _eval_predictions(
    label: str,
    pred_codes: list,
    pred_diag: list,
    pred_proc: list,
    target_tensors: tuple,
    class_maps: tuple,
) -> tuple[dict, dict, dict, dict, dict]:
    """Filter codes, compute metrics, print results.

    Returns (combined, diag, proc, raw_stats, filtered_stats).
    """
    class_map, diag_class_map, proc_class_map = class_maps
    target_tensor, target_diag_tensor, target_proc_tensor = target_tensors

    pred_filtered = [[c for c in codes if c in class_map] for codes in pred_codes]
    pred_diag_filtered = [
        [c for c in codes if c in diag_class_map] for codes in pred_diag
    ]
    pred_proc_filtered = [
        [c for c in codes if c in proc_class_map] for codes in pred_proc
    ]

    raw_stats = _length_stats([len(c) for c in pred_codes])
    filtered_stats = _length_stats([len(c) for c in pred_filtered])
    print(
        f"{label} predicted codes (raw): mean={raw_stats['mean']:.6f}, std={raw_stats['std']:.6f}"
    )
    print(
        f"{label} predicted codes (filtered): mean={filtered_stats['mean']:.6f}, std={filtered_stats['std']:.6f}"
    )

    combined = compute_metrics(
        codes_to_tensor(pred_filtered, class_map), target_tensor, len(class_map)
    )
    diag = compute_metrics(
        codes_to_tensor(pred_diag_filtered, diag_class_map),
        target_diag_tensor,
        len(diag_class_map),
    )
    proc = compute_metrics(
        codes_to_tensor(pred_proc_filtered, proc_class_map),
        target_proc_tensor,
        len(proc_class_map),
    )

    print_metrics(label, combined)
    print_metrics(f"{label} Diagnoses", diag)
    print_metrics(f"{label} Procedures", proc)

    return combined, diag, proc, raw_stats, filtered_stats


# --- CLI entry points ---


def run_evaluation(
    pred_path: Path,
    class_map_path: Path,
    label_path: Path,
    label_config: dict,
    code_type: str = "ICD-9",
):
    pred = _load_pred(pred_path)
    is_searched = "diag_scores" in pred.columns
    print(f"Loaded predictions: {len(pred)} records")

    labels = _load_labels(label_path, label_config)
    print(f"Loaded labels: {len(labels)} records")

    class_maps = _build_class_maps_from_parquet(
        pl.read_parquet(class_map_path), code_type=code_type
    )
    id_cols = get_id_columns(pred)
    df = labels.join(pred, on=id_cols)
    print(f"Merged dataset: {len(df)} records after join")

    output_path = pred_path.parent / f"{pred_path.stem}_metrics.json"

    target_stats = _length_stats([len(t) for t in df["target"].to_list()])
    completion_token_stats = _token_stats(df["completion_tokens"].to_list())
    total_token_stats = _token_stats(df["total_tokens"].to_list())
    print(
        f"Target codes: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}"
    )
    print(
        f"Completion tokens: mean={completion_token_stats['mean']:.6f}, std={completion_token_stats['std']:.6f}, "
        f"median={completion_token_stats['median']:.6f}, p25={completion_token_stats['p25']:.6f}, p75={completion_token_stats['p75']:.6f}"
    )
    print(
        f"Total tokens: mean={total_token_stats['mean']:.6f}, std={total_token_stats['std']:.6f}, "
        f"median={total_token_stats['median']:.6f}, p25={total_token_stats['p25']:.6f}, p75={total_token_stats['p75']:.6f}"
    )

    target_tensors = _build_target_tensors(df, label_config, class_maps)
    metrics: dict = {
        "target_code_stats": target_stats,
        "completion_token_stats": completion_token_stats,
        "total_token_stats": total_token_stats,
    }

    if is_searched:
        for k in [1, 3, 5]:
            pred_codes = df.select(
                pl.concat_list(
                    [
                        top_k_codes("diag_codes", k).fill_null([]),
                        top_k_codes("proc_codes", k).fill_null([]),
                    ]
                )
                .list.unique()
                .alias("top_codes")
            )["top_codes"].to_list()
            pred_diag = df.select(top_k_codes("diag_codes", k).fill_null([]))[
                "diag_codes"
            ].to_list()
            pred_proc = df.select(top_k_codes("proc_codes", k).fill_null([]))[
                "proc_codes"
            ].to_list()

            combined, diag, proc, raw_stats, filtered_stats = _eval_predictions(
                f"Top-{k}", pred_codes, pred_diag, pred_proc, target_tensors, class_maps
            )
            metrics.update(
                {
                    f"top_{k}": combined,
                    f"top_{k}_diag": diag,
                    f"top_{k}_proc": proc,
                    f"top_{k}_pred_raw_stats": raw_stats,
                    f"top_{k}_pred_filtered_stats": filtered_stats,
                }
            )
    else:
        pred_codes = df.select(
            pl.concat_list(
                [
                    flatten_codes("diag_codes").fill_null([]),
                    flatten_codes("proc_codes").fill_null([]),
                ]
            )
            .list.unique()
            .alias("raw_codes")
        )["raw_codes"].to_list()
        pred_diag = df.select(flatten_codes("diag_codes").fill_null([]))[
            "diag_codes"
        ].to_list()
        pred_proc = df.select(flatten_codes("proc_codes").fill_null([]))[
            "proc_codes"
        ].to_list()

        combined, diag, proc, raw_stats, filtered_stats = _eval_predictions(
            "Raw", pred_codes, pred_diag, pred_proc, target_tensors, class_maps
        )
        metrics.update(
            {
                "raw": combined,
                "raw_diag": diag,
                "raw_proc": proc,
                "raw_pred_raw_stats": raw_stats,
                "raw_pred_filtered_stats": filtered_stats,
            }
        )

    _write_metrics(output_path, metrics)
    print(f"Metrics written to {output_path}")


def run_evaluation_combined(
    verification_paths: list[Path],
    label_path: Path,
    label_config: dict,
    threshold: float,
    data_path: Path,
    code_type: str = "ICD-9",
):
    """Combine SFT codes (>= threshold) with verified-correct codes from one
    or more verification JSONLs, then evaluate.

    Each path in `verification_paths` contributes the union of its
    correct-verdict codes per note. Shapes supported: one-line-per-note
    (run_verification) and one-line-per-(note, term) (run_verify_top_n).
    """
    print(f"Threshold: {threshold}")

    id_col, sft_lookup, class_map = _load_sft_codes(data_path, threshold)
    print(
        f"Loaded SFT predictions: {len(sft_lookup)} notes at threshold >= {threshold}"
    )

    verified_by_path: list[dict] = []
    for p in verification_paths:
        _, verified = _load_verified_codes(p)
        verified_by_path.append(verified)
        n_codes = sum(len(v) for v in verified.values())
        print(f"  {p.name}: {len(verified)} notes, {n_codes} verified codes")

    all_event_ids: set = set(sft_lookup)
    for v in verified_by_path:
        all_event_ids |= set(v)

    records = []
    for eid in all_event_ids:
        combined_codes = set(sft_lookup.get(eid, set()))
        for v in verified_by_path:
            combined_codes |= v.get(eid, set())
        diag = [c for c in combined_codes if not is_procedure_code(c, code_type)]
        proc = [c for c in combined_codes if is_procedure_code(c, code_type)]
        records.append(
            {
                id_col: eid,
                "diag_codes": diag,
                "proc_codes": proc,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )

    pred = cast_id_columns_to_str(pl.DataFrame(records))
    print(f"Combined predictions: {len(pred)} notes")

    labels = _load_labels(label_path, label_config)
    class_maps = _build_class_maps_from_parquet(class_map, code_type=code_type)
    print(f"Loaded labels: {len(labels)} records")

    id_cols = get_id_columns(pred)
    df = labels.join(pred, on=id_cols)
    print(f"Merged dataset: {len(df)} records after join")

    target_stats = _length_stats([len(t) for t in df["target"].to_list()])
    print(
        f"Target codes: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}"
    )

    target_tensors = _build_target_tensors(df, label_config, class_maps)
    pred_codes = df["diag_codes"].list.concat(df["proc_codes"]).list.unique().to_list()
    pred_diag = df["diag_codes"].to_list()
    pred_proc = df["proc_codes"].to_list()

    combined_metrics, diag_metrics, proc_metrics, raw_stats, filtered_stats = (
        _eval_predictions(
            "Combined", pred_codes, pred_diag, pred_proc, target_tensors, class_maps
        )
    )

    output_path = verification_paths[0].parent / "metrics_verified_combined.json"
    _write_metrics(
        output_path,
        {
            "threshold": threshold,
            "verification_paths": [str(p) for p in verification_paths],
            "combined": combined_metrics,
            "combined_diag": diag_metrics,
            "combined_proc": proc_metrics,
            "target_code_stats": target_stats,
            "pred_raw_stats": raw_stats,
            "pred_filtered_stats": filtered_stats,
        },
    )
    print(f"Metrics written to {output_path}")


def run_evaluation_combined_ood(
    verification_paths: list[Path],
    label_path: Path,
    label_config: dict,
    threshold: float,
    data_path: Path,
    code_type: str = "ICD-9",
):
    """Combine SFT codes (>= threshold) with verified-correct codes and report
    metrics on three vocabularies: in-dist (class_map), OOD (label codes
    missing from class_map), and full (in-dist ∪ OOD).
    """
    print(f"Threshold: {threshold}")

    id_col, sft_lookup, class_map_df = _load_sft_codes(data_path, threshold)
    print(
        f"Loaded SFT predictions: {len(sft_lookup)} notes at threshold >= {threshold}"
    )

    verified_by_path: list[dict] = []
    for p in verification_paths:
        _, verified = _load_verified_codes(p)
        verified_by_path.append(verified)
        n_codes = sum(len(v) for v in verified.values())
        print(f"  {p.name}: {len(verified)} notes, {n_codes} verified codes")

    all_event_ids: set = set(sft_lookup)
    for v in verified_by_path:
        all_event_ids |= set(v)

    records = []
    for eid in all_event_ids:
        combined_codes = set(sft_lookup.get(eid, set()))
        for v in verified_by_path:
            combined_codes |= v.get(eid, set())
        diag = [c for c in combined_codes if not is_procedure_code(c, code_type)]
        proc = [c for c in combined_codes if is_procedure_code(c, code_type)]
        records.append(
            {
                id_col: eid,
                "diag_codes": diag,
                "proc_codes": proc,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )

    pred = cast_id_columns_to_str(pl.DataFrame(records))
    print(f"Combined predictions: {len(pred)} notes")

    labels = _load_labels(label_path, label_config)
    in_class_maps = _build_class_maps_from_parquet(class_map_df, code_type=code_type)
    ood_maps, full_maps = _build_ood_and_full_class_maps(
        labels, label_config, in_class_maps
    )
    print(f"Loaded labels: {len(labels)} records")
    print(
        f"Vocab sizes — in-dist: combined={len(in_class_maps[0])}, "
        f"diag={len(in_class_maps[1])}, proc={len(in_class_maps[2])}"
    )
    print(
        f"Vocab sizes — OOD:     combined={len(ood_maps[0])}, "
        f"diag={len(ood_maps[1])}, proc={len(ood_maps[2])}"
    )
    print(
        f"Vocab sizes — full:    combined={len(full_maps[0])}, "
        f"diag={len(full_maps[1])}, proc={len(full_maps[2])}"
    )

    id_cols = get_id_columns(pred)
    df = labels.join(pred, on=id_cols)
    print(f"Merged dataset: {len(df)} records after join")

    target_stats = _length_stats([len(t) for t in df["target"].to_list()])
    print(
        f"Target codes: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}"
    )

    pred_codes = df["diag_codes"].list.concat(df["proc_codes"]).list.unique().to_list()
    pred_diag = df["diag_codes"].to_list()
    pred_proc = df["proc_codes"].to_list()

    in_targets = _build_target_tensors(df, label_config, in_class_maps)
    in_combined, in_diag, in_proc, in_raw_stats, in_filt_stats = _eval_predictions(
        "In-dist", pred_codes, pred_diag, pred_proc, in_targets, in_class_maps
    )

    output: dict = {
        "threshold": threshold,
        "verification_paths": [str(p) for p in verification_paths],
        "vocab_sizes": {
            "in_dist": {
                "combined": len(in_class_maps[0]),
                "diag": len(in_class_maps[1]),
                "proc": len(in_class_maps[2]),
            },
            "ood": {
                "combined": len(ood_maps[0]),
                "diag": len(ood_maps[1]),
                "proc": len(ood_maps[2]),
            },
            "full": {
                "combined": len(full_maps[0]),
                "diag": len(full_maps[1]),
                "proc": len(full_maps[2]),
            },
        },
        "target_code_stats": target_stats,
        "in_dist": in_combined,
        "in_dist_diag": in_diag,
        "in_dist_proc": in_proc,
        "in_dist_pred_raw_stats": in_raw_stats,
        "in_dist_pred_filtered_stats": in_filt_stats,
    }

    if len(ood_maps[0]) == 0:
        print("WARNING: no OOD codes found in labels — skipping OOD eval block.")
    else:
        ood_targets = _build_target_tensors(df, label_config, ood_maps)
        ood_c, ood_d, ood_p, ood_raw, ood_filt = _eval_predictions(
            "OOD", pred_codes, pred_diag, pred_proc, ood_targets, ood_maps
        )
        output.update(
            {
                "ood": ood_c,
                "ood_diag": ood_d,
                "ood_proc": ood_p,
                "ood_pred_raw_stats": ood_raw,
                "ood_pred_filtered_stats": ood_filt,
            }
        )

    full_targets = _build_target_tensors(df, label_config, full_maps)
    full_c, full_d, full_p, full_raw, full_filt = _eval_predictions(
        "Full", pred_codes, pred_diag, pred_proc, full_targets, full_maps
    )
    output.update(
        {
            "full": full_c,
            "full_diag": full_d,
            "full_proc": full_p,
            "full_pred_raw_stats": full_raw,
            "full_pred_filtered_stats": full_filt,
        }
    )

    output_path = verification_paths[0].parent / "metrics_verified_combined_ood.json"
    _write_metrics(output_path, output)
    print(f"Metrics written to {output_path}")


def run_evaluation_from_verification(
    verification_path: Path,
    label_path: Path,
    label_config: dict,
    code_type: str = "ICD-9",
):
    records = []
    with open(verification_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            verdicts = record["content"]["verdicts"]
            correct_codes = [v["code"] for v in verdicts if v["correct"]]
            diag_codes = [c for c in correct_codes if not is_procedure_code(c, code_type)]
            proc_codes = [c for c in correct_codes if is_procedure_code(c, code_type)]
            records.append(
                {
                    "event_id": record["event_id"],
                    "diag_codes": diag_codes,
                    "proc_codes": proc_codes,
                    "completion_tokens": record.get("completion_tokens"),
                    "total_tokens": record.get("total_tokens"),
                }
            )

    pred = cast_id_columns_to_str(pl.DataFrame(records))
    print(f"Loaded predictions: {len(pred)} records")

    labels = _load_labels(label_path, label_config)
    print(f"Loaded labels: {len(labels)} records")

    id_cols = get_id_columns(pred)
    df = labels.join(pred, on=id_cols)
    print(f"Merged dataset: {len(df)} records after join")

    target_stats = _length_stats([len(t) for t in df["target"].to_list()])
    completion_token_stats = _token_stats(df["completion_tokens"].to_list())
    total_token_stats = _token_stats(df["total_tokens"].to_list())
    print(
        f"Target codes: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}"
    )
    print(
        f"Completion tokens: mean={completion_token_stats['mean']:.6f}, std={completion_token_stats['std']:.6f}, "
        f"median={completion_token_stats['median']:.6f}, p25={completion_token_stats['p25']:.6f}, p75={completion_token_stats['p75']:.6f}"
    )
    print(
        f"Total tokens: mean={total_token_stats['mean']:.6f}, std={total_token_stats['std']:.6f}, "
        f"median={total_token_stats['median']:.6f}, p25={total_token_stats['p25']:.6f}, p75={total_token_stats['p75']:.6f}"
    )

    # Note: class_map path must be inferred from label_path directory or passed separately
    class_map_path = label_path.parent / "class_map.parquet"
    class_maps = _build_class_maps_from_parquet(
        pl.read_parquet(class_map_path), code_type=code_type
    )

    target_tensors = _build_target_tensors(df, label_config, class_maps)
    pred_codes = df["diag_codes"].list.concat(df["proc_codes"]).list.unique().to_list()
    pred_diag = df["diag_codes"].to_list()
    pred_proc = df["proc_codes"].to_list()

    combined, diag, proc, raw_stats, filtered_stats = _eval_predictions(
        "Verified", pred_codes, pred_diag, pred_proc, target_tensors, class_maps
    )
    output_path = verification_path.parent / f"{verification_path.stem}_metrics.json"
    _write_metrics(
        output_path,
        {
            "verified": combined,
            "verified_diag": diag,
            "verified_proc": proc,
            "target_code_stats": target_stats,
            "pred_raw_stats": raw_stats,
            "pred_filtered_stats": filtered_stats,
            "completion_token_stats": completion_token_stats,
            "total_token_stats": total_token_stats,
        },
    )
    print(f"Metrics written to {output_path}")


def run_evaluation_from_per_code_verification(
    verification_path: Path,
    label_path: Path,
    label_config: dict,
    threshold: float | None = None,
    data_path: Path | None = None,
    code_type: str = "ICD-9",
):
    """Combine SFT codes (confidence >= threshold) with per-code verified-correct codes, then evaluate.

    Mirrors run_evaluation_combined but handles JSONL from pipeline/verification.py run_verification_per_code,
    where each line is a single (note, code) verdict rather than all verdicts for a note.
    Lines are grouped by event_id before combining with SFT codes.

    The threshold can be passed explicitly or parsed from the verification filename, e.g.
    'gpt-oss-20b_0.35_0.15_verification_low.jsonl' -> threshold=0.15.

    SFT codes are loaded from predictions.json + class_map.parquet in data_path
    (defaults to the verification file's parent directory).
    """
    if threshold is None:
        match = re.search(r"_(\d+\.\d+)_", verification_path.stem)
        if match is None:
            raise ValueError(
                f"Could not parse threshold from filename: {verification_path.name}. "
                "Expected a float like '0.15' surrounded by underscores, "
                "or pass --threshold explicitly."
            )
        threshold = float(match.group(1))
    print(f"Parsed threshold: {threshold}")

    if data_path is None:
        data_path = verification_path.parent

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
    print(
        f"Loaded SFT predictions: {len(sft_lookup)} notes at threshold >= {threshold}"
    )

    verified_lookup: dict = {}
    id_col: str | None = None

    with open(verification_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            if id_col is None:
                id_col = next((c for c in _ID_CANDIDATES if c in r), None)
                if id_col is None:
                    raise ValueError(
                        f"No known ID column found in {verification_path.name}"
                    )

            eid = str(r[id_col])
            correct = [v["code"] for v in r["content"]["verdicts"] if v["correct"]]

            if eid not in verified_lookup:
                verified_lookup[eid] = {"diag": set(), "proc": set()}
            for code in correct:
                if is_procedure_code(code, code_type):
                    verified_lookup[eid]["proc"].add(code)
                else:
                    verified_lookup[eid]["diag"].add(code)

    print(f"Loaded verification results: {len(verified_lookup)} notes")

    all_event_ids = set(sft_lookup) | set(verified_lookup)
    records = []
    for eid in all_event_ids:
        sft_codes = sft_lookup.get(eid, set())
        ver = verified_lookup.get(eid, {"diag": set(), "proc": set()})
        combined_codes = sft_codes | ver["diag"] | ver["proc"]
        diag = [c for c in combined_codes if not is_procedure_code(c, code_type)]
        proc = [c for c in combined_codes if is_procedure_code(c, code_type)]
        records.append(
            {
                id_col: eid,
                "diag_codes": diag,
                "proc_codes": proc,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )

    pred = cast_id_columns_to_str(pl.DataFrame(records))
    print(f"Combined predictions: {len(pred)} notes")

    labels = _load_labels(label_path, label_config)
    class_maps = _build_class_maps_from_parquet(class_map, code_type=code_type)
    print(f"Loaded labels: {len(labels)} records")

    id_cols = get_id_columns(pred)
    df = labels.join(pred, on=id_cols)
    print(f"Merged dataset: {len(df)} records after join")

    target_stats = _length_stats([len(t) for t in df["target"].to_list()])
    print(
        f"Target codes: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}"
    )

    target_tensors = _build_target_tensors(df, label_config, class_maps)
    pred_codes = df["diag_codes"].list.concat(df["proc_codes"]).list.unique().to_list()
    pred_diag = df["diag_codes"].to_list()
    pred_proc = df["proc_codes"].to_list()

    combined_metrics, diag_metrics, proc_metrics, raw_stats, filtered_stats = (
        _eval_predictions(
            "Combined (per-code)",
            pred_codes,
            pred_diag,
            pred_proc,
            target_tensors,
            class_maps,
        )
    )

    output_path = (
        verification_path.parent
        / f"{verification_path.stem}_per_code_combined_metrics.json"
    )
    _write_metrics(
        output_path,
        {
            "threshold": threshold,
            "combined": combined_metrics,
            "combined_diag": diag_metrics,
            "combined_proc": proc_metrics,
            "target_code_stats": target_stats,
            "pred_raw_stats": raw_stats,
            "pred_filtered_stats": filtered_stats,
        },
    )
    print(f"Metrics written to {output_path}")


def run_evaluation_top_n(
    pred_path: Path,
    label_path: Path,
    label_config: dict,
    n: int,
    threshold: float | None = None,
    data_path: Path | None = None,
    code_type: str = "ICD-9",
):
    """Combine SFT codes (confidence >= threshold) with top-N searched codes, then evaluate.

    The threshold can be passed explicitly or parsed from the prediction filename, e.g.
    'openai_gpt-oss-20b_0.35_gap_analysis_low_searched.parquet' -> threshold=0.35.

    SFT codes are loaded from predictions.json + class_map.parquet in data_path
    (defaults to the prediction file's parent directory).

    LLM codes come from the searched parquet where diag_codes/proc_codes are nested
    ranked lists; top-n candidates per diagnosis/procedure are taken.
    """
    if threshold is None:
        match = re.search(r"_(\d+\.\d+)_", pred_path.stem)
        if match is None:
            raise ValueError(
                f"Could not parse threshold from filename: {pred_path.name}. "
                "Expected a float like '0.35' surrounded by underscores, "
                "or pass --threshold explicitly."
            )
        threshold = float(match.group(1))
    print(f"Parsed threshold: {threshold}")

    if data_path is None:
        data_path = pred_path.parent

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
    print(
        f"Loaded SFT predictions: {len(sft_lookup)} notes at threshold >= {threshold}"
    )

    searched = cast_id_columns_to_str(pl.read_parquet(pred_path))
    llm_id_cols = get_id_columns(searched)
    llm_diag = searched.select(
        llm_id_cols + [top_k_codes("diag_codes", n).fill_null([]).alias("diag_top")]
    )
    llm_proc = searched.select(
        llm_id_cols + [top_k_codes("proc_codes", n).fill_null([]).alias("proc_top")]
    )
    llm_lookup = {}
    for row in llm_diag.join(llm_proc, on=llm_id_cols).iter_rows(named=True):
        eid = row[llm_id_cols[0]]
        llm_lookup[eid] = {
            "diag": set(row["diag_top"]),
            "proc": set(row["proc_top"]),
        }
    print(f"Loaded top-{n} LLM codes: {len(llm_lookup)} notes")

    all_event_ids = set(sft_lookup) | set(llm_lookup)
    records = []
    for eid in all_event_ids:
        sft_codes = sft_lookup.get(eid, set())
        llm = llm_lookup.get(eid, {"diag": set(), "proc": set()})
        combined_codes = sft_codes | llm["diag"] | llm["proc"]
        diag = [c for c in combined_codes if not is_procedure_code(c, code_type)]
        proc = [c for c in combined_codes if is_procedure_code(c, code_type)]
        records.append(
            {
                id_cols[0]: eid,
                "diag_codes": diag,
                "proc_codes": proc,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )

    pred = cast_id_columns_to_str(pl.DataFrame(records))
    print(f"Combined predictions: {len(pred)} notes")

    labels = _load_labels(label_path, label_config)
    class_maps = _build_class_maps_from_parquet(class_map, code_type=code_type)
    print(f"Loaded labels: {len(labels)} records")

    id_cols = get_id_columns(pred)
    df = labels.join(pred, on=id_cols)
    print(f"Merged dataset: {len(df)} records after join")

    target_stats = _length_stats([len(t) for t in df["target"].to_list()])
    print(
        f"Target codes: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}"
    )

    target_tensors = _build_target_tensors(df, label_config, class_maps)
    pred_codes = df["diag_codes"].list.concat(df["proc_codes"]).list.unique().to_list()
    pred_diag = df["diag_codes"].to_list()
    pred_proc = df["proc_codes"].to_list()

    combined_metrics, diag_metrics, proc_metrics, raw_stats, filtered_stats = (
        _eval_predictions(
            f"Combined (top-{n})",
            pred_codes,
            pred_diag,
            pred_proc,
            target_tensors,
            class_maps,
        )
    )

    output_path = pred_path.parent / "metrics_top_n.json"
    _write_metrics(
        output_path,
        {
            "threshold": threshold,
            "n": n,
            "combined": combined_metrics,
            "combined_diag": diag_metrics,
            "combined_proc": proc_metrics,
            "target_code_stats": target_stats,
            "pred_raw_stats": raw_stats,
            "pred_filtered_stats": filtered_stats,
        },
    )
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    import sys

    sys.argv = ["null" if a == "None" else a for a in sys.argv]
    CLI(
        [
            run_evaluation,
            run_evaluation_from_verification,
            run_evaluation_combined,
            run_evaluation_combined_ood,
            run_evaluation_from_per_code_verification,
            run_evaluation_top_n,
        ],
        as_positional=False,
    )
