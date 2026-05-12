import json
import polars as pl
from pathlib import Path

from coding.utils.helpers import cast_id_columns_to_str, filter_existing_responses


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file; creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_predictions(data_path: Path) -> pl.DataFrame:
    """Load predictions.json from *data_path* into a per-row DataFrame.

    Accepts two on-disk shapes:
      - Column-oriented dict: ``{"event_id": [...], "labels": [...], ...}``
        — the legacy shape. Wrapped as a 1-row DataFrame that callers then
        ``explode(pl.all())`` to get one row per note.
      - List of per-row dicts: ``[{"event_id": ..., "labels": ...}, ...]``
        — loaded directly; height equals number of notes.

    Raises FileNotFoundError (with a pointer to the SFT stage) if the file
    is missing, or ValueError for any other top-level JSON shape.
    """
    pred_file = data_path / "predictions.json"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"predictions.json not found at {pred_file}. "
            "Run the SFT extraction stage first (see dvc.yaml)."
        )
    with open(pred_file) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return pl.from_dicts(raw)
    if isinstance(raw, dict):
        return pl.from_dict({k: [v] for k, v in raw.items()})
    raise ValueError(
        f"predictions.json at {pred_file} must be a list of row dicts or a "
        f"column-oriented dict, got top-level {type(raw).__name__}."
    )


def write_merged_jsonl(
    path: Path,
    *,
    merged_verdicts: list[dict],
    model: str,
    token_usage: dict,
    reasoning: str | None = None,
    finish_reason: str | None = None,
    parse_error: bool = False,
    **extra_fields,
) -> None:
    """Build and append a merged verification record as a single JSONL line.

    Replaces _write_merged_jsonl in vllm_verify.py and vllm_verify_per_code.py.
    """
    record = {
        "model": model,
        "content": {"verdicts": merged_verdicts},
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "parse_error": parse_error,
        **token_usage,
    }
    record.update(extra_fields)
    append_jsonl(path, record)


def filter_existing(
    data_df: pl.DataFrame,
    output_path: Path,
) -> pl.DataFrame:
    """Return rows from data_df not yet present in output_path (JSONL).

    If output_path does not exist, returns data_df unchanged.
    ID columns are auto-detected via coding.utils.helpers.get_id_columns().
    """
    if not output_path.exists():
        return data_df
    existing = cast_id_columns_to_str(pl.read_ndjson(output_path))
    return filter_existing_responses(data_df, existing)
