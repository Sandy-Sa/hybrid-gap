"""Shared data loading helpers for the ICD coding pipeline.

Consolidates the near-identical data loading blocks that appear in
run_gap_analysis.py, run_iterative_gap_analysis.py, and run_verification.py.
"""

import json
import polars as pl
from pathlib import Path

from coding.pipeline.io import load_predictions
from coding.utils.helpers import (
    cast_id_columns_to_str,
    get_id_columns,
    parse_content,
    reformat_icd9cm_code,
    reformat_icd9pcs_code,
    reformat_icd10cm_code,
    resolve_all_descriptions,
)


def _resolve_csv(directory: Path, name_upper: str) -> Path:
    upper = directory / name_upper
    if upper.exists():
        return upper
    return directory / name_upper.lower()


def _normalize_code_type(code_type: str) -> str:
    """Normalize 'ICD-9' / 'icd9' / 'ICD10' / 'ICD-10' → 'ICD-9' or 'ICD-10'."""
    key = code_type.upper().replace("-", "").replace("_", "")
    if key == "ICD9":
        return "ICD-9"
    if key == "ICD10":
        return "ICD-10"
    raise ValueError(f"Unsupported code_type: {code_type!r}. Use 'ICD-9' or 'ICD-10'.")


def _code_type_config(code_type: str):
    """Return (icd_version, diag_reformat, proc_reformat) for the given code_type."""
    normalized = _normalize_code_type(code_type)
    if normalized == "ICD-9":
        return 9, reformat_icd9cm_code, reformat_icd9pcs_code
    # ICD-10-PCS codes are 7-char fixed, no dots — identity reformat.
    return 10, reformat_icd10cm_code, lambda code: code


def _load_icd_table(csv_path: Path, icd_version: int, reformat_fn) -> pl.DataFrame:
    """Load one ICD description CSV (MIMIC-III or MIMIC-IV format) into a common schema.

    MIMIC-III has columns ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE (ICD-9 only).
    MIMIC-IV has columns icd_code, icd_version, long_title (ICD-9 and ICD-10 mixed).

    Returns a DataFrame with columns ICD9_CODE, SHORT_TITLE, LONG_TITLE, with the
    ICD9_CODE column reformatted. SHORT_TITLE falls back to LONG_TITLE for MIMIC-IV.
    """
    header = pl.read_csv(csv_path, n_rows=0).columns

    if "icd_code" in header and "icd_version" in header:
        df = (
            pl.read_csv(
                csv_path,
                schema_overrides={"icd_code": pl.String, "icd_version": pl.Int64},
            )
            .filter(pl.col("icd_version") == icd_version)
            .rename({"icd_code": "ICD9_CODE", "long_title": "LONG_TITLE"})
            .with_columns(pl.col("LONG_TITLE").alias("SHORT_TITLE"))
            .select("ICD9_CODE", "SHORT_TITLE", "LONG_TITLE")
        )
    else:
        if icd_version != 9:
            raise ValueError(
                f"{csv_path} is MIMIC-III format (ICD-9 only) but code_type requests "
                f"ICD-{icd_version}."
            )
        df = pl.read_csv(
            csv_path,
            schema_overrides={"ICD9_CODE": pl.String},
        ).select("ICD9_CODE", "SHORT_TITLE", "LONG_TITLE")

    return df.with_columns(
        pl.col("ICD9_CODE").map_elements(reformat_fn, return_dtype=pl.String)
    )


def load_icd_descriptions(
    code_description_path: Path,
    description_column: str = "LONG_TITLE",
    code_type: str = "ICD-9",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and reformat ICD diagnosis and procedure description tables.

    Supports both MIMIC-III (uppercase D_ICD_*.csv.gz; columns ROW_ID,
    ICD9_CODE, SHORT_TITLE, LONG_TITLE; ICD-9 only) and MIMIC-IV (lowercase
    d_icd_*.csv.gz; columns icd_code, icd_version, long_title; ICD-9 + ICD-10).
    For MIMIC-IV, rows are filtered to the requested icd_version.

    Args:
        code_description_path: Directory containing the description files.
        description_column: Column to use for descriptions (kept for symmetry).
        code_type: "ICD-9" or "ICD-10".

    Returns:
        (diagnoses_df, procedures_df) with columns ICD9_CODE, SHORT_TITLE,
        LONG_TITLE. ICD9_CODE is retained as the column name for both code
        systems for downstream compatibility.
    """
    icd_version, diag_reformat, proc_reformat = _code_type_config(code_type)
    diagnoses = _load_icd_table(
        _resolve_csv(code_description_path, "D_ICD_DIAGNOSES.csv.gz"),
        icd_version,
        diag_reformat,
    )
    procedures = _load_icd_table(
        _resolve_csv(code_description_path, "D_ICD_PROCEDURES.csv.gz"),
        icd_version,
        proc_reformat,
    )
    return diagnoses, procedures


def prepare_predictions(
    data_path: Path,
    threshold: float,
    description_column: str,
    diagnoses_df: pl.DataFrame,
    procedures_df: pl.DataFrame,
    threshold_high: float | None = None,
    code_type: str = "ICD-9",
) -> pl.DataFrame:
    """Load SFT predictions and join with note text, returning per-note grouped data.

    Reads predictions.json, test_dataset.parquet, and class_map.parquet from
    data_path. Filters predictions by probability threshold, joins ICD
    descriptions, and groups per note.

    Args:
        data_path: Directory with predictions.json, test_dataset.parquet,
            class_map.parquet.
        threshold: Minimum prediction probability to include (inclusive).
        description_column: Column in the ICD CSVs to use as description
            (e.g. "LONG_TITLE").
        diagnoses_df: DataFrame from load_icd_descriptions (diagnoses).
        procedures_df: DataFrame from load_icd_descriptions (procedures).
        threshold_high: If set, applies a band filter: threshold <= prob < threshold_high.
            When None, only the lower bound (>= threshold) is applied.

    Returns:
        DataFrame grouped by note with columns:
            [id_cols..., "code", "description", "sft_prob", "text"]
        where code, description, and sft_prob are list columns.
    """
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
    data = cast_id_columns_to_str(pl.read_parquet(data_path / "test_dataset.parquet"))
    class_map = pl.read_parquet(data_path / "class_map.parquet")

    code = resolve_all_descriptions(
        class_map.select(pl.col("key").alias("ICD9_CODE")),
        diagnoses_df,
        procedures_df,
        code_type=code_type,
    ).rename({description_column: "description"})

    df = prediction.with_columns(
        pl.int_ranges(pl.col("labels").list.len()).alias("value"),
    )

    id_cols = get_id_columns(df)

    if threshold_high is not None:
        prob_filter = pl.col("predictions").is_between(
            threshold, threshold_high, closed="left"
        )
    else:
        prob_filter = pl.col("predictions") >= threshold

    df = (
        df.explode(["labels", "predictions", "value"])
        .filter(prob_filter)
        .join(class_map, on="value", how="left")
        .rename({"labels": "target", "value": "code_index", "key": "code"})
        .join(code, left_on="code", right_on="ICD9_CODE", how="left")
        .sort(id_cols)
    )

    id_cols = get_id_columns(df)
    grouped = (
        df.group_by(id_cols)
        .agg(
            pl.col("code"),
            pl.col("description"),
            pl.col("predictions").alias("sft_prob"),
        )
        .join(
            data.select(id_cols + ["text"]),
            on=id_cols,
            how="left",
        )
    )

    return grouped


def prepare_search_inputs(
    gap_jsonl_path: Path,
    data_path: Path,
) -> pl.DataFrame:
    """Parse gap analysis JSONL output and join with note text.

    Reads the gap analysis output, extracts normalized diagnosis and procedure
    terms from each row, and joins with the note text from test_dataset.parquet.
    The search step (CodeSearcher) is intentionally NOT run here — that belongs
    in coding/pipeline/gap_pipeline.py (R5).

    Args:
        gap_jsonl_path: Path to gap analysis output JSONL.
        data_path: Directory containing test_dataset.parquet.

    Returns:
        DataFrame with columns:
            [id_cols..., "diagnosis_normalized", "procedure_normalized", "text"]
        where diagnosis_normalized and procedure_normalized are list[str].
    """
    gaps = cast_id_columns_to_str(pl.read_ndjson(gap_jsonl_path))
    id_cols = get_id_columns(gaps)

    records = []
    for row in gaps.iter_rows(named=True):
        content = row["content"]
        if isinstance(content, str):
            content = json.loads(content)

        parsed = parse_content(content)
        id_values = {c: row[c] for c in id_cols}
        records.append(
            {
                **id_values,
                "diagnosis_normalized": parsed["diagnosis_normalized"],
                "procedure_normalized": parsed["procedure_normalized"],
            }
        )

    if not records:
        schema = {c: pl.String for c in id_cols} | {
            "diagnosis_normalized": pl.List(pl.String),
            "procedure_normalized": pl.List(pl.String),
        }
        result = pl.DataFrame(schema=schema)
    else:
        result = pl.DataFrame(records)

    data = cast_id_columns_to_str(pl.read_parquet(data_path / "test_dataset.parquet"))
    result = result.join(data.select(id_cols + ["text"]), on=id_cols, how="left")

    return result
