import json
import polars as pl
from pathlib import Path
from coding.utils.helpers import (
    cast_id_columns_to_str,
    get_id_columns,
    resolve_all_descriptions,
    safe_model_name,
)
from coding.pipeline.data_loader import load_icd_descriptions, prepare_predictions
from coding.pipeline.io import load_optimal_thresholds, load_predictions
from coding.llm.verification import (
    generate_responses,
    generate_top_n_responses,
    generate_per_code_responses,
)
from coding.utils.reproducibility import set_global_seed, DEFAULT_SEED
from coding.utils.guideline_integration import init_guideline_searcher

from jsonargparse import CLI
from dotenv import load_dotenv

load_dotenv()
set_global_seed(DEFAULT_SEED)


async def run_verification(
    data_path: Path,
    code_description_path: Path,
    description_column: str,
    threshold_high: float | None,
    threshold_low: float | None,
    separate_by_code_type: bool,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    max_concurrent: int,
    seed: int,
    max_output_tokens: int | None,
    chunk_size: int,
    guidelines_path: Path | None,
    code_type: str = "ICD-9",
    task_timeout: float | None = None,
) -> None:
    if threshold_high is None or threshold_low is None:
        opt = load_optimal_thresholds(data_path)
        if threshold_high is None:
            threshold_high = float(opt["micro"])
        if threshold_low is None:
            threshold_low = float(opt["macro"])

    if guidelines_path is not None:
        init_guideline_searcher(guidelines_path, code_type=code_type)

    diagnoses, procedures = load_icd_descriptions(
        code_description_path, description_column, code_type=code_type
    )

    if separate_by_code_type:
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
        data = cast_id_columns_to_str(
            pl.read_parquet(data_path / "test_dataset.parquet")
        )
        class_map = pl.read_parquet(data_path / "class_map.parquet")
        id_cols = get_id_columns(prediction)

        code = resolve_all_descriptions(
            class_map.select(pl.col("key").alias("ICD9_CODE")),
            diagnoses,
            procedures,
            code_type=code_type,
        ).rename({description_column: "description"})

        prediction = prediction.with_columns(
            pl.int_ranges(pl.col("labels").list.len()).alias("value"),
        )

        df = (
            prediction.explode(["labels", "predictions", "value"])
            .filter(
                pl.col("predictions").is_between(
                    threshold_low, threshold_high, closed="left"
                )
            )
            .join(class_map, on="value", how="left")
            .rename({"labels": "target", "value": "code_index", "key": "code"})
            .join(code, left_on="code", right_on="ICD9_CODE", how="left")
            .sort(id_cols)
        )

        grouped = (
            df.group_by(id_cols + ["code_type"])
            .agg(
                pl.col("code"),
                pl.col("description"),
                pl.col("predictions"),
            )
            .join(
                data.select(id_cols + ["text"]),
                on=id_cols,
                how="left",
            )
        )

        diag_df = grouped.filter(pl.col("code_type") == "diagnosis").drop("code_type")
        proc_df = grouped.filter(pl.col("code_type") == "procedure").drop("code_type")

        diag_output = output_dir / "in_band_diagnosis_verification.jsonl"

        proc_output = output_dir / "in_band_procedure_verification.jsonl"

        await generate_responses(
            model=model,
            notes=diag_df,
            output_path=diag_output,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size,
            extra_fields={"code_type": "diagnosis"},
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            task_timeout=task_timeout,
        )
        await generate_responses(
            model=model,
            notes=proc_df,
            output_path=proc_output,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size,
            extra_fields={"code_type": "procedure"},
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            task_timeout=task_timeout,
        )
    else:
        grouped = prepare_predictions(
            data_path,
            threshold_low,
            description_column,
            diagnoses,
            procedures,
            threshold_high=threshold_high,
            code_type=code_type,
        ).rename({"sft_prob": "predictions"})

        output_path = output_dir / "in_band_verification.jsonl"

        await generate_responses(
            model=model,
            notes=grouped,
            output_path=output_path,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size,
            extra_fields=None,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            task_timeout=task_timeout,
        )


async def run_verify_top_n(
    searched_path: Path,
    data_path: Path,
    code_description_path: Path,
    description_column: str,
    top_n: int,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    max_concurrent: int,
    seed: int,
    max_output_tokens: int | None,
    chunk_size: int,
    guidelines_path: Path | None,
    code_type: str = "ICD-9",
    task_timeout: float | None = None,
) -> None:
    """Verify top-N candidate codes from search results for each extracted medical term.

    For each normalized term in the searched parquet, the top-N candidate ICD codes
    are verified against the medical note. At most one code per term is expected to
    be correct (mutual exclusivity enforced in the prompt).

    Args:
        searched_path: Path to *_searched.parquet produced by pipeline/search.py.
        data_path: Directory containing test_dataset.parquet (provides 'text' column).
        code_description_path: Directory with D_ICD_DIAGNOSES.csv.gz and D_ICD_PROCEDURES.csv.gz.
        description_column: Column to use for descriptions, e.g. "LONG_TITLE" or "SHORT_TITLE".
        top_n: Number of top candidate codes to verify per term.
        model: LLM model identifier.
        reasoning_effort: Reasoning effort level ("low", "medium", "high").
        output_dir: Directory to write the output JSONL file.
        max_concurrent: Maximum concurrent LLM requests.
        seed: Random seed for reproducibility.
        max_output_tokens: Optional token limit for LLM responses.
        chunk_size: Max codes per LLM call (rarely needed since top_n is usually small).
        guidelines_path: Optional path to coding guidelines for per-code hints.
    """
    if guidelines_path is not None:
        init_guideline_searcher(guidelines_path, code_type=code_type)

    searched = cast_id_columns_to_str(pl.read_parquet(searched_path))
    data = cast_id_columns_to_str(pl.read_parquet(data_path / "test_dataset.parquet"))
    id_cols = get_id_columns(searched)

    diagnoses, procedures = load_icd_descriptions(
        code_description_path, description_column, code_type=code_type
    )

    diag_codes_flat = (
        searched.select(pl.col("diag_codes").explode().explode().alias("ICD9_CODE"))
        .drop_nulls()
        .unique()
    )
    proc_codes_flat = (
        searched.select(pl.col("proc_codes").explode().explode().alias("ICD9_CODE"))
        .drop_nulls()
        .unique()
    )
    all_codes_df = pl.concat([diag_codes_flat, proc_codes_flat]).unique()

    code_desc = resolve_all_descriptions(
        all_codes_df,
        diagnoses,
        procedures,
        code_type=code_type,
    ).rename({description_column: "description"})

    description_lookup = dict(
        zip(code_desc["ICD9_CODE"].to_list(), code_desc["description"].to_list())
    )

    diag = (
        searched.select(id_cols + ["diagnosis_normalized", "diag_codes", "diag_scores"])
        .explode(["diagnosis_normalized", "diag_codes", "diag_scores"])
        .rename(
            {
                "diagnosis_normalized": "term",
                "diag_codes": "codes",
                "diag_scores": "scores",
            }
        )
        .with_columns(pl.lit("diagnosis").alias("code_type"))
    )

    proc = (
        searched.select(id_cols + ["procedure_normalized", "proc_codes", "proc_scores"])
        .explode(["procedure_normalized", "proc_codes", "proc_scores"])
        .rename(
            {
                "procedure_normalized": "term",
                "proc_codes": "codes",
                "proc_scores": "scores",
            }
        )
        .with_columns(pl.lit("procedure").alias("code_type"))
    )

    terms = pl.concat([diag, proc])

    terms = terms.with_columns(
        [
            pl.col("codes").list.head(top_n),
            pl.col("scores").list.head(top_n),
        ]
    )

    terms = terms.with_columns(
        pl.col("codes")
        .map_elements(
            lambda codes: [description_lookup.get(c, "") for c in codes],
            return_dtype=pl.List(pl.String),
        )
        .alias("descriptions")
    )

    terms = terms.join(
        data.select(id_cols + ["text"]),
        on=id_cols,
        how="left",
    )

    terms = terms.filter(pl.col("term").is_not_null())

    output_path = output_dir / "searched_topn_verification.jsonl"

    await generate_top_n_responses(
        model=model,
        terms_df=terms,
        output_path=output_path,
        reasoning_effort=reasoning_effort,
        seed=seed,
        max_output_tokens=max_output_tokens,
        max_concurrent=max_concurrent,
        chunk_size=chunk_size,
        top_n=top_n,
        task_timeout=task_timeout,
    )


async def run_verification_per_code(
    data_path: Path,
    code_description_path: Path,
    description_column: str,
    threshold_high: float | None,
    threshold_low: float | None,
    separate_by_code_type: bool,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    max_concurrent: int,
    seed: int,
    max_output_tokens: int | None,
    chunk_size: int,
    guidelines_path: Path | None,
    code_type: str = "ICD-9",
    task_timeout: float | None = None,
) -> None:
    """Verify codes one per row (no chunking/grouping) using generate_per_code_responses."""
    if threshold_high is None or threshold_low is None:
        opt = load_optimal_thresholds(data_path)
        if threshold_high is None:
            threshold_high = float(opt["micro"])
        if threshold_low is None:
            threshold_low = float(opt["macro"])

    if guidelines_path is not None:
        init_guideline_searcher(guidelines_path, code_type=code_type)

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
    id_cols = get_id_columns(prediction)

    diagnoses, procedures = load_icd_descriptions(
        code_description_path, description_column, code_type=code_type
    )

    code = resolve_all_descriptions(
        class_map.select(pl.col("key").alias("ICD9_CODE")),
        diagnoses,
        procedures,
        code_type=code_type,
    ).rename({description_column: "description"})

    prediction = prediction.with_columns(
        pl.int_ranges(pl.col("labels").list.len()).alias("value"),
    )

    df = (
        prediction.explode(["labels", "predictions", "value"])
        .filter(
            pl.col("predictions").is_between(
                threshold_low, threshold_high, closed="left"
            )
        )
        .join(class_map, on="value", how="left")
        .rename({"labels": "target", "value": "code_index", "key": "code"})
        .join(code, left_on="code", right_on="ICD9_CODE", how="left")
        .sort(id_cols)
    )

    if separate_by_code_type:
        grouped = (
            df.group_by(id_cols + ["code_type"])
            .agg(
                pl.col("code"),
                pl.col("description"),
                pl.col("predictions"),
            )
            .join(
                data.select(id_cols + ["text"]),
                on=id_cols,
                how="left",
            )
        )

        diag_df = grouped.filter(pl.col("code_type") == "diagnosis").drop("code_type")
        proc_df = grouped.filter(pl.col("code_type") == "procedure").drop("code_type")

        safe_model = safe_model_name(model)
        diag_output = (
            output_dir
            / safe_model
            / f"threshold_{threshold_low}_{threshold_high}"
            / f"diagnosis_per_code_verification_{reasoning_effort}.jsonl"
        )
        proc_output = (
            output_dir
            / safe_model
            / f"threshold_{threshold_low}_{threshold_high}"
            / f"procedure_per_code_verification_{reasoning_effort}.jsonl"
        )

        await generate_per_code_responses(
            model=model,
            notes=diag_df,
            output_path=diag_output,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size,
            extra_fields={"code_type": "diagnosis"},
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            task_timeout=task_timeout,
        )
        await generate_per_code_responses(
            model=model,
            notes=proc_df,
            output_path=proc_output,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size,
            extra_fields={"code_type": "procedure"},
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            task_timeout=task_timeout,
        )
    else:
        grouped = df.join(data.select(id_cols + ["text"]), on=id_cols, how="left")

        safe_model = safe_model_name(model)
        output_path = (
            output_dir
            / safe_model
            / f"threshold_{threshold_low}_{threshold_high}"
            / f"per_code_verification_{reasoning_effort}.jsonl"
        )

        await generate_per_code_responses(
            model=model,
            notes=grouped,
            output_path=output_path,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            task_timeout=task_timeout,
        )


if __name__ == "__main__":
    import sys

    sys.argv = ["null" if a == "None" else a for a in sys.argv]
    CLI(
        [run_verification, run_verify_top_n, run_verification_per_code],
        as_positional=False,
    )
