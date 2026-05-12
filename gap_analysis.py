"""Thin CLI for gap analysis.

Pass 1 (pass_index=1):
    gap_analysis(initial SFT codes) → gap_analysis_pass1.jsonl

Pass 2 (pass_index=2, requires --verification_path):
    Merges verified-correct codes from a prior run_verification output into
    the SFT-≥-threshold code list, then runs gap analysis on that expanded
    set → gap_analysis_pass2.jsonl
"""

import polars as pl
from pathlib import Path
from coding.pipeline.data_loader import load_icd_descriptions, prepare_predictions
from coding.pipeline.gap_pipeline import merge_gap_codes
from coding.pipeline.io import filter_existing
from coding.llm.gap_analysis import generate_responses as gap_generate
from coding.utils.helpers import cast_id_columns_to_str, ensure_dict, format_id, get_id_columns
from coding.utils.reproducibility import set_global_seed, DEFAULT_SEED
from jsonargparse import CLI
from dotenv import load_dotenv

load_dotenv()
set_global_seed(DEFAULT_SEED)


async def run_gap_analysis(
    data_path: Path,
    code_description_path: Path,
    description_column: str,
    threshold: float,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    seed: int,
    max_concurrent: int,
    max_output_tokens: int | None,
    code_type: str = "ICD-9",
    pass_index: int = 1,
    verification_path: Path | None = None,
    max_notes: int | None = None,
    task_timeout: float | None = None,
) -> None:
    """Run one pass of gap analysis.

    Args:
        pass_index: 1 for initial pass on SFT predictions; 2 to run on an
            expanded code list that merges verified-correct codes from a
            prior run_verification output with SFT predictions ≥ threshold.
        verification_path: Required when pass_index=2. Path to a
            run_verification JSONL (records contain `content.verdicts[]`
            with `code` and `correct`).
        max_notes: Limit number of notes processed (for smoke-testing).
    """
    if pass_index == 2 and verification_path is None:
        raise ValueError("--verification_path is required when pass_index=2")

    # --- Load shared data ---
    diagnoses, procedures = load_icd_descriptions(
        code_description_path, description_column, code_type=code_type
    )
    grouped = prepare_predictions(
        data_path,
        threshold,
        description_column,
        diagnoses,
        procedures,
        code_type=code_type,
    )
    print(f"Loaded {grouped.height} notes with ML-predicted codes.")
    if max_notes is not None:
        grouped = grouped.head(max_notes)
        print(f"Limiting to {max_notes} notes for testing.")

    # =========================================================================
    # PASS 1 — gap analysis on initial SFT predictions
    # =========================================================================
    if pass_index == 1:
        output_path = output_dir / f"gap_analysis_pass1.jsonl"

        notes = filter_existing(grouped, output_path)

        if notes.height == 0:
            print("Pass 1: all notes already processed.")
            return

        print(f"Pass 1: running gap analysis on {notes.height} notes...")
        result = await gap_generate(
            model=model,
            notes=notes,
            output_path=output_path,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            code_type=code_type,
        )
        print(
            f"Pass 1: processed {result['processed']} notes, {len(result['errors'])} errors."
        )
        return

    # =========================================================================
    # PASS 2 — merge verified-correct codes from run_verification with the
    # SFT-≥-threshold code list, then run gap analysis on the combined set.
    # =========================================================================
    output_path = output_dir / "gap_analysis_pass2.jsonl"

    id_cols = get_id_columns(grouped)

    # Extract verified-correct codes from run_verification JSONL
    verified = cast_id_columns_to_str(pl.read_ndjson(verification_path))
    verified_records: list[dict] = []
    for row in verified.iter_rows(named=True):
        content = ensure_dict(row["content"], row_id=format_id(row, id_cols))
        codes = [
            v["code"]
            for v in content.get("verdicts", [])
            if v.get("correct") is True
        ]
        verified_records.append(
            {**{c: row[c] for c in id_cols}, "gap_codes": codes}
        )

    verified_codes_df = (
        pl.DataFrame(verified_records)
        if verified_records
        else pl.DataFrame(
            schema={c: grouped.schema[c] for c in id_cols}
            | {"gap_codes": pl.List(pl.String)}
        )
    )

    total_verified = verified_codes_df.select(
        pl.col("gap_codes").list.len().sum()
    ).item()
    n_notes_with_verified = verified_codes_df.filter(
        pl.col("gap_codes").list.len() > 0
    ).height
    print(
        f"Merging {total_verified} verified codes across "
        f"{n_notes_with_verified} notes."
    )

    # Build code → description lookup for merge
    code_desc_lookup: dict[str, str] = {}
    for row in diagnoses.iter_rows(named=True):
        code_desc_lookup[row["ICD9_CODE"]] = row[description_column]
    for row in procedures.iter_rows(named=True):
        code_desc_lookup[row["ICD9_CODE"]] = row[description_column]

    expanded_df = merge_gap_codes(grouped, verified_codes_df, code_desc_lookup)

    expanded_df = filter_existing(expanded_df, output_path)

    if expanded_df.height == 0:
        print("Pass 2: all notes already processed.")
        return

    print(f"Pass 2: running gap analysis on {expanded_df.height} notes...")
    result = await gap_generate(
        model=model,
        notes=expanded_df,
        output_path=output_path,
        reasoning_effort=reasoning_effort,
        seed=seed,
        max_output_tokens=max_output_tokens,
        max_concurrent=max_concurrent,
        code_type=code_type,
        task_timeout=task_timeout,
    )
    print(
        f"Pass 2: processed {result['processed']} notes, {len(result['errors'])} errors."
    )


if __name__ == "__main__":
    import sys

    sys.argv = ["null" if a == "None" else a for a in sys.argv]
    CLI(run_gap_analysis, as_positional=False)
