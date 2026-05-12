"""Pure orchestration helpers for the two-pass iterative gap analysis pipeline.

No CLI, no load_dotenv. All functions take and return plain DataFrames or dicts.
Called by pipeline/gap_analysis.py (the thin CLI entry point).
"""

import polars as pl
from pathlib import Path

from coding.utils.helpers import (
    cast_id_columns_to_str,
    ensure_dict,
    format_id,
    get_id_columns,
    parse_content,
)
from coding.utils.search import CodeSearcher, search_terms


def extract_gap_codes(
    gap_jsonl_path: Path,
    diagnosis_searcher: CodeSearcher,
    procedure_searcher: CodeSearcher,
    min_score: float,
) -> pl.DataFrame:
    """Parse gap analysis output, search for ICD codes, return per-note candidates.

    For each note in the gap JSONL, normalised diagnosis and procedure terms are
    searched against the ICD-9 dictionary. The top-1 code per term (above
    min_score) is collected into a list for that note.

    Args:
        gap_jsonl_path: Path to gap analysis JSONL (output of vllm_gap_analysis).
        diagnosis_searcher: CodeSearcher built on the diagnosis code table.
        procedure_searcher: CodeSearcher built on the procedure code table.
        min_score: Minimum RRF score — results below this are dropped.

    Returns:
        DataFrame with columns [id_cols..., "gap_codes": List[str]].
        Each "gap_codes" entry is the list of top-1 ICD codes found for that note
        (may be empty if no terms cleared the score threshold).
    """
    gaps = cast_id_columns_to_str(pl.read_ndjson(gap_jsonl_path))
    id_cols = get_id_columns(gaps)

    records = []
    for row in gaps.iter_rows(named=True):
        content = ensure_dict(row["content"], row_id=format_id(row, id_cols))

        parsed = parse_content(content)
        diag_terms = parsed["diagnosis_normalized"]
        proc_terms = parsed["procedure_normalized"]

        new_codes: list[str] = []

        if diag_terms:
            diag_results = search_terms(diag_terms, diagnosis_searcher, min_score=min_score)
            for codes_list in diag_results["codes"]:
                if codes_list:
                    new_codes.append(codes_list[0])  # top-1

        if proc_terms:
            proc_results = search_terms(proc_terms, procedure_searcher, min_score=min_score)
            for codes_list in proc_results["codes"]:
                if codes_list:
                    new_codes.append(codes_list[0])  # top-1

        id_values = {c: row[c] for c in id_cols}
        records.append({**id_values, "gap_codes": new_codes})

    if not records:
        return pl.DataFrame(
            schema={c: pl.String for c in id_cols} | {"gap_codes": pl.List(pl.String)}
        )

    return pl.DataFrame(records)


def notes_with_gaps(gap_jsonl_path: Path) -> pl.DataFrame:
    """Return a DataFrame of note IDs where the LLM found at least one gap.

    A note "has gaps" if the gap analysis content contains at least one entry
    in diagnoses or procedures — regardless of whether ICD codes were found
    via search. Used to limit pass 2 to notes worth re-analysing.

    Args:
        gap_jsonl_path: Path to gap analysis JSONL.

    Returns:
        DataFrame with id_cols only (no other columns), one row per note with gaps.
    """
    gaps = cast_id_columns_to_str(pl.read_ndjson(gap_jsonl_path))
    id_cols = get_id_columns(gaps)

    has_gap_ids = []
    for row in gaps.iter_rows(named=True):
        content = ensure_dict(row["content"], row_id=format_id(row, id_cols))
        if content.get("diagnoses") or content.get("procedures"):
            has_gap_ids.append({c: row[c] for c in id_cols})

    if not has_gap_ids:
        return pl.DataFrame(schema={c: pl.String for c in id_cols})

    return pl.DataFrame(has_gap_ids)


def merge_gap_codes(
    original_df: pl.DataFrame,
    gap_codes_df: pl.DataFrame,
    code_desc_lookup: dict[str, str],
) -> pl.DataFrame:
    """Merge newly discovered gap codes into the original assigned code list.

    For each note in original_df, any codes in gap_codes_df that are not
    already in the assigned set are appended. Deduplication is done by code
    string — no two identical codes appear in the merged list.

    Args:
        original_df: Per-note DataFrame with columns [id_cols..., "text",
            "code": List[str], "description": List[str]].
        gap_codes_df: Output of extract_gap_codes — columns [id_cols...,
            "gap_codes": List[str]].
        code_desc_lookup: Mapping from ICD code string to description string.
            Used to populate "description" for newly added codes.

    Returns:
        DataFrame with the same schema as original_df but with expanded
        "code" and "description" lists.
    """
    id_cols = get_id_columns(original_df)

    expanded_records = []
    for row in original_df.iter_rows(named=True):
        original_codes = list(row["code"])
        original_descs = list(row["description"])

        gap_row = gap_codes_df.filter(
            pl.all_horizontal([pl.col(c) == row[c] for c in id_cols])
        )
        if gap_row.height > 0:
            new_codes = list(gap_row["gap_codes"][0])
            if new_codes:
                existing_set = set(original_codes)
                for nc in new_codes:
                    if nc not in existing_set:
                        original_codes.append(nc)
                        original_descs.append(code_desc_lookup.get(nc, "Unknown"))
                        existing_set.add(nc)

        expanded_records.append(
            {
                **{c: row[c] for c in id_cols},
                "text": row["text"],
                "code": original_codes,
                "description": original_descs,
            }
        )

    return pl.DataFrame(expanded_records)
