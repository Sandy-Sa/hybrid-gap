import gc

import polars as pl
import torch
from pathlib import Path
from coding.utils.search import CodeSearcher, search_terms
from coding.utils.helpers import (
    cast_id_columns_to_str,
    parse_content,
)
from coding.pipeline.data_loader import load_icd_descriptions, _normalize_code_type
from jsonargparse import CLI


def run_search(
    code_dir: Path,
    mrconso_path: Path | None,
    code_type: str,
    extraction_path: Path,
    min_score: float,
):
    diag, proc = load_icd_descriptions(code_dir, code_type=code_type)
    print(f"Diagnoses: {diag.shape[0]} codes, Procedures: {proc.shape[0]} codes")
    normalized = _normalize_code_type(code_type)
    if normalized == "ICD-9":
        diag_sab = "ICD9CM"
        proc_sab = "ICD9CM"
    else:
        diag_sab = "ICD10CM"
        proc_sab = "ICD10PCS"

    data = cast_id_columns_to_str(pl.read_ndjson(extraction_path))

    data = data.with_columns(
        pl.col("content")
        .map_elements(
            parse_content,
            return_dtype=pl.Struct(
                [
                    pl.Field("diagnosis_normalized", pl.List(pl.Utf8)),
                    pl.Field("procedure_normalized", pl.List(pl.Utf8)),
                ]
            ),
        )
        .alias("parsed")
    ).unnest("parsed")

    if mrconso_path is not None:
        diagnosis_search = CodeSearcher(
            diag, top_k=10, mrconso_path=mrconso_path, icd_sab=diag_sab
        )
    else:
        diagnosis_search = CodeSearcher(diag, top_k=10)

    data = (
        data.with_columns(
            pl.col("diagnosis_normalized")
            .map_elements(
                lambda x: search_terms(x, diagnosis_search, min_score=min_score),
                return_dtype=pl.Struct(
                    [
                        pl.Field("codes", pl.List(pl.List(pl.String))),
                        pl.Field("scores", pl.List(pl.List(pl.Float64))),
                    ]
                ),
            )
            .alias("diag_search")
        )
        .unnest("diag_search")
        .rename(
            {
                "codes": "diag_codes",
                "scores": "diag_scores",
            }
        )
    )

    del diagnosis_search
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if mrconso_path is not None:
        procedure_search = CodeSearcher(
            proc, top_k=10, mrconso_path=mrconso_path, icd_sab=proc_sab
        )
    else:
        procedure_search = CodeSearcher(proc, top_k=10)

    data = (
        data.with_columns(
            pl.col("procedure_normalized")
            .map_elements(
                lambda x: search_terms(x, procedure_search, min_score=min_score),
                return_dtype=pl.Struct(
                    [
                        pl.Field("codes", pl.List(pl.List(pl.String))),
                        pl.Field("scores", pl.List(pl.List(pl.Float64))),
                    ]
                ),
            )
            .alias("proc_search")
        )
        .unnest("proc_search")
        .rename(
            {
                "codes": "proc_codes",
                "scores": "proc_scores",
            }
        )
    )

    del procedure_search
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data.write_parquet(
        extraction_path.parent / f"{extraction_path.stem}_searched.parquet"
    )
    return data


if __name__ == "__main__":
    import sys

    sys.argv = ["null" if a == "None" else a for a in sys.argv]
    CLI(run_search, as_positional=False)
