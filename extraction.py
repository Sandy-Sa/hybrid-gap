import polars as pl
from pathlib import Path
from coding.llm.extraction import generate_responses
from coding.utils.reproducibility import set_global_seed, DEFAULT_SEED
from coding.pipeline.io import filter_existing
from coding.utils.helpers import cast_id_columns_to_str
from jsonargparse import CLI
from dotenv import load_dotenv

load_dotenv()
set_global_seed(DEFAULT_SEED)


async def extract_clinical_notes(
    data_dir: Path,
    model: str,
    output_dir: Path,
    reasoning_effort: str,
    seed: int,
    max_tokens: int | None,
    max_concurrent: int,
    code_type: str = "ICD-9",
    task_timeout: float | None = None,
) -> None:
    data = cast_id_columns_to_str(pl.read_parquet(data_dir))
    output_path = output_dir / f"extraction.jsonl"

    data = filter_existing(data, output_path)

    if data.height == 0:
        print("All notes already processed. Nothing to do.")
        return

    print(f"Processing {data.height} notes for extraction...")

    result = await generate_responses(
        model=model,
        notes=data,
        output_path=output_path,
        reasoning_effort=reasoning_effort,
        seed=seed,
        max_output_tokens=max_tokens,
        max_concurrent=max_concurrent,
        code_type=code_type,
        task_timeout=task_timeout,
    )

    print(f"Processed {result['processed']} notes.")
    if result["errors"]:
        print(f"Completed with {len(result['errors'])} errors:")
        for err in result["errors"]:
            print(f"  {err}")
    else:
        print("Extraction completed successfully.")


if __name__ == "__main__":
    import sys

    sys.argv = ["null" if a == "None" else a for a in sys.argv]
    CLI(extract_clinical_notes, as_positional=False)
