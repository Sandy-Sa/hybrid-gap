import polars as pl
from pathlib import Path
from .vllm_generate import generate_responses
from coding.utils.reproducibility import set_global_seed, DEFAULT_SEED
from coding.utils.helpers import cast_id_columns_to_str, filter_existing_responses
from jsonargparse import CLI
import asyncio
from dotenv import load_dotenv

load_dotenv()
set_global_seed(DEFAULT_SEED)


async def main(
    data_dir: Path = Path(
        "/scratch/gp50/zs5704/projects/prompt-icd/datasets/mdace_full.parquet"
    ),
    model: str = "openai/gpt-oss-20b",
    max_concurrent: int = 2,
    examples: list[tuple[str, str, str]] = None,
    output_dir: Path = Path(f"/scratch/gp50/zs5704/projects/prompt-icd/outputs/mdace"),
    reasoning_effort: str = "low",
    seed: int = DEFAULT_SEED,
    max_tokens: int = None,
):
    data = cast_id_columns_to_str(pl.read_parquet(data_dir))
    output_path = (
        output_dir / f"{model.replace(':','_')}_responses_{reasoning_effort}.jsonl"
    )

    if output_path.exists():
        current_outputs = cast_id_columns_to_str(pl.read_ndjson(output_path))
        data = filter_existing_responses(data, current_outputs)

    response = await generate_responses(
        model=model,
        notes=data,
        examples=examples,
        output_path=output_path,
        max_concurrent=max_concurrent,
        reasoning_effort=reasoning_effort,
        seed=seed,
        max_tokens=max_tokens,
    )
    return response


if __name__ == "__main__":
    CLI(main)
