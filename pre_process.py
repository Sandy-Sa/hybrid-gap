from jsonargparse import CLI
from pathlib import Path
import polars as pl

from coding.data.settings import DATASETS
from coding.data.pre_process_functions import (
    pre_process_mimiciii,
    pre_process_mimiciv,
    pre_process_mimiciii_clean,
)
from coding.utils.reproducibility import DEFAULT_SEED, set_global_seed


PROCESSORS = {
    "MIMIC-III": pre_process_mimiciii,
    "MIMIC-IV": pre_process_mimiciv,
    "MIMIC-III-CLEAN": pre_process_mimiciii_clean,
}


def pre_process_data(
    raw_data_path: Path,
    dataset_name: str,
    remove_lab_results: bool = True,
    processed_data_path: Path = Path("data/processed_data.parquet"),
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """Preprocess a dataset and optionally persist the result.

    Parameters
    ----------
    raw_data_path:
        Directory containing the raw dataset files.
    dataset_name:
        Name of the dataset to preprocess (e.g. ``"MIMIC-III"`` or ``"MIMIC-IV"``).
    remove_lab_results:
        Whether to remove lab results from clinical notes.
    processed_data_path:
        Directory where the processed data should be written as a Parquet file.
        If ``None``, the processed data is returned without being written to disk.

    seed:
        Random seed applied to guarantee reproducible preprocessing.

    Returns
    -------
    pl.DataFrame
        The processed dataset. When ``processed_data_path`` is provided, this
        DataFrame is also saved in that directory as a Parquet file.
    """
    set_global_seed(seed)

    processor = PROCESSORS.get(dataset_name)
    if processor is None:
        available = ", ".join(PROCESSORS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available datasets are: {available}"
        )
    config = DATASETS[dataset_name]

    # Load the raw data lazily to avoid unnecessary memory use
    raw_notes = pl.scan_csv(raw_data_path / config["notes_path"])
    raw_diagnoses = pl.scan_csv(
        raw_data_path / config["diagnoses_path"],
        schema_overrides={config["diagnoses_code"]: pl.String},
    )
    raw_procedures = pl.scan_csv(
        raw_data_path / config["procedure_path"],
        schema_overrides={config["procedure_code"]: pl.String},
    )

    # Process notes
    return processor(
        raw_notes,
        raw_diagnoses,
        raw_procedures,
        config,
        remove_lab_results,
        processed_data_path,
    )


if __name__ == "__main__":
    CLI(pre_process_data)
