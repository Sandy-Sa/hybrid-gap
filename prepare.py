from jsonargparse import CLI
from pathlib import Path
import warnings
import polars as pl
import torch
import pyarrow.feather as feather

from coding.data.data_prep import map_class, one_hot_encode
from coding.data.pre_process_functions import remove_rare_codes
from coding.utils.helpers import get_base_model
from coding.utils.reproducibility import DEFAULT_SEED, set_global_seed

from transformers import AutoTokenizer
from typing import Literal, Optional  # Optional kept for class_map
from coding.utils.settings import (
    ID_COLUMN,
    TEXT_COLUMN,
    TARGET_COLUMN,
    SUBJECT_ID_COLUMN,
    MODELS,
)


def create_dataset(
    input_path: Path,
    output_path: Path,
    target_type: str,
    target_columns: list[str],
    model_name: str,
    split_path: Path,
    max_length: int = 512,
    batch_size: int = 4096,
    class_map: Optional[dict] = None,
    truncation_side: Literal["left", "right"] = "left",
    seed: int = DEFAULT_SEED,
):
    """
    Create a dataset from the input data, tokenize the text, and prepare it for training.
    Args:
        input_path (Path): Path to the input data file.
        output_path (Path): Path to save the processed dataset.
        target_columns (list[str]): List of target columns to be combined.
        model_name (str): Name of the model to use for tokenization.
        max_length (int): Maximum length for tokenization.
        split_path (Path): Path to the feather split file.
        class_map (Optional[dict]): Predefined class map for target encoding.
        seed (int): Random seed applied to make dataset creation reproducible.
    Returns:
        train_dataset (pl.DataFrame): Training dataset.
        val_dataset (pl.DataFrame): Validation dataset.
        test_dataset (pl.DataFrame): Test dataset.
    """
    set_global_seed(seed)

    base_model = get_base_model(model_name)
    if base_model not in MODELS:
        raise ValueError(f"Model {base_model} not found in MODELS dictionary.")
    target_token_name = MODELS[base_model].get("target_token")
    target_token = MODELS[base_model].get(target_token_name)
    data = pl.scan_parquet(input_path)

    # Validate that target_columns exist in the data schema
    schema_columns = set(data.collect_schema().names())
    missing = [col for col in target_columns if col not in schema_columns]
    if missing:
        raise ValueError(
            f"target_columns {missing} not found in data. "
            f"Available columns: {sorted(schema_columns)}"
        )

    # filter rows that contain string of target_type
    target_type_columns = [f"{target_column}_type" for target_column in target_columns]
    exprs = [
        pl.col(col).str.contains(target_type, literal=True)
        for col in target_type_columns
    ]
    data = data.filter(pl.any_horizontal(exprs))

    data_count = data.select(pl.len()).collect().item()
    if data_count == 0:
        raise ValueError(
            f"No rows remain after filtering for target_type={target_type!r} "
            f"in columns {target_type_columns}. Check that processed data "
            f"contains matching code types."
        )

    # Null out code columns whose type does not match target_type, so
    # concat_list only includes codes of the requested ICD version.
    null_exprs = []
    for target_column in target_columns:
        type_col = f"{target_column}_type"
        null_exprs.append(
            pl.when(pl.col(type_col).str.contains(target_type, literal=True))
            .then(pl.col(target_column))
            .otherwise(pl.lit(None))
            .alias(target_column)
        )
    data = data.with_columns(null_exprs)

    # Inner-join with the split file *before* counting codes. This makes the
    # split-file the authoritative scope for rare-code filtering, matching
    # Edin et al.'s recipe (load -> split-join -> filter ICD-X -> remove_rare_codes).
    splits = pl.from_arrow(feather.read_table(split_path)).rename({"_id": ID_COLUMN})
    data = data.join(splits.lazy(), on=ID_COLUMN, how="inner").collect(
        engine="streaming"
    )

    # Remove rare codes within the split-scoped dataset.
    data = remove_rare_codes(data, target_columns, 10)

    data = data.with_columns(
        pl.concat_list([pl.col(col).fill_null([]) for col in target_columns]).alias(
            TARGET_COLUMN
        )
    )

    # Drop admissions whose entire code list was eliminated by the rare-code filter.
    data = data.filter(pl.col(TARGET_COLUMN).list.len() > 0)

    # check if output_path exists, if not create it
    output_path.mkdir(parents=True, exist_ok=True)

    if class_map is None:
        class_map = map_class(data, TARGET_COLUMN)

        pl.DataFrame(
            {"key": list(class_map.keys()), "value": list(class_map.values())}
        ).write_parquet(output_path / "class_map.parquet")

    # Load the tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": target_token})
    tokenizer.truncation_side = truncation_side

    # Collect the texts into a list
    texts = data[TEXT_COLUMN]

    # Batch process the texts using the tokenizer

    all_tokenized = {"input_ids": [], "attention_mask": []}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size].to_list()

        tokenized_batch = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        all_tokenized["input_ids"].append(tokenized_batch["input_ids"])
        all_tokenized["attention_mask"].append(tokenized_batch["attention_mask"])

    # Concatenate all batches
    tokenized = {
        "input_ids": torch.cat(all_tokenized["input_ids"], dim=0),
        "attention_mask": torch.cat(all_tokenized["attention_mask"], dim=0),
    }
    # Extract the input IDs
    tokens_list = tokenized["input_ids"]

    # Compute last_token_index from the attention mask rather than from token IDs.
    # Using token IDs fails when the pad token also appears in real content (e.g.
    # <|eot_id|> is both llama3's pad token and its turn-separator token), causing
    # last_token_index to be reported far short of the actual sequence end.
    # attention_mask is 1 for real tokens and 0 for padding (right-padded), so
    # sum-1 gives the 0-based index of the last real token.
    last_token_index_list = (
        tokenized["attention_mask"].sum(dim=1).clamp(min=1) - 1
    ).tolist()

    # Add the tokens as a new column to the DataFrame
    data = data.with_columns(pl.Series("tokens", tokens_list))

    all_labels = set(data[TARGET_COLUMN].explode().drop_nulls().to_list())
    missing_labels = all_labels - set(class_map.keys())
    if missing_labels:
        warnings.warn(
            f"{len(missing_labels)} label(s) in the data are not in class_map and will "
            f"be silently ignored. First few: {sorted(missing_labels)[:5]}",
            stacklevel=2,
        )

    data = data.with_columns(
        pl.col(TARGET_COLUMN)
        .map_elements(
            lambda x: one_hot_encode(x, class_map),
            return_dtype=list[int],
        )
        .alias("target_tensor")
    )

    first_diagnosis_column = (
        "first_diagnosis_code" if "first_diagnosis_code" in data.columns else None
    )

    if first_diagnosis_column is not None:
        data = data.with_columns(
            pl.col(first_diagnosis_column)
            .map_elements(
                lambda code: class_map.get(code, -1) if code is not None else -1,
                return_dtype=int,
            )
            .alias("first_diagnosis_index")
        )

    data = data.with_columns(pl.Series("last_token_index", last_token_index_list))

    split_names = ("train", "val", "test")
    split_datasets = {}
    for name in split_names:
        ds = data.filter(pl.col("split") == name)
        if len(ds) == 0:
            raise ValueError(
                f"Split '{name}' is empty after partitioning. "
                f"Available splits in file: {data['split'].unique().to_list()}"
            )
        ds = ds.drop("split")
        ds.write_parquet(output_path / f"{name}_dataset.parquet")
        split_datasets[name] = ds

    return split_datasets["train"], split_datasets["val"], split_datasets["test"]


if __name__ == "__main__":
    CLI(create_dataset)
