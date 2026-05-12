import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from coding.utils.settings import ID_COLUMN


def _load_class_map(map_path: Path) -> dict[str, int]:
    if not map_path.exists():
        return {}
    class_map_df = pl.read_parquet(map_path)
    return dict(zip(class_map_df["key"], class_map_df["value"]))


def _unwrap_field(value):
    if isinstance(value, pl.Series):
        if value.len() == 0:
            return None
        return value[0]
    return value


def map_class(data: pl.LazyFrame, target_column: str) -> dict:
    """Create a mapping from class labels to indices based on the target column.
    Args:
        data (pl.LazyFrame): The input data containing the target column.
        target_column (str): The name of the target column to map.
    Returns:
        dict: A dictionary mapping class labels to indices.
    """
    # Extract unique class labels from
    # the target column and create a mapping
    # from class label to index.
    data_class = data[target_column].explode().unique().sort().to_list()
    class_map = {data_class[i]: i for i in range(len(data_class))}
    return class_map


def pad_truncate_tensor(
    tensor: torch.Tensor, max_length: int, pad_token: int
) -> torch.Tensor:
    """
    Truncate or pad a tensor to a fixed length.

    Args:
        tensor (torch.Tensor): The input tensor to process.
        max_length (int): The desired fixed length.
        pad_token (int): The token value used for padding.

    Returns:
        torch.Tensor: The truncated or padded tensor.
    """
    current_length = tensor.size(0)
    if current_length > max_length:
        # Truncate the tensor
        return tensor[:max_length]
    elif current_length < max_length:
        # Create a padding tensor
        padding = torch.full(
            (max_length - current_length,), pad_token, dtype=tensor.dtype
        )
        # Concatenate the original tensor with padding
        return torch.cat([tensor, padding], dim=0)
    else:
        return tensor


def one_hot_encode(target: list, class_map: dict) -> list:
    """
    One-hot encode a list of target labels.

    Args:
        target (list): The list of target labels to encode.
        class_map (dict): The mapping from target labels to indices.

    Returns:
        list: The one-hot encoded list.
    """
    num_classes = len(class_map)
    one_hot_list = [0] * num_classes
    target = set(target)
    for label in target:
        idx = class_map.get(label)
        if idx is not None:
            one_hot_list[idx] = 1
    return one_hot_list


class MIMIC_Dataset(Dataset):
    """Dataset of tokenised notes and ICD targets stored in Parquet files."""

    def __init__(
        self,
        data_dir,
        # pad_token=128004,
    ):
        self.data = pl.read_parquet(data_dir)
        # self.pad_token = pad_token
        data_path = Path(data_dir)
        self._first_index_column = (
            "first_diagnosis_index"
            if "first_diagnosis_index" in self.data.columns
            else None
        )
        self._has_id = ID_COLUMN in self.data.columns
        self._first_code_column = None
        if self._first_index_column is None:
            if "first_diagnosis_code" in self.data.columns:
                self._first_code_column = "first_diagnosis_code"

        class_map_path = data_path.with_name("class_map.parquet")
        self._class_to_index = _load_class_map(class_map_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token = torch.tensor(self.data[idx]["tokens"][0])
        # token = token.squeeze(0)
        class_label = torch.tensor(self.data[idx]["target_tensor"][0])
        class_label = class_label.float()
        last_token_index = torch.tensor(self.data[idx]["last_token_index"][0])
        first_diagnosis_index = self._resolve_first_diagnosis_index(idx)
        val = _unwrap_field(self.data[idx][ID_COLUMN]) if self._has_id else None
        ids = str(val) if val is not None else ""
        return (
            token,
            last_token_index,
            class_label,
            torch.tensor(first_diagnosis_index, dtype=torch.long),
            ids,
        )

    def _resolve_first_diagnosis_index(self, idx: int) -> int:
        if self._first_index_column is not None:
            value = _unwrap_field(self.data[idx][self._first_index_column])
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, list) and value:
                return int(value[0])
            return -1

        if self._first_code_column is None or not self._class_to_index:
            return -1

        value = _unwrap_field(self.data[idx][self._first_code_column])
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, str):
            return int(self._class_to_index.get(value, -1))
        return -1


def _collate_with_ids(batch):
    """Collate tuples of (tensor, tensor, tensor, tensor, str)."""
    tokens = torch.stack([item[0] for item in batch])
    last_token_indices = torch.stack([item[1] for item in batch])
    class_labels = torch.stack([item[2] for item in batch])
    first_diagnosis_indices = torch.stack([item[3] for item in batch])
    ids = [item[4] for item in batch]
    return tokens, last_token_indices, class_labels, first_diagnosis_indices, ids


def get_dataloaders(
    data_dir,
    batch_size=2,
    num_workers=4,
    pin_memory=True,
    seed: int | None = None,
):
    """Create train/val/test dataloaders from saved Parquet datasets.

    Parameters
    ----------
    data_dir:
        Directory containing ``train_dataset.parquet``, ``val_dataset.parquet`` and
        ``test_dataset.parquet``.
    batch_size:
        Batch size for each loader.
    num_workers:
        Number of worker processes used to load data.
    pin_memory:
        Whether to copy Tensors into CUDA pinned memory.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Training, validation and test dataloaders.
    """
    train_dataset = MIMIC_Dataset(Path(f"{data_dir}/train_dataset.parquet"))
    val_dataset = MIMIC_Dataset(Path(f"{data_dir}/val_dataset.parquet"))
    test_dataset = MIMIC_Dataset(Path(f"{data_dir}/test_dataset.parquet"))

    for name, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ]:
        if len(dataset) == 0:
            raise ValueError(
                f"{name}_dataset.parquet is empty (0 rows). "
                f"Re-run the 'prepare' DVC stage."
            )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=_collate_with_ids,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=_collate_with_ids,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=_collate_with_ids,
    )
    return train_loader, val_loader, test_loader
