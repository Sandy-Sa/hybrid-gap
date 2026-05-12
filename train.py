from pathlib import Path
import torch
import os
import polars as pl
from coding.data.data_prep import get_dataloaders
from coding.modules.base import LowRankAdaptationConfig, OptimiserConfig
from coding.modules.classifier import LLMClassifier
from coding.modules.label_attention import LLMLabelAttention
from coding.modules.label_cross_attention import LLMLabelCrossAttention
from coding.utils.reproducibility import DEFAULT_SEED, set_global_seed
from peft import LoraConfig
import wandb

# import mlflow
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger

# import lightning.pytorch as pl
from jsonargparse import CLI
from dataclasses import dataclass
from typing import Optional

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


@dataclass
class WandbConfig:
    project: str = "study2_gadi"
    save_dir: str = "/scratch/gp50/zs5704/projects/llm-coding/.wandb/"


# @dataclass
# class SchedulerConfig:
#     T_max: int = 10
#     eta_min: float = 1e-6


def _collect_code_groups(loaders, class_map_path: Path) -> dict[str, list[int]]:
    if not class_map_path.exists():
        return {}

    class_map_df = pl.read_parquet(class_map_path)
    code_to_index = dict(zip(class_map_df["key"], class_map_df["value"]))

    code_groups: dict[str, set[int]] = {"diagnosis": set(), "procedure": set()}

    def _update_group(dataset, column: str, group_key: str) -> None:
        if column not in dataset.columns:
            return
        series = dataset.get_column(column)
        for codes in series:
            if codes is None:
                continue
            code_groups[group_key].update(
                code_to_index[code] for code in codes if code in code_to_index
            )

    for loader in loaders:
        data = loader.dataset.data
        _update_group(data, "diagnoses_code", "diagnosis")
        _update_group(data, "procedure_code", "procedure")

    return {name: sorted(indices) for name, indices in code_groups.items() if indices}


def train(
    model_name: str = "meta-llama/Llama-3.2-1B",
    ckp_path: Optional[str] = None,
    data_path: Path = Path("data/datasets/"),
    module: str = "label_attention",
    max_epochs: int = 15,
    optimizer: OptimiserConfig = OptimiserConfig(),
    threshold: float = 0.5,
    batch_size: int = 2,
    update_batch: int = 64,
    dropout: float = 0.1,
    loss_weights: Optional[str] = None,
    hidden_state_index: int = -1,
    unfreeze="last",
    n_blocks=0,
    log_code_types: str = "all",
    # scheduler_name: str = 'CosineAnnealingLR',
    # scheduler_hparams: SchedulerConfig = SchedulerConfig(),
    lora_config: Optional[LowRankAdaptationConfig] = None,
    logger: Optional[WandbConfig] = None,
    seed: int = DEFAULT_SEED,
    limit_train_batches: Optional[float] = None,
    limit_val_batches: Optional[float] = None,
    max_steps: int = -1,
):
    """Fine-tune a language model to predict ICD codes.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier or checkpoint path.
    data_path:
        Directory containing ``train_dataset.parquet``, ``val_dataset.parquet`` and
        ``test_dataset.parquet``.
    module:
        Model head to use (``"label_attention"`` or ``"classifier"``).
    max_epochs:
        Number of training epochs.
    optimizer:
        Hyperparameter configuration for the optimiser.
    threshold:
        Decision threshold applied when computing metrics.
    batch_size:
        Per-step batch size.
    update_batch:
        Effective batch size after gradient accumulation.
    dropout:
        Dropout probability for classifier layers.
    hidden_state_index:
        Which hidden layer of the base model to use.
    unfreeze:
        Strategy for unfreezing pretrained layers (``"last"`` or ``"all"``).
    n_blocks:
        Additional number of blocks to unfreeze when using ``"last"``.
    log_code_types:
        Controls which code group metrics are logged. ``"all"`` logs full, diagnosis,
        and procedure metrics. ``"full"`` logs only the combined metrics, while
        ``"diag"`` and ``"proc"`` log only the diagnosis or procedure metrics,
        respectively.
    logger:
        Logger type, e.g., "wandb" for Weights & Biases logger. Set to ``None`` to disable logging.
    seed:
        Random seed controlling model initialisation, data shuffling, and
        Lightning worker seeds for reproducible training runs.
    """
    set_global_seed(seed)
    L.seed_everything(seed, workers=True)

    if ckp_path is not None and ckp_path.lower() == "none":
        ckp_path = None

    allowed_logging = {"all", "full", "diag", "proc"}
    log_code_types = log_code_types.lower()
    if log_code_types not in allowed_logging:
        raise ValueError(
            "log_code_types must be one of 'all', 'full', 'diag', or 'proc'."
        )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    use_cuda = True if torch.cuda.is_available() else False
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Some data arguments.
    NUM_WORKERS = int(os.cpu_count() / 2)
    kwargs = {"num_workers": NUM_WORKERS, "pin_memory": True} if use_cuda else {}
    kwargs["batch_size"] = batch_size  # Batch size used during training.

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath="models",
        filename="model",
    )
    es = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    if logger is not None:
        pl_logger = WandbLogger(
            project=logger.project,
            save_dir=logger.save_dir,
        )
    else:
        pl_logger = None
    # mlflow.pytorch.autolog()
    num_gpus = torch.cuda.device_count()
    accumulate_grad_batches = update_batch // (batch_size * num_gpus)
    if accumulate_grad_batches < 1:
        raise ValueError(
            f"update_batch ({update_batch}) must be >= batch_size * num_gpus "
            f"({batch_size} * {num_gpus} = {batch_size * num_gpus})"
        )

    trainer_config = {
        "accelerator": "gpu",
        "devices": num_gpus,
        "strategy": "ddp" if num_gpus > 1 else "auto",
        "logger": pl_logger,
        "precision": "bf16-true",
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "accumulate_grad_batches": accumulate_grad_batches,
        "callbacks": [checkpoint_callback, LearningRateMonitor("epoch"), es],
        "deterministic": True,
        "profiler": "simple",
    }
    if limit_train_batches is not None:
        trainer_config["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        trainer_config["limit_val_batches"] = limit_val_batches
    trainer = L.Trainer(**trainer_config)

    # get datasets
    train_loader, val_loader, _ = get_dataloaders(
        data_path, seed=seed, **kwargs
    )

    n_classes = len(train_loader.dataset.data["target_tensor"][0])

    code_index_groups = _collect_code_groups(
        (train_loader, val_loader), data_path / "class_map.parquet"
    )

    model_kwargs = {
        "checkpoint_dir": model_name,
        "n_classes": n_classes,
        "hidden_state_index": hidden_state_index,
        "threshold": threshold,
        "classifier_dropout": dropout,
        "loss_weights": loss_weights,
        "optimizer_hparams": optimizer,
        "batch_size": batch_size,
        "update_batch": update_batch,
        "max_epochs": max_epochs,
        "data_size": len(train_loader.dataset),
        "unfreeze": unfreeze,
        "n_blocks": n_blocks,
        "lora_config": lora_config,
        "code_index_groups": code_index_groups,
        "log_code_types": log_code_types,
    }
    if module == "classifier":
        model = LLMClassifier(**model_kwargs)
    elif module == "label_attention":
        model = LLMLabelAttention(**model_kwargs)
    elif module == "label_cross_attention":
        model = LLMLabelCrossAttention(**model_kwargs)
    else:
        assert False, f"Unknown model: {module}"
    print(f"Using model: {model.__class__.__name__}")

    trainer.fit(
        model,
        ckpt_path=ckp_path,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    CLI(train)
