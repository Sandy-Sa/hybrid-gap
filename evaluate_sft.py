"""Evaluate the trained SFT classifier.

Sweeps decision thresholds on the validation set to pick the best
micro/macro F1 thresholds, then dumps full test-set predictions for
downstream LLM gap-fill stages.

Writes into ``--data_path``:
    - val_predictions.json     (intermediate; raw val probs + labels)
    - optimal_thresholds.json  {"micro": float, "macro": float}
    - val_metrics.json         per-threshold F1 + chosen thresholds
    - predictions.json         test-set predictions consumed by verification_sft
"""
import json
import os
from pathlib import Path

import lightning as L
import torch
from jsonargparse import CLI
from torchmetrics.functional.classification import multilabel_f1_score

from coding.data.data_prep import get_dataloaders
from coding.modules.classifier import LLMClassifier
from coding.modules.label_attention import LLMLabelAttention
from coding.modules.label_cross_attention import LLMLabelCrossAttention
from coding.utils.reproducibility import DEFAULT_SEED, set_global_seed


MODULE_CLASSES = {
    "classifier": LLMClassifier,
    "label_attention": LLMLabelAttention,
    "label_cross_attention": LLMLabelCrossAttention,
}


def _sweep_thresholds(preds: torch.Tensor, labels: torch.Tensor, num: int = 99):
    """Sweep thresholds in (0, 1) and return best micro/macro F1 plus per-step metrics."""
    thresholds = torch.linspace(0.01, 0.99, num)
    n_labels = preds.size(1)
    per_t = []
    best = {"micro": (0.5, -1.0), "macro": (0.5, -1.0)}
    for t in thresholds:
        t_val = float(t.item())
        f1_mi = float(
            multilabel_f1_score(
                preds, labels, num_labels=n_labels, threshold=t_val, average="micro"
            ).item()
        )
        f1_ma = float(
            multilabel_f1_score(
                preds, labels, num_labels=n_labels, threshold=t_val, average="macro"
            ).item()
        )
        per_t.append({"threshold": t_val, "f1_micro": f1_mi, "f1_macro": f1_ma})
        if f1_mi > best["micro"][1]:
            best["micro"] = (t_val, f1_mi)
        if f1_ma > best["macro"][1]:
            best["macro"] = (t_val, f1_ma)
    return best, per_t


def evaluate_sft(
    data_path: Path = Path("data/datasets/"),
    ckpt_path: Path = Path("models/model.ckpt"),
    module: str = "label_attention",
    batch_size: int = 8,
    seed: int = DEFAULT_SEED,
):
    """Run validation-set threshold sweep and full test-set prediction dump.

    Parameters
    ----------
    data_path:
        Directory containing val/test parquets plus class_map.parquet (i.e. the
        output of the `prepare` stage). All outputs are written back here.
    ckpt_path:
        Lightning checkpoint produced by `train.py`.
    module:
        Module head used at training time (`classifier`, `label_attention`, or
        `label_cross_attention`). Must match the class that produced the ckpt.
    batch_size:
        Inference batch size; can differ from training batch size.
    seed:
        Random seed for dataloader determinism.
    """
    set_global_seed(seed)
    L.seed_everything(seed, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_path = Path(data_path)
    ckpt_path = Path(ckpt_path)

    if module not in MODULE_CLASSES:
        raise ValueError(
            f"Unknown module: {module!r}; expected one of {sorted(MODULE_CLASSES)}"
        )
    model_cls = MODULE_CLASSES[module]

    use_cuda = torch.cuda.is_available()
    num_workers = max(1, int(os.cpu_count() / 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "batch_size": batch_size,
    }
    _, val_loader, test_loader = get_dataloaders(data_path, seed=seed, **loader_kwargs)

    model = model_cls.load_from_checkpoint(ckpt_path)
    model.eval()

    num_gpus = torch.cuda.device_count() if use_cuda else 0
    trainer_config = {
        "accelerator": "gpu" if use_cuda else "cpu",
        "devices": num_gpus if use_cuda else 1,
        "logger": False,
        "precision": "bf16-true" if use_cuda else 32,
        "deterministic": True,
    }
    trainer = L.Trainer(**trainer_config)

    # --- Validation: dump preds, then sweep thresholds ---
    val_pred_path = data_path / "val_predictions.json"
    model.predictions_path = val_pred_path
    trainer.test(model, dataloaders=val_loader)

    with open(val_pred_path) as f:
        val_raw = json.load(f)
    val_preds = torch.tensor(val_raw["predictions"])
    val_labels = torch.tensor(val_raw["labels"]).int()
    best, per_threshold = _sweep_thresholds(val_preds, val_labels)
    optimal = {"micro": best["micro"][0], "macro": best["macro"][0]}
    val_metrics = {
        "optimal_thresholds": optimal,
        "f1_at_optimal": {"micro": best["micro"][1], "macro": best["macro"][1]},
        "per_threshold": per_threshold,
    }
    (data_path / "optimal_thresholds.json").write_text(json.dumps(optimal, indent=2))
    (data_path / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))

    # --- Test: dump full predictions to predictions.json ---
    model.predictions_path = data_path / "predictions.json"
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    CLI(evaluate_sft)
