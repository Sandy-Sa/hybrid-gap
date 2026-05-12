import torch
from lightning.pytorch import LightningModule
from transformers import AutoModel
import math
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig
from typing import Optional, Callable
from dataclasses import dataclass
import json

# from coding.modules.attention import LabelAttention
from torchmetrics.functional.classification import (
    multilabel_accuracy,
    multilabel_auroc,
    multilabel_average_precision,
    multilabel_f1_score,
    multilabel_precision,
    multilabel_recall,
)


@dataclass
class LowRankAdaptationConfig:
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    r: int = 64
    bias: str = "none"
    target_modules: list[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class MetricSpec:
    """Specification for a metric to log during training/validation/testing."""

    name: str
    fn: Callable
    average: str = "micro"
    prog_bar: bool = False
    on_step: bool = False
    on_epoch: bool = True
    stages: tuple[str, ...] = ("train", "val", "test")


def get_scheduler(optimizer, lr, data_size, max_epochs, update_batch):
    """Configure a OneCycle learning rate scheduler.

    Parameters
    ----------
    optimizer:
        Optimiser whose learning rate will be scheduled.
    lr:
        Peak learning rate during training.
    data_size:
        Number of training examples.
    max_epochs:
        Total number of epochs to run.
    update_batch:
        Effective batch size after gradient accumulation.

    Returns
    -------
    torch.optim.lr_scheduler.OneCycleLR
        Scheduler configured for the given training run.
    """
    # Number of optimizer steps per epoch equals the number of effective-batch
    # updates; use ceil to account for a final partial update each epoch.
    steps_per_epoch = math.ceil(data_size / update_batch)
    max_steps = steps_per_epoch * max_epochs
    min_lr = lr / 10

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    return scheduler


class LitLLM(LightningModule):
    """Base Lightning module wrapping a pretrained language model for multilabel classification."""

    STANDARD_METRICS = [
        MetricSpec(
            "f1_micro", multilabel_f1_score, "micro", prog_bar=True, on_step=True
        ),
        MetricSpec("f1_macro", multilabel_f1_score, "macro"),
        MetricSpec(
            "accuracy",
            multilabel_accuracy,
            "micro",
            prog_bar=True,
            on_step=True,
            stages=("train",),
        ),
        MetricSpec("accuracy", multilabel_accuracy, "micro", stages=("val", "test")),
        MetricSpec("precision_micro", multilabel_precision, "micro"),
        MetricSpec(
            "precision_macro", multilabel_precision, "macro", stages=("val", "test")
        ),
        MetricSpec("recall_micro", multilabel_recall, "micro"),
        MetricSpec("auroc_micro", multilabel_auroc, "micro", stages=("test",)),
        MetricSpec("auroc_macro", multilabel_auroc, "macro", stages=("test",)),
        MetricSpec(
            "prc_micro", multilabel_average_precision, "micro", stages=("test",)
        ),
        MetricSpec(
            "prc_macro", multilabel_average_precision, "macro", stages=("test",)
        ),
    ]

    def __init__(
        self,
        checkpoint_dir: str,
        n_classes=50,
        hidden_state_index=-1,
        threshold=0.5,
        classifier_dropout=0.1,
        loss_weights: Optional[str] = None,
        optimizer_hparams: dict = None,
        batch_size=2,
        update_batch=64,
        max_epochs=30,
        data_size=31571,
        unfreeze=None,
        n_blocks=0,
        lora_config: Optional[LowRankAdaptationConfig] = None,
        code_index_groups: dict[str, list[int]] | None = None,
        log_code_types: str = "all",
    ):
        super().__init__()
        # self.save_hyperparameters()
        # Load the pre-trained language model
        self.llm = AutoModel.from_pretrained(checkpoint_dir)
        self.hidden_size = self.llm.config.hidden_size
        self.threshold = threshold
        self.n_classes = n_classes
        self.dropout = torch.nn.Dropout(classifier_dropout)
        if hidden_state_index < 0:
            self.hidden_state_index = (
                hidden_state_index + self.llm.config.num_hidden_layers
            )
        else:
            self.hidden_state_index = hidden_state_index

        n_pop = max(0, self.llm.config.num_hidden_layers - self.hidden_state_index - 1)

        for i in range(n_pop):
            self.llm.base_model.layers.pop(-1)

        # Loss function
        self.loss_module = torch.nn.BCEWithLogitsLoss(reduction="none")
        if loss_weights is not None and loss_weights not in {"none", "first_diagnosis"}:
            raise ValueError(
                "loss_weights must be one of None, 'none', or 'first_diagnosis'."
            )
        self.loss_weights = loss_weights
        allowed_logging = {"all", "full", "diag", "proc"}
        log_code_types = log_code_types.lower()
        if log_code_types not in allowed_logging:
            raise ValueError(
                "log_code_types must be one of 'all', 'full', 'diag', or 'proc'."
            )

        self.log_code_types = log_code_types
        self.code_index_groups = code_index_groups or {}

        # freeze all parameters by default
        for param in self.llm.parameters():
            param.requires_grad = False

        print(f"Unfreezing {unfreeze} layers\nn_blocks: {n_blocks}")

        if unfreeze == "all":
            # Unfreeze all layers
            for param in self.llm.parameters():
                param.requires_grad = True

        elif unfreeze == "last":
            # Unfreeze only the last block
            for param in self.llm.base_model.layers[-1].parameters():
                param.requires_grad = True

            # Unfreeze the last n blocks
            for i in range(n_blocks):
                for param in self.llm.base_model.layers[-1 - i].parameters():
                    param.requires_grad = True
        elif unfreeze == "lora":
            if lora_config is None:
                raise ValueError(
                    "lora_config must be provided when unfreezing 'lora' layers."
                )
            lora_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
            )

            self.llm = get_peft_model(self.llm, lora_config)
        elif unfreeze == "none":
            # keep all layers frozen
            pass

    def get_hidden_state(self, input_ids, last_token_index, attention_mask=None):
        # Get outputs from the LLM, including hidden states
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states  # Tuple of hidden states
        # Select the desired hidden state (default is the last one)
        hidden_state = hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]

        hidden_state = self.dropout(hidden_state)

        return hidden_state

    def _precision_at_k(
        self, preds: torch.Tensor, labels: torch.Tensor, k: int
    ) -> torch.Tensor:
        num_labels = preds.size(1)
        if num_labels == 0:
            return torch.tensor(0.0, device=preds.device)
        k = min(k, num_labels)
        if k == 0:
            return torch.tensor(0.0, device=preds.device)

        topk_indices = torch.topk(preds, k, dim=1).indices
        label_k = torch.gather(labels, 1, topk_indices)
        pred_k = torch.ones(label_k.shape, device=preds.device)

        tp = torch.sum(label_k * pred_k, dim=1)
        fp = torch.sum((1 - label_k) * pred_k, dim=1)
        precision = tp / (tp + fp + 1e-10)

        return precision.mean()

    def _recall_at_k(
        self, preds: torch.Tensor, labels: torch.Tensor, k: int
    ) -> torch.Tensor:
        num_labels = preds.size(1)
        if num_labels == 0:
            return torch.tensor(0.0, device=preds.device)
        k = min(k, num_labels)
        if k == 0:
            return torch.tensor(0.0, device=preds.device)

        topk_indices = torch.topk(preds, k, dim=1).indices
        label_k = torch.gather(labels, 1, topk_indices)
        pred_k = torch.ones(label_k.shape, device=preds.device)

        tp = torch.sum(label_k * pred_k, dim=1)
        total_relevant = torch.sum(labels, dim=1)

        recall = tp / (total_relevant + 1e-10)
        return recall.mean()

    def _exact_match_ratio(
        self, preds: torch.Tensor, labels: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        binary_preds = (preds >= threshold).to(labels.dtype)
        matches = torch.all(torch.eq(binary_preds, labels), dim=1).sum()

        return matches.float().mean()

    def _r_precision(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = preds.device
        scores = []
        for sample_pred, sample_target in zip(preds, labels):
            num_relevant = int(sample_target.sum().item())
            if num_relevant == 0:
                scores.append(torch.tensor(0.0, device=device))
                continue
            topk_indices = torch.topk(sample_pred, num_relevant).indices
            retrieved_relevant = sample_target[topk_indices].sum()
            scores.append(retrieved_relevant / num_relevant)
        if not scores:
            return torch.tensor(0.0, device=device)
        return torch.stack(scores).mean()

    def _mean_average_precision(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        device = preds.device
        ap_scores = []
        for sample_pred, sample_target in zip(preds, labels):
            num_relevant = int(sample_target.sum().item())
            if num_relevant == 0:
                ap_scores.append(torch.tensor(0.0, device=device))
                continue
            sorted_indices = torch.argsort(sample_pred, descending=True)
            sorted_targets = sample_target[sorted_indices]
            relevant_positions = torch.nonzero(
                sorted_targets > 0, as_tuple=False
            ).squeeze(-1)
            if relevant_positions.numel() == 0:
                ap_scores.append(torch.tensor(0.0, device=device))
                continue
            if relevant_positions.dim() == 0:
                relevant_positions = relevant_positions.unsqueeze(0)
            ranks = relevant_positions.float() + 1.0
            precision_at_hits = (
                torch.arange(
                    1,
                    relevant_positions.numel() + 1,
                    dtype=torch.float32,
                    device=device,
                )
                / ranks
            )
            ap_scores.append(precision_at_hits.sum() / num_relevant)
        if not ap_scores:
            return torch.tensor(0.0, device=device)
        return torch.stack(ap_scores).mean()

    def _compute_first_diagnosis_metrics(
        self,
        preds: torch.Tensor,
        first_indices: torch.Tensor | None,
        candidate_indices: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if first_indices is None or preds.ndim != 2:
            return {}

        num_samples, num_classes = preds.shape
        if first_indices.numel() != num_samples:
            return {}

        device = preds.device
        indices = first_indices.to(device=device, dtype=torch.long).view(-1)
        valid_mask = (indices >= 0) & (indices < num_classes)

        candidate_tensor = None
        if candidate_indices:
            candidate_tensor = torch.tensor(
                candidate_indices, dtype=torch.long, device=device
            )
            if candidate_tensor.numel() == 0:
                zero = torch.tensor(0.0, device=device)
                return {
                    "first_diagnosis_accuracy": zero,
                    "first_diagnosis_top1": zero,
                }
            membership_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
            membership_mask[candidate_tensor] = True
            safe_indices = indices.clone()
            safe_indices[safe_indices < 0] = 0
            valid_mask &= membership_mask[safe_indices]

        if not torch.any(valid_mask):
            zero = torch.tensor(0.0, device=device)
            return {
                "first_diagnosis_accuracy": zero,
                "first_diagnosis_top1": zero,
            }

        row_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1).to(device)
        selected_indices = indices.index_select(0, row_indices)
        selected_preds = preds.index_select(0, row_indices)

        gathered_scores = selected_preds.gather(
            1, selected_indices.unsqueeze(1)
        ).squeeze(1)
        accuracy = (gathered_scores >= self.threshold).float().mean()

        if candidate_tensor is not None:
            candidate_preds = selected_preds.index_select(1, candidate_tensor)
            top_positions = candidate_preds.argmax(dim=1)
            top_global_indices = candidate_tensor.index_select(0, top_positions)
        else:
            top_global_indices = selected_preds.argmax(dim=1)

        top1 = (top_global_indices == selected_indices).float().mean()

        return {
            "first_diagnosis_accuracy": accuracy,
            "first_diagnosis_top1": top1,
        }

    def _log_standard_metrics(
        self,
        stage: str,
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Log all standard multilabel classification metrics for a given stage."""
        num_labels = preds.size(1)
        for spec in self.STANDARD_METRICS:
            if stage not in spec.stages:
                continue
            if spec.fn == multilabel_auroc or spec.fn == multilabel_average_precision:
                value = spec.fn(
                    preds, labels, num_labels=num_labels, average=spec.average
                )
            else:
                value = spec.fn(
                    preds,
                    labels,
                    num_labels=num_labels,
                    threshold=self.threshold,
                    average=spec.average,
                )
            self.log(
                f"{stage}_{spec.name}",
                value,
                on_step=spec.on_step if stage == "train" else False,
                on_epoch=spec.on_epoch,
                prog_bar=spec.prog_bar if stage in ("train", "val") else False,
            )

    def _log_test_custom_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Log test-only custom metrics (precision@k, recall@k, etc.)."""
        custom_metrics = [
            ("precision_at_8", self._precision_at_k(preds, labels, k=8)),
            ("precision_at_15", self._precision_at_k(preds, labels, k=15)),
            ("recall_at_8", self._recall_at_k(preds, labels, k=8)),
            ("recall_at_15", self._recall_at_k(preds, labels, k=15)),
            ("emr", self._exact_match_ratio(preds, labels, threshold=self.threshold)),
            ("r_precision", self._r_precision(preds, labels)),
            ("map", self._mean_average_precision(preds, labels)),
        ]
        for name, value in custom_metrics:
            self.log(
                f"test_{name}", value, on_step=False, on_epoch=True, prog_bar=False
            )

    def _compute_loss(
        self,
        y_hat: torch.Tensor,
        labels: torch.Tensor,
        first_diagnosis_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the loss with optional weighting for first diagnosis."""
        base_loss = self.loss_module(y_hat, labels)
        if self.loss_weights is None or self.loss_weights == "none":
            return base_loss.mean()
        elif self.loss_weights == "first_diagnosis" and first_diagnosis_indices is not None:
            weight = torch.ones_like(labels)
            idx = first_diagnosis_indices.to(device=labels.device, dtype=torch.long)
            valid = (idx >= 0) & (idx < labels.size(1))
            rows = torch.arange(labels.size(0), device=labels.device)
            weight[rows[valid], idx[valid]] = 2
            return (base_loss * weight).sum() / weight.sum()
        else:
            raise ValueError(f"Unknown loss_weights type: {self.loss_weights}")

    def _log_first_diagnosis_metrics(
        self,
        preds: torch.Tensor,
        first_diagnosis_indices: torch.Tensor | None,
    ) -> None:
        """Log first diagnosis metrics during test stage."""
        if first_diagnosis_indices is None:
            return

        if not isinstance(first_diagnosis_indices, torch.Tensor):
            first_diagnosis_indices = torch.tensor(
                first_diagnosis_indices,
                dtype=torch.long,
                device=preds.device,
            )
        first_diagnosis_indices_detached = (
            first_diagnosis_indices.detach()
            .to(device=preds.device, dtype=torch.long)
            .view(-1)
        )

        candidate_indices = self.code_index_groups.get("diagnosis")
        primary_metrics = self._compute_first_diagnosis_metrics(
            preds, first_diagnosis_indices_detached, candidate_indices
        )
        if primary_metrics:
            self.log(
                "test_first_diagnosis_accuracy",
                primary_metrics["first_diagnosis_accuracy"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "test_first_diagnosis_top1",
                primary_metrics["first_diagnosis_top1"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def _store_test_predictions(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        last_token_index: torch.Tensor,
        first_diagnosis_indices: torch.Tensor | None,
        ids: list[str] | None = None,
    ) -> None:
        """Store test predictions for later analysis."""
        self._test_preds.append(preds.cpu())
        self._test_labels.append(labels.cpu())
        self._test_last_token_idxs.append(last_token_index.cpu())
        if first_diagnosis_indices is not None:
            if not isinstance(first_diagnosis_indices, torch.Tensor):
                first_diagnosis_indices = torch.tensor(
                    first_diagnosis_indices, dtype=torch.long, device=preds.device
                )
            self._test_first_diag.append(
                first_diagnosis_indices.detach().to(dtype=torch.long).view(-1).cpu()
            )

        if ids is not None:
            self._test_id.extend(ids)

    def get_scores(self, batch, stage: str = "train") -> torch.Tensor | None:
        """Compute loss and log metrics for a batch.

        Parameters
        ----------
        batch : tuple
            Tuple of (tokens, last_token_index, labels, first_diagnosis_indices, ids).
        stage : str
            One of "train", "val", or "test".

        Returns
        -------
        torch.Tensor | None
            Loss tensor for training stage, None otherwise.
        """
        token, last_token_index, labels, first_diagnosis_indices, ids = batch

        y_hat = self.forward(input_ids=token, last_token_index=last_token_index)
        preds = torch.sigmoid(y_hat)
        loss = self._compute_loss(y_hat, labels, first_diagnosis_indices)

        preds_detached = preds.detach()
        labels_int = labels.detach().to(device=preds_detached.device, dtype=torch.int)

        # Log loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
        )

        # Log standard metrics
        self._log_standard_metrics(stage, preds_detached, labels_int)

        # Test-only: custom metrics and predictions storage
        if stage == "test":
            self._log_test_custom_metrics(preds_detached, labels_int)
            self._log_first_diagnosis_metrics(preds_detached, first_diagnosis_indices)
            self._store_test_predictions(
                preds_detached,
                labels.detach(),
                last_token_index,
                first_diagnosis_indices,
                ids,
            )

        return loss if stage == "train" else None

    def training_step(self, batch, attention_mask):

        loss = self.get_scores(batch, stage="train")

        return loss

    def validation_step(self, batch):
        self.get_scores(batch, stage="val")

        # for section_name in ("diagnosis", "procedure"):
        #     if section_name in metrics and self._should_log_section(section_name):
        #         self._log_metrics_for_stage(
        #             "val", metrics[section_name], section=section_name
        #         )

    def on_test_start(self):
        self._test_preds = []
        self._test_labels = []
        self._test_last_token_idxs = []
        self._test_first_diag = []
        self._test_id = []

    def test_step(self, batch):
        self.get_scores(batch, stage="test")

        # for section_name in ("diagnosis", "procedure"):
        #     if section_name in metrics and self._should_log_section(section_name):
        #         self._log_metrics_for_stage(
        #             "test", metrics[section_name], section=section_name
        #         )

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds).tolist()
        labels = torch.cat(self._test_labels).tolist()
        last_token_idxs = torch.cat(self._test_last_token_idxs).tolist()
        first_diag = (
            torch.cat(self._test_first_diag).tolist() if self._test_first_diag else None
        )
        with open("models/predictions.json", "w") as f:
            json.dump(
                {
                    "labels": labels,
                    "predictions": preds,
                    "last_token_indices": last_token_idxs,
                    "first_diagnosis_indices": first_diag,
                    "ids": self._test_id if any(self._test_id) else None,
                },
                f,
            )

    def configure_optimizers(
        self,
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.hparams.optimizer_hparams.weight_decay,
            },
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.optimizer_hparams.lr,
            betas=self.hparams.optimizer_hparams.betas,
        )

        scheduler = get_scheduler(
            optimizer,
            lr=self.hparams.optimizer_hparams.lr,
            data_size=self.hparams.data_size,
            max_epochs=self.hparams.max_epochs,
            update_batch=self.hparams.update_batch,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
