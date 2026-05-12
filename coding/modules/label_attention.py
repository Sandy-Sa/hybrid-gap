import torch
from torch import nn
from torch.nn import functional as F
from coding.modules.base import LitLLM, LowRankAdaptationConfig
from typing import Optional

WEIGHT_INIT_MEAN = 0.0
WEIGHT_INIT_STD = 0.03


class LabelAttention(nn.Module):
    """Label-wise attention mechanism inspired by the LAAT model."""

    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = F.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(
        self, mean: float = WEIGHT_INIT_MEAN, std: float = WEIGHT_INIT_STD
    ) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        nn.init.normal_(self.first_linear.weight, mean, std)
        nn.init.normal_(self.second_linear.weight, mean, std)
        nn.init.normal_(self.third_linear.weight, mean, std)


class LLMLabelAttention(LitLLM):
    """Lightning module that applies label attention before classification."""

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
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            n_classes=n_classes,
            hidden_state_index=hidden_state_index,
            threshold=threshold,
            classifier_dropout=classifier_dropout,
            loss_weights=loss_weights,
            optimizer_hparams=optimizer_hparams,
            batch_size=batch_size,
            update_batch=update_batch,
            max_epochs=max_epochs,
            data_size=data_size,
            unfreeze=unfreeze,
            n_blocks=n_blocks,
            lora_config=lora_config,
            code_index_groups=code_index_groups,
            log_code_types=log_code_types,
        )
        self.save_hyperparameters()
        self.attention = LabelAttention(
            input_size=self.llm.config.hidden_size,
            projection_size=self.llm.config.hidden_size,
            num_classes=n_classes,
        )

    def forward(self, input_ids, last_token_index, attention_mask=None):
        """Compute logits using label attention over token representations."""

        # Extract the hidden state corresponding to the last token
        hidden_state = self.get_hidden_state(
            input_ids=input_ids,
            last_token_index=last_token_index,
            attention_mask=attention_mask,
        )
        # Apply the classification head to the hidden states
        logits = self.attention(hidden_state)
        return logits
