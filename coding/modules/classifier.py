import torch
from typing import Optional
from coding.modules.base import LitLLM


class LLMClassifier(LitLLM):
    """Linear classification head applied to the final token representation."""

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
        lora_config=None,
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
        self.classification = torch.nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, input_ids, last_token_index, attention_mask=None):
        """Compute logits from the hidden state of the last token."""

        # Extract the hidden state corresponding to the last token
        hidden_state = self.get_hidden_state(
            input_ids=input_ids,
            last_token_index=last_token_index,
            attention_mask=attention_mask,
        )
        batch_size = hidden_state.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_state.device)
        last_token_hidden_state = hidden_state[batch_indices, last_token_index, :]

        # add dropout
        output = self.dropout(last_token_hidden_state)

        # Apply the classification head to the hidden states
        logits = self.classification(output)  # Shape: [batch_size, seq_len, n_classes]
        return logits
