"""Label Cross-Attention module following Edin et al. (2024) / codeseeker style.

This implements cross-attention between all tokens and a learned representation per code,
as described in the "Code Like Humans" paper.
"""

import torch
from torch import nn
from typing import Optional

from coding.modules.base import LitLLM, LowRankAdaptationConfig

WEIGHT_INIT_MEAN = 0.0
WEIGHT_INIT_STD = 0.03


class LabelCrossAttention(nn.Module):
    """Cross-attention between token representations and learned label embeddings.

    Each ICD code has a learned query vector that attends over all token hidden states.
    This follows the PLM-ICD / Edin et al. (2024) approach.

    Args:
        input_size: Hidden dimension of the LLM (e.g., 2048 for Llama-3.2-1B)
        num_classes: Number of ICD codes to predict
        use_flash_attention: Whether to use scaled_dot_product_attention (faster, less memory)
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_flash_attention = use_flash_attention

        # Learned query vector per label (the "learned representation per code")
        self.label_representations = nn.Parameter(
            torch.empty(num_classes, input_size), requires_grad=True
        )

        # Key and Value projections for token hidden states
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.weights_v = nn.Linear(input_size, input_size, bias=True)

        # Output projection: [hidden] -> [1] per class
        self.output_linear = nn.Linear(input_size, 1)

        # Layer normalization before output
        self.layernorm = nn.LayerNorm(input_size)

        # Initialize weights
        self._init_weights(mean=WEIGHT_INIT_MEAN, std=WEIGHT_INIT_STD)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute label-wise attention over token representations.

        Args:
            x: Hidden states from LLM [batch_size, seq_len, input_size]
            attention_mask: Mask for padding tokens [batch_size, seq_len]
                           1 = valid token, 0 = padding
            return_attention: Whether to return attention weights (slower)

        Returns:
            logits: [batch_size, num_classes]
            attention: [batch_size, num_classes, seq_len] if return_attention else None
        """
        batch_size, seq_len, _ = x.shape

        # Project tokens to keys and values
        K = self.weights_k(x)  # [batch, seq_len, hidden]
        V = self.weights_v(x)  # [batch, seq_len, hidden]

        # Queries are the learned label representations
        # Expand for batch: [num_classes, hidden] -> [batch, num_classes, hidden]
        Q = self.label_representations.unsqueeze(0).expand(batch_size, -1, -1)

        # Prepare attention mask if provided
        # Convert [batch, seq_len] -> [batch, num_classes, seq_len]
        attn_mask = None
        if attention_mask is not None:
            # Pad mask if needed (in case hidden states have different length)
            if attention_mask.size(1) < seq_len:
                attention_mask = nn.functional.pad(
                    attention_mask, (0, seq_len - attention_mask.size(1)), value=0
                )
            elif attention_mask.size(1) > seq_len:
                attention_mask = attention_mask[:, :seq_len]

            # For scaled_dot_product_attention: True = attend, False = mask out
            attn_mask = attention_mask.bool().unsqueeze(1).expand(-1, self.num_classes, -1)

        if self.use_flash_attention and not return_attention:
            # Fast path using PyTorch's optimized attention
            # Note: Q [batch, n_classes, hidden], K [batch, seq, hidden], V [batch, seq, hidden]
            # We need K, V transposed for attention: K^T is [batch, hidden, seq]
            y = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                scale=1 # Scale used in Edin et al 2024. of PLM-ICD
                # scale=1.0 / (self.input_size ** 0.5),  # Standard scaling from attention, but not used in Edin et al.
            )  # [batch, num_classes, hidden]
            attention = None
        else:
            # orignal calculation with no scaling
            scores = torch.matmul(Q, K.transpose(-2, -1))
            # Manual attention computation (slower but returns attention weights)
            # scores = Q @ K^T / sqrt(d)
            # scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_size ** 0.5)
            # [batch, num_classes, seq_len]

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))

            attention = torch.nn.functional.softmax(scores, dim=-1)
            # [batch, num_classes, seq_len]

            y = torch.matmul(attention, V)
            # [batch, num_classes, hidden]

        # Layer norm and output projection
        y = self.layernorm(y)
        logits = self.output_linear(y).squeeze(-1)
        # [batch, num_classes]

        return logits, attention

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.label_representations, mean, std)
        nn.init.normal_(self.weights_k.weight, mean, std)
        nn.init.normal_(self.weights_v.weight, mean, std)
        nn.init.normal_(self.output_linear.weight, mean, std)
        if self.weights_v.bias is not None:
            nn.init.zeros_(self.weights_v.bias)
        if self.output_linear.bias is not None:
            nn.init.zeros_(self.output_linear.bias)


class LLMLabelCrossAttention(LitLLM):
    """Lightning module combining LLM backbone with Label Cross-Attention head.

    This implements the architecture described in the "Code Like Humans" paper:
    - Llama backbone with optional LoRA adapters
    - Label cross-attention head (fully trained)
    - Binary cross-entropy loss over all labels
    - Threshold tuning on validation F1-macro
    """

    def __init__(
        self,
        checkpoint_dir: str,
        n_classes: int = 50,
        hidden_state_index: int = -1,
        threshold: float = 0.5,
        classifier_dropout: float = 0.1,
        loss_weights: Optional[str] = None,
        optimizer_hparams: dict = None,
        batch_size: int = 2,
        update_batch: int = 64,
        max_epochs: int = 30,
        data_size: int = 31571,
        unfreeze: Optional[str] = None,
        n_blocks: int = 0,
        lora_config: Optional[LowRankAdaptationConfig] = None,
        code_index_groups: dict[str, list[int]] | None = None,
        log_code_types: str = "all",
        use_flash_attention: bool = True,
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

        # Label cross-attention head (Edin et al. 2024 style)
        self.attention = LabelCrossAttention(
            input_size=self.llm.config.hidden_size,
            num_classes=n_classes,
            use_flash_attention=use_flash_attention,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        last_token_index: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: LLM hidden states -> Label Cross-Attention -> logits.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            last_token_index: Index of last valid token per sample [batch_size]
            attention_mask: Padding mask [batch_size, seq_len]

        Returns:
            logits: [batch_size, n_classes]
        """
        # Get hidden states from all tokens
        hidden_state = self.get_hidden_state(
            input_ids=input_ids,
            last_token_index=last_token_index,
            attention_mask=attention_mask,
        )
        # hidden_state: [batch_size, seq_len, hidden_size]

        # Apply label cross-attention
        logits, _ = self.attention(hidden_state, attention_mask=attention_mask)
        # logits: [batch_size, n_classes]

        return logits
