"""PyTorch Transformer used in the AR(1) recovery experiment."""

import torch
from torch import nn

from .config import ModelConfig


class ARTransformer(nn.Module):
    """Encoder-only Transformer for one-step univariate forecasting.

    Input shape is ``[batch, lookback]`` and output shape is ``[batch]``.
    Each scalar is embedded into ``d_model`` dimensions, augmented by a
    learned positional embedding, and processed by a Transformer encoder.
    The representation at the final observed time step predicts the next
    standardized value.

    There is no causal mask: the complete input window is observed when the
    forecast is made, and the target is never included in that window.
    """

    def __init__(self, lookback: int, config: ModelConfig) -> None:
        super().__init__()
        if lookback <= 0:
            raise ValueError("lookback must be positive.")
        self.lookback = lookback
        self.input_projection = nn.Linear(1, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, lookback, config.d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.output_projection = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forecast the observation following each input window."""

        if inputs.ndim != 2 or inputs.shape[1] != self.lookback:
            raise ValueError(
                f"Expected inputs with shape [batch, {self.lookback}], "
                f"got {tuple(inputs.shape)}."
            )
        hidden = self.input_projection(inputs.unsqueeze(-1))
        hidden = hidden + self.position_embedding
        encoded = self.encoder(hidden)
        return self.output_projection(encoded[:, -1]).squeeze(-1)
