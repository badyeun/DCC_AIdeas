"""Mission 1 model definitions.

The competition submission keeps model-related utilities in this module.
The trained WavLM classification head architecture must match training.
"""

from __future__ import annotations

import torch
from torch import nn


class WavLMHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
