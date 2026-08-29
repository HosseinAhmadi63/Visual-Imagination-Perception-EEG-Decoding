import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = inputs.shape
        scale = self.pool(inputs).reshape(batch, channels)
        scale = self.excitation(scale).reshape(batch, channels, 1, 1)
        return inputs * scale
