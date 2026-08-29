from typing import Any

import torch
from torch import nn

from vip_eeg.models.se import SqueezeExcitation


class ConvolutionSEBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int,
        dropout: float,
        reduction: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channels),
            SqueezeExcitation(output_channels, reduction),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TopomapSECNN(nn.Module):
    def __init__(
        self,
        image_size: int = 150,
        input_channels: int = 3,
        filters: list[int] | tuple[int, ...] = (32, 64, 128, 256, 512),
        kernel_size: int = 3,
        padding: int = 1,
        convolution_dropout: float = 0.1,
        se_reduction: int = 16,
        dense_units: int = 256,
        dense_dropout: float = 0.3,
    ):
        super().__init__()
        blocks = []
        channels = input_channels
        for output_channels in filters:
            blocks.append(
                ConvolutionSEBlock(
                    channels,
                    output_channels,
                    kernel_size,
                    padding,
                    convolution_dropout,
                    se_reduction,
                )
            )
            channels = output_channels
        self.features = nn.ModuleList(blocks)
        spatial = image_size
        for _ in filters:
            spatial //= 2
        self.feature_shape = (int(filters[-1]), spatial, spatial)
        flattened = int(filters[-1]) * spatial * spatial
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, dense_units),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(dense_units),
            nn.Dropout(dense_dropout),
            nn.Linear(dense_units, 1),
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for block in self.features:
            values = block(values)
        return values

    def intermediate_shapes(self, inputs: torch.Tensor) -> list[tuple[int, ...]]:
        values = inputs
        shapes = []
        for block in self.features:
            values = block(values)
            shapes.append(tuple(values.shape))
        return shapes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        return self.classifier(features).squeeze(1)


def build_model(config: dict[str, Any]) -> TopomapSECNN:
    settings = config["cnn"]
    topomaps = config["topomaps"]
    return TopomapSECNN(
        image_size=int(topomaps["image_size"]),
        input_channels=int(topomaps["rgb_channels"]),
        filters=[int(value) for value in settings["filters"]],
        kernel_size=int(settings["kernel_size"]),
        padding=int(settings["convolution_padding"]),
        convolution_dropout=float(settings["convolution_dropout"]),
        se_reduction=int(settings["se_reduction"]),
        dense_units=int(settings["dense_units"]),
        dense_dropout=float(settings["dense_dropout"]),
    )
