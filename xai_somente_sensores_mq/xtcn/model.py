"""X-TCN 2D com atenção dupla para Fingerprint Maps."""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        maximum = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * self.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat((avg, maximum), dim=1)))


class DualAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class TemporalResidualBlock(nn.Module):
    """Bloco dilatado; o eixo horizontal preserva a ordem dos descritores."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + identity)


class XTCN(nn.Module):
    """Backbone convolucional explicável com CA + SA e seis saídas."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.25) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = TemporalResidualBlock(16, 24, dilation=1)
        self.block2 = TemporalResidualBlock(24, 40, dilation=2)
        self.block3 = TemporalResidualBlock(40, 64, dilation=2)
        self.attention = DualAttention(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Última convolução do backbone, antes da atenção dupla."""
        return self.block3.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attention(x)
        return self.classifier(self.pool(x))

