"""Backbone X-TCN alinhado à Tabela S4 e ao notebook oficial do artigo."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def make_divisible(value: int, divisor: int = 8, minimum: int | None = None) -> int:
    minimum = divisor if minimum is None else minimum
    new_value = max(minimum, int(value + divisor / 2) // divisor * divisor)
    return new_value + divisor if new_value < 0.9 * value else new_value


class HSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=False) / 6.0


class HSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=False) / 6.0


def activation(name: str) -> nn.Module:
    if name == "RE":
        return nn.ReLU(inplace=False)
    if name == "HS":
        return HSwish()
    raise ValueError(f"Ativacao desconhecida: {name}")


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_name: str | None = "RE",
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation_name is not None:
            layers.append(activation(activation_name))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = make_divisible(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1),
            HSigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        activation_name: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if hidden_channels != in_channels:
            layers.append(
                ConvBNAct(in_channels, hidden_channels, 1, activation_name=activation_name)
            )
        layers.append(
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride=stride,
                groups=hidden_channels,
                activation_name=activation_name,
            )
        )
        if use_se:
            layers.append(SEModule(hidden_channels))
        layers.append(
            ConvBNAct(hidden_channels, out_channels, 1, activation_name=None)
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        return x + output if self.use_residual else output


class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 576, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.maximum = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.shared_mlp(self.average(x)) + self.shared_mlp(self.maximum(x))
        return x * self.sigmoid(weights)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1
        )
        return x * self.sigmoid(self.conv(descriptors))


class CBAM(nn.Module):
    def __init__(self, channels: int = 576, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class ArticleAlignedXTCN(nn.Module):
    """Arquitetura publicada, adaptando apenas a saída de quatro para duas classes."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        specs = (
            # in, hidden, out, kernel, stride, SE, activation
            (16, 16, 16, 3, 2, True, "RE"),
            (16, 72, 24, 3, 2, False, "RE"),
            (24, 88, 24, 3, 1, False, "RE"),
            (24, 96, 40, 5, 2, True, "HS"),
            (40, 240, 40, 5, 1, True, "HS"),
            (40, 240, 40, 5, 1, True, "HS"),
            (40, 120, 48, 5, 1, True, "HS"),
            (48, 144, 48, 5, 1, True, "HS"),
            (48, 288, 96, 5, 2, True, "HS"),
            (96, 576, 96, 5, 1, True, "HS"),
            (96, 576, 96, 5, 1, True, "HS"),
        )
        blocks = [ConvBNAct(3, 16, 3, stride=2, activation_name="HS")]
        blocks.extend(InvertedResidual(*spec) for spec in specs)
        blocks.append(ConvBNAct(96, 576, 1, activation_name="HS"))
        self.features = nn.Sequential(*blocks)
        self.attention = CBAM(576, reduction=16)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            HSwish(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.features[12].block[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.features(x))
        return self.classifier(torch.flatten(self.avgpool(x), 1))

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

