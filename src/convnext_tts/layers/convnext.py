import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from convnext_tts.utils.typing import Float


class ConvNeXtLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        h_channels: int,
        scale: float,
        stochastic_depth_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.stochastic_depth_rate = stochastic_depth_rate

        self.dw_conv = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pw_layer1 = nn.Linear(channels, h_channels)
        self.pw_layer2 = nn.Linear(h_channels, channels)
        self.scale = nn.Parameter(
            torch.full(size=(channels,), fill_value=scale), requires_grad=True
        )

    def forward(self, x: Float["B C T"]) -> Float["B C T"]:
        res = x
        x = self.dw_conv(x)
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_layer1(x)
        x = F.gelu(x)
        x = self.pw_layer2(x)
        x = x * self.scale
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        if self.training and self.stochastic_depth_rate > 0:
            if random.random() < self.stochastic_depth_rate:
                return res
        x = x + res
        return x
