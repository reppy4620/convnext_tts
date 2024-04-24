import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtLayer(nn.Module):
    def __init__(self, channels, h_channels, scale):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pw_conv1 = nn.Linear(channels, h_channels)
        self.pw_conv2 = nn.Linear(h_channels, channels)
        self.scale = nn.Parameter(
            torch.full(size=(channels,), fill_value=scale), requires_grad=True
        )

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = x.transpose(1, 2)
        x = res + x
        return x
