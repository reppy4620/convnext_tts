import torch.nn as nn
from convnext_tts.layers.convnext import ConvNeXtLayer


class WaveNeXt(nn.Module):
    def __init__(self, in_channels, channels, h_channels, hop_length, num_layers):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channels)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channels, h_channels, scale) for _ in range(num_layers)]
        )
        self.norm_last = nn.LayerNorm(channels)
        self.out_conv = nn.Conv1d(channels, hop_length, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        o = self.out_conv(x)
        o = o.reshape(o.size(0), 1, -1)
        return o
