import torch.nn as nn
from convnext_tts.modules.convnext import ConvNeXtModule


class WaveNeXt(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        h_channels,
        n_fft,
        hop_length,
        num_layers,
        apply_tanh=True,
    ):
        super().__init__()
        self.apply_tanh = apply_tanh
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channels)
        self.convnext = ConvNeXtModule(channels, h_channels, num_layers)
        self.norm_last = nn.LayerNorm(channels)
        self.out_conv = nn.Conv1d(channels, n_fft, 1)
        self.fc = nn.Conv1d(n_fft, hop_length, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.convnext(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        o = self.out_conv(x)
        o = self.fc(o)
        o = o.reshape(o.size(0), 1, -1)
        if self.apply_tanh:
            o = o.tanh()
        return o
