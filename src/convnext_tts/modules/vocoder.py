import torch.nn as nn

from convnext_tts.modules.convnext import ConvNeXtModule
from convnext_tts.utils.typing import Float


class ConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        h_channels,
        num_layers,
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channels)
        self.convnext = ConvNeXtModule(channels, h_channels, num_layers)
        self.norm_last = nn.LayerNorm(channels)

    def forward(self, x: Float["batch channel frame"]) -> Float["batch frame channel"]:
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.convnext(x)
        x = self.norm_last(x.transpose(1, 2))
        return x


class VocosHead(nn.Module):
    def __init__(self, channels, out_channels, istft):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.out_conv = nn.Conv1d(channels, out_channels, 1)
        self.istft = istft

    def forward(self, x: Float["batch frame channel"]) -> Float["batch 1 sample"]:
        x = self.pad(x.transpose(1, 2))
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp_max(max=1e2)
        s = mag * (phase.cos() + 1j * phase.sin())
        o = self.istft(s).unsqueeze(1)
        return o


class WaveNeXtHead(nn.Module):
    def __init__(self, channels, n_fft, hop_length):
        super().__init__()
        self.out_linear = nn.Linear(channels, n_fft)
        self.fc = nn.Linear(n_fft, hop_length)

    def forward(self, x: Float["batch frame channel"]) -> Float["batch 1 sample"]:
        x = self.out_linear(x)
        x = self.fc(x)
        x = x.reshape(x.size(0), 1, -1).clip(-1, 1)
        return x


class Vocoder(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Float["batch channel frame"]) -> Float["batch 1 sample"]:
        x = self.backbone(x)
        o = self.head(x)
        return o
