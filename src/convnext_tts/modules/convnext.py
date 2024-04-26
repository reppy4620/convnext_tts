import torch.nn as nn
from convnext_tts.layers.convnext import ConvNeXtLayer


# Stack of ConvNeXt layers
class ConvNeXtModule(nn.Module):
    def __init__(self, channels, h_channels, num_layers):
        super().__init__()
        scale = 1.0 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channels, h_channels, scale) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
        return x
