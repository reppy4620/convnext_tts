from typing import Optional

import torch.nn as nn

from convnext_tts.layers.convnext import ConvNeXtLayer
from convnext_tts.utils.typing import Float


# Stack of ConvNeXt layers
class ConvNeXtModule(nn.Module):
    def __init__(self, channels: int, h_channels: int, num_layers: int) -> None:
        super().__init__()
        scale = 1.0 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channels, h_channels, scale) for _ in range(num_layers)]
        )

    def forward(
        self, x: Float["B C T"], mask: Optional[Float["B 1 T"]] = None
    ) -> Float["B C T"]:
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
        return x
