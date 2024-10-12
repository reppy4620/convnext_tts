import torch.nn as nn

from convnext_tts.utils.typing import Float


class EmbeddingLayer(nn.Module):
    def __init__(self, num_vocab, channels):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, channels, padding_idx=0)

    def forward(self, x: Float["B P"], mask: Float["B 1 P"]) -> Float["B C P"]:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = x * mask
        return x
