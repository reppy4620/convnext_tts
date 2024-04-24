import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_vocab, channels):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, channels, padding_idx=0)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = x * mask
        return x
