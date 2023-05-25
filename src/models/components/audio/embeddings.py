import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, code_size=1280) -> None:
        super().__init__()
        self.code_size = code_size

    def forward(self, x):
        return x
