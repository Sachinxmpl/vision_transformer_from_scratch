import torch.nn as nn 
import torch
from .SingleEncoderBlock import SingleEncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, emb_dims: int = 768, num_heads: int = 8, dropout: float = 0.1, mlp_ratio: int = 4):
        """
        Transformer encoder: stack of L identical layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            SingleEncoderBlock(emb_dims, num_heads, dropout, mlp_ratio) 
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Pass through each encoder block
        return x