import torch
import torch.nn as nn 

from .LayerNormalization import LayerNorm
from .Multihead_attention import Multihead_attention_layer
from .FeedForwardNetwork import FeedForward

class SingleEncoderBlock(nn.Module):
    def __init__(self, emb_dims: int = 768, num_heads: int = 8, dropout: float = 0.1, mlp_ratio: int = 4):
        """
        A single Transformer encoder block.
        """
        super().__init__()
        self.norm1 = LayerNorm(emb_dims)
        self.attn = Multihead_attention_layer(emb_dims, num_heads, dropout)  # You already built this
        self.norm2 = LayerNorm(emb_dims)
        self.ffn = FeedForward(emb_dims, emb_dims * mlp_ratio, dropout)

    def forward(self, x):
        # Attention + residual
        attn_out, _ = self.attn(self.norm1(x))  # (B, seq_len, emb_dims)
        x = x + attn_out

        # Feed Forward + residual
        ffn_out = self.ffn(self.norm2(x))       # (B, seq_len, emb_dims)
        x = x + ffn_out

        return x
