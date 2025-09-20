import torch
import torch.nn as nn

from .InputEmbedding import InputEmbedding
from .ViT_encoder import TransformerEncoder
from .LayerNormalization import LayerNorm

class ViT(nn.Module):
    def __init__(
        self, 
        image_size = 224, 
        patch_size: int = 16, 
        num_classes: int = 1000, 
        emb_dims: int = 768, 
        depth: int = 12, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        mlp_ratio: int = 4
    ):
        """
        Vision Transformer (ViT) complete .
        """
        super().__init__()
        # Patch + Position + CLS token embedding
        self.embedding = InputEmbedding(image_size, patch_size, 3, emb_dims, dropout)
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(depth, emb_dims, num_heads, dropout, mlp_ratio)
        
        # Final LayerNorm before classifier
        self.norm = LayerNorm(emb_dims)
        
        # Classification layer
        self.mlp_head = nn.Linear(emb_dims, num_classes)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)         # (B, 197, emb_dims)

        x = self.encoder(x)           # (B, 197, emb_dims)

        x = self.norm(x)            

        cls_token = x[:, 0]           # (B, emb_dims)

        logits = self.mlp_head(self.mlp_dropout(cls_token))  # (B, num_classes)
        
        return logits
