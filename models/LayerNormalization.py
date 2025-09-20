import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dims, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        # learnable params
        self.alpha = nn.Parameter(torch.ones(emb_dims))
        self.beta = nn.Parameter(torch.zeros(emb_dims))

    def forward(self, input_embeddings):
        """
        input_embeddings: (batch_size, seq_len, emb_dims)
        """
        mean = input_embeddings.mean(dim=-1, keepdim=True)  # (B, seq_len, 1)
        var = input_embeddings.var(dim=-1, keepdim=True, unbiased=False)  # (B, seq_len, 1)

        input_normalized = (input_embeddings - mean) / torch.sqrt(var + self.epsilon)

        return self.alpha * input_normalized + self.beta
