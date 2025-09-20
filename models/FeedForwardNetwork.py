import torch 
import torch.nn as nn 

class FeedForward(nn.Module):
    def __init__(self , emb_dims , hidden_dims = 3072 , dropout = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(emb_dims , hidden_dims)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dims , emb_dims)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self , x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        return x 