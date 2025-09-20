import torch
import torch.nn as nn 
import math

class Multihead_attention_layer(nn.Module):
    def __init__(self , emb_dims = 768 , num_heads = 8 , dropout = 0.1 ):
        super().__init__()
        self.emb_dims = emb_dims
        self.num_attention_heads = num_heads
        self.attention_head_size = self.emb_dims//self.num_attention_heads
        self.all_head_size = self.attention_head_size * num_heads

        self.query = nn.Linear(self.emb_dims , self.all_head_size)
        self.key = nn.Linear(self.emb_dims , self.all_head_size)
        self.value = nn.Linear(self.emb_dims , self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.all_head_size , emb_dims)

    
    def forward(self , input_embeddings):
        """
        x = (batch_size , patches/seq_len , emb_dims)
        """

        batch_size , seq_len , _ = input_embeddings.shape

        #  project input to Q , K , V
        query = self.query(input_embeddings) # outputs vector of (batch_size , seq_len , all_head_size)
        key = self.key(input_embeddings)
        value = self.value(input_embeddings)

        # (batch_size , seq_len , num_heads , head_size) --> (batch_size , num_heads, seq_len , head_size)
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # attention_scores 
        # Q @ K^T -> (B, num_heads, seq_len, seq_len)
        scores = torch.matmul(query , key.transpose(-2 , -1)) / math.sqrt(self.attention_head_size)
        
        # attention_probabilities
        attention_probabilities = nn.Softmax(dim = -1)(scores)
        attention_probabilities = self.dropout(attention_probabilities)

        # weighted sum of values 
        context  = torch.matmul(attention_probabilities , value) # (B, num_heads, seq_len, head_size)

        # concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size) , 
        # gives output of (B , 197 , 768)

        # Final linear projection
        output = self.out_proj(context)  # (B, seq_len, emb_dims)

        return output , attention_probabilities
        
