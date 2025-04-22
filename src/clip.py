import torch
from torch import nn

from torch.nn import functional as F
from attention import SelfAttention


class CLIP_Embedding(nn.Module):
    def __init__(self, n_vocab, d_embed, n_tokens):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_embed)
        self.position_embedding = nn.Parameter(torch.zeros((n_tokens, d_embed)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x = x + self.position_embedding
        return x


class CLIP_Layer(nn.Module):
    def __init__(self, n_heads, d_embed):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_heads, d_embed)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_2 = nn.Linear(d_embed * 4, d_embed)
    
    def forward(self, x):
        residual = x
        
        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask=True)
        x = x + residual

        residual = x
    
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)
        x = x + residual

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = CLIP_Embedding(49406, 768, 77)

        self.layers = nn.ModuleList([
            CLIP_Layer(12, 768) for _ in range(12)
        ])

        self.layernom = nn.LayerNorm(768)
    
    def forward(self, tokens):
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        state = self.layernorm(state)
        return state
