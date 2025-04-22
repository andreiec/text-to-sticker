import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        assert d_embed % n_heads == 0, f'd_embed {d_embed} must be divisible by n_heads {n_heads}.'

        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x, causal_mask=False):
        B, N, C = x.shape
        #input_shape = x.shape
        #batch_size, seq_len, d_embed = input_shape

        #interim_shape = (batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = self.in_proj(x)
        #q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        #q = q.view(interim_shape).transpose(1, 2)
        #k = k.view(interim_shape).transpose(1, 2)
        #v = v.view(interim_shape).transpose(1, 2)

        #weight = q @ k.transpose(-1, -2)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        
        #if casual_mask:
        #    mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        #    weight.masked_fill_(mask, -torch.inf)

        if causal_mask:
            mask = torch.triu(torch.ones_like(attn_scores, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        #weight = weight / math.sqrt(self.d_head)
        #weight = F.softmax(weight, dim=-1)

        #out = weight @ v
        #out = out.transpose(1, 2)
        #out = out.reshape(input_shape)
        #out = self.out_proj(out)

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)

        #return out
        return self.out_proj(out)


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_context, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_context, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_context, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.n_head = d_embed // n_heads

    def forward(self, x, y):
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.n_head)
        q = self.q_proj(x).view(interim_shape).transpose(1, 2)
        k = self.k_proj(y).view(interim_shape).transpose(1, 2)
        v = self.v_proj(y).view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight = weight / math.sqrt(self.n_head)
        weight = F.softmax(weight, dim=-1)

        out = weight @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(input_shape)
        out = self.out_proj(out)

        return out