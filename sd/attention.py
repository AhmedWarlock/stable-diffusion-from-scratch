import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x, causal_mask=False):

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head )

        # (batch_size, seq_len, d_embed) --> (batch_size, seq_len, 3 * d_embed) -->3 tensors = (batch_size, seq_len, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim= -1)
       
        # (batch_size, seq_len, d_embed) --> (batch_size, seq_len, H, d_head ) --> (batch_size, H, seq_len, d_head ) 
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        # (batch_size, H, seq_len, seq_len ) 
        weights = q @ k.transpose(-1,-2)

        if causal_mask: 
            mask = torch.ones_like(weights, dtype= torch.bool).triu(1)
            weights = weights.masked_fill(mask, -torch.inf)

        weights /= math.sqrt(self.d_head)
        weights = F.softmax(weights, dim=-1)

        # (batch_size, H, seq_len, d_head )
        output = weights @ v
        # (batch_size, seq_len, H, d_head )
        output = output.transpsoe(1,2)
        
        # (batch_size, seq_len, d_embed)
        output = output.view(input)

        output = self.out_proj(output)

        return output





class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True ):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.q_proj = nn.Linear(d_embed, d_embed, bias= in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias= in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias= in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias= out_proj_bias)


    def forward (self, x: torch.Tensor, y:torch.Tensor):
        # x: (latent) : (Batch, seq_len_Q, dim)
        # y: (contex) : (Batch, seq_len_KV, dim) = (Batch, 77, 768)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.q_proj(y)
        v = self.q_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.trasnpose(-1,-2)

        weight/= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1,2).continuous()

        output = output.view(input_shape)

        output = self.out_proj(output)
        
        return output
    