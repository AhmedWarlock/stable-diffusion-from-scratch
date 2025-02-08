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





    