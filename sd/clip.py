import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class ClipEmbedding(nn.Module):
    
    def __init__(self, vocab_size: int, d_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.positional_embed = nn.Parameter(torch.zeros(n_tokens, d_embed))


    def forward(self, tokens: torch.Tensor):

        embed = self.token_embedding(tokens)
        embed += self.positional_embed
        
        return embed
    


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(d_embed)
        self.layernorm2 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_heads, d_embed)
        self.linear1 = nn.Linear(d_embed, 4*d_embed)
        self.linear2 = nn.Linear(4*d_embed, d_embed)

    def forward(self, x: torch.Tensor):
        #(Batch_size, seq_len, d_embed)
        residue = x 

        # Attention
        x = self.layernorm1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        residue = x 

        #FeedForward 
        x = self.layernorm2(x)

        x = self.linear1(x)

        x = x * F.sigmoid(1.702 * x) # Quick GELU activation funtion

        x = self.linear2(x)

        x += residue

        return x
        


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ClipEmbedding(vocab_size=49408, d_embed= 768, n_tokens= 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        #(Batch_size, seq_len) --> Batch_size, seq_len, dim)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output
