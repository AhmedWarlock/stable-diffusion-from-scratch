import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, 4*dim)
        self.linear_2 = nn.Linear(4*dim, 4*dim)

    def forward(self, x: torch.Tensor):
        # x: (1, 320)
        
        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        return x # (1 , 1280)
    

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
      super.__init__()

      self.groupnorm_feature = nn.GroupNorm(32, in_channels)
      self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
      self.time_lin_layer = nn.Linear(n_time, out_channels)

      self.groupnorm_merged = nn.GroupNorm(32, out_channels)
      self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

      if in_channels == out_channels:
          self.residual_layer = nn.Identity()
      else : 
          self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, feature, time):
        # feature: (Batch_size, in_channels, Height, Width)
        # time: (1, 1280)

        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.time_lin_layer(time)

        merged = feature + time.unsqeeze(-1).unsqeeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)



class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_context=768):
      super.__init__()

      channels = n_heads * d_embed

      self.groupnorm = nn.GroupNorm(32, channels,  eps=1e-6)
      self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

      self.layernorm_1 = nn.LayerNorm(channels)
      self.self_attention = SelfAttention(n_heads=n_heads, d_embed=d_embed, in_proj_bias=False)
      self.layernorm_2 = nn.LayerNorm(channels)
      self.cross_attention = CrossAttention(n_heads=n_heads, d_embed=d_embed, d_context= d_context, in_proj_bias=False)
      self.layernorm_3 = nn.LayerNorm(channels)

      self.gegelu_layer_1 = nn.Linear(channels, 4 * channels * 2)
      self.gegelu_layer_2 = nn.Linear(4 * channels, channels)

      self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor): 
        # x: (Batch_size, channels, Height, Width)
        # context: (Batch_size, seq_len, d_context)

        residue_long = x 

        x = self.groupnorm(x)
        x = self.conv_input(x)

        b, c, h, w = x.shape
        #(Batch_size, channels, Height, Width) --> (Batch_size, Height * Width, channels)
        x = x.view((b, c, h * w))
        x = x.transpose(-1, -2)

        # Normalization + self attention
        residue_short = x

        x = self.layernorm_1(x)
        x = self.self_attention(x)
        x += residue_short
        
        # Normalization + cross attention
        residue_short = x 
        x = self.layernorm_2(x)
        x = self.cross_attention(x, context)
        x += residue_short

        residue_short = x 
        x, gate = self.gegelu_layer_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.gegelu_layer_2(x)

        x+= residue_short


        x =  x.transpose(-1, -2)
        x = x.view((b, c, h, w))

        x = self.conv_out(x)

        return x + residue_long





class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                layer(x, time)
            else:
                layer(x)




class Upsample(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)

    def forward(self, x):
         # (Batch, channels, Heigh, Width) --> (Batch, 320, Heigh * 2, Width * 2)
         x = F.interpolate(x, scale_factor=2, mode= "nearest")

         return self.conv(x)

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([
            # (Batch, 4, Heigh/8, Width/8) --> (Batch, 320, Heigh/8, Width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),


            # (Batch, 320, Heigh/8, Width/8) --> (Batch, 320, Heigh/16, Width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # (Batch, 320, Heigh/16, Width/16) --> (Batch, 640, Heigh/16, Width/16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (Batch, 320, Heigh/16, Width/16) --> (Batch, 320, Heigh/32, Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=2)),

            # (Batch, 640, Heigh/32, Width/32) --> (Batch, 1280, Heigh/32, Width/32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch, 320, Heigh/32, Width/32) --> (Batch, 320, Heigh/64, Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            # (Batch, 640, Heigh/64, Width/64) --> (Batch, 1280, Heigh/64, Width/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = nn.SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.Module([
            # (Batch, 2560 (2x1280), Heigh/64, Width/64) --> (Batch, 1280, Heigh/64, Width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch, 2560 (2x1280), Heigh/64, Width/64) --> (Batch, 1280, Heigh/32, Width/32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),


            # (Batch, 1920 (1280 + 640), Heigh/32, Width/32) --> (Batch, 1280, Heigh/16, Width/16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            # (Batch, 1920 (1280 + 640), Heigh/16, Width/16) --> (Batch, 640, Heigh/16, Width/16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            # (Batch, 1280 (640 + 640), Heigh/16, Width/16) --> (Batch, 640, Heigh/16, Width/16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),


            # (Batch, 960 (640+320), Heigh/16, Width/16) --> (Batch, 640, Heigh/8, Width/8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            # (Batch, 960 (640+320), Heigh/8, Width/8) --> (Batch, 320, Heigh/8, Width/8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            # (Batch, 640 (320+320), Heigh/8, Width/8) --> (Batch, 320, Heigh/8, Width/8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
            # (Batch, 640 (320+320), Heigh/8, Width/8) --> (Batch, 320, Heigh/8, Width/8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

        ])

        

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        #x: (Batch_size, 320, Heigh/8, Width/8)
        x = self.groupnorm(x)

        #x: (Batch_size, 320, Heigh/8, Width/8) -- > (Batch_size, 4, Heigh/8, Width/8)
        x = self.conv(x)

        return x
    

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.output_layer = UNET_OutputLayer(320, 4)

    def forward(self, latnent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_size, 4, Heigh/8, Width/8) output of decoder
        # context: (Batch_size, seq_len, d_embed) output of CLIP
        # time: (1, 320)
        # (1, 320) --> (1, 1280)
        time = self.time_embedding(time)

        # (Batch_size, 4, Heigh/8, Width/8) --> (Batch_size, 320, Heigh/8, Width/8)
        output = self.unet(latnent, context, time)

        # (Batch_size, 320, Heigh/8, Width/8) --> (Batch_size, 4, Heigh/8, Width/8)
        output = self.output_layer(output)

        return output