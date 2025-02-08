import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            #(Batch_size, 3, Height, Width) --> (Batch_size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            #(Batch_size, 128, Height, Width) --> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            #(Batch_size, 128, Height, Width) --> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2 ,padding=0),

            #(Batch_size, 128, Height/2, Width/2) --> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            #(Batch_size, 256, Height/2, Width/2) --> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2 ,padding=0),

            #(Batch_size, 256, Height/4, Width/4) --> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            #(Batch_size, 512, Height/4, Width/4) --> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2 ,padding=0),

            #(Batch_size, 512, Height/8, Width/8) --> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            #(Batch_size, 512, Height/8, Width/8) --> (Batch_size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),

            #(Batch_size, 512, Height/8, Width/8) --> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            #(Batch_size, 8, Height/8, Width/8) --> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=1),
        )


    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: ((Batch_size, channels, Height, Width))
        # noise: (Batch_size, out_channels, Height/8, Width/8)

        for module in self:
            # getattr(object, name: string , default)
            if getattr(module, 'stride', None) == (2,2):
                # (Left_padding, Right_padding, Top_padding, Down_padding)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # torch.Chunk: Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
        # (Batch_size, 8, Height/8, Width/8) --> two tensors of size (Batch_size, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, chunks=2, dim=1) 

        log_variance = torch.clamp(log_variance, -30, 20)      #(Batch_size, 4, Height/8, Width/8)
        variance = log_variance.exp()                           #(Batch_size, 4, Height/8, Width/8)
        stdev = variance.sqrt()                                 #(Batch_size, 4, Height/8, Width/8)

        '''
        To transorm a standard Normal dist Z = N(0,1) to a learned Normal dist X = N(mean, variance) to sample from it:
        X = mean + stdev*Z
        Directly sampling from X would introduce stochasticity into the computation graph, making gradients undefined. Instead, the reparameterization trick reformulates 
        the sampling process so that the randomness comes only from  "epsilon" , which is independent of mu and sigma. 
        This makes the operation differentiable and allows backpropagation to optimize the parameters of mu and sigma .
        '''
        x = mean + stdev * noise 

        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215


        return x 