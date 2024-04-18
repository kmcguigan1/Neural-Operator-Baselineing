import torch 
import torch.nn as nn 
import torch.nn.functional as F

import math

from models.layers.base_layers import ConvMLP
from einops import rearrange

class SpectralConv2d(nn.Module):
    """We expect the input to be in the format B C H W.
    This means we expect channel first image style data"""
    def __init__(self, in_channels, out_channels, modes1, modes2, padding_mode=None):
        super().__init__()
        # variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.padding_mode = padding_mode
        # layers
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        if(self.padding_mode == 'EVERY_SINGLE'):
            padding = int(math.sqrt(x.size(-1)))
            x = F.pad(x, [0, padding, 0, padding])
        elif(self.padding_mode == 'EVERY_DUAL'):
            padding = int(math.sqrt(x.size(-1)) // 2)
            x = F.pad(x, [padding, padding, padding, padding])
            
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        if(self.padding_mode == 'EVERY_SINGLE'):
            x = x[..., :-padding, :-padding]
        elif(self.padding_mode == 'EVERY_DUAL'):
            x = x[..., padding:-padding, padding:-padding]
        return x

class FNOBlockWithW(nn.Module):
    def __init__(self, latent_dims, modes1, modes2, mlp_ratio=2, padding_mode=None):
        super().__init__()
        # save the variables
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        self.mlp_ratio = mlp_ratio
        self.padding_mode = padding_mode
        # generate the layers
        self.conv = SpectralConv2d(self.latent_dims, self.latent_dims, self.modes1, self.modes2, padding_mode=self.padding_mode)
        if(self.mlp_ratio is not None):
            self.mlp = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * self.mlp_ratio)
            self.w = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * self.mlp_ratio)
        else:
            self.mlp = nn.Conv2d(self.latent_dims, self.latent_dims, 1)
            self.w = nn.Conv2d(self.latent_dims, self.latent_dims, 1)
        self.norm = nn.InstanceNorm2d(self.latent_dims)

    def forward(self, x):
        # fourier branch
        x1 = self.norm(self.conv(self.norm(x)))
        x1 = self.mlp(x1)
        # spatial branch
        x2 = self.w(x)
        # add and activate
        return x1 + x2
    
class FNOBlock(nn.Module):
    def __init__(self, latent_dims, modes1, modes2, mlp_ratio=2, padding_mode=None):
        super().__init__()
        # save the variables
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        self.mlp_ratio = mlp_ratio
        self.padding_mode = padding_mode
        # generate the layers
        self.conv = SpectralConv2d(self.latent_dims, self.latent_dims, self.modes1, self.modes2, padding_mode=self.padding_mode)
        if(self.mlp_ratio is not None):
            self.mlp = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * self.mlp_ratio)
        else:
            self.mlp = nn.Conv2d(self.latent_dims, self.latent_dims, 1)
        self.norm = nn.InstanceNorm2d(self.latent_dims)

    def forward(self, x):
        # fourier branch
        x = self.norm(self.conv(self.norm(x)))
        x = self.mlp(x)
        return x

class TokenFNOBranch(nn.Module):
    def __init__(self, latent_dims:int, modes1:int, modes2:int, mlp_ratio=2, padding_mode=None):
        super().__init__()
        # save the variables
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        self.mlp_ratio = mlp_ratio
        self.padding_mode = padding_mode
        # generate the layers
        self.conv = SpectralConv2d(self.latent_dims, self.latent_dims, self.modes1, self.modes2, padding_mode=self.padding_mode)
        if(self.mlp_ratio is not None):
            self.mlp = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * self.mlp_ratio)
        else:
            self.mlp = nn.Conv2d(self.latent_dims, self.latent_dims, 1)    
        self.norm = nn.InstanceNorm2d(self.latent_dims)

    def forward(self, x, image_size, batch_size):
        B, C = x.shape
        # reshape the inputs
        x = rearrange(x, "(b h w) c -> b h w c", b=batch_size, c=C, h=image_size[0], w=image_size[1])
        x = rearrange(x, "b h w c -> b c h w")
        # fourier branch
        x = self.norm(self.conv(self.norm(x)))
        x = self.mlp(x)
        # reshape the outputs
        x = rearrange(x, "b c h w -> b h w c")
        x = rearrange(x, "b h w c -> (b h w) c", b=batch_size, c=C, h=image_size[0], w=image_size[1])
        return x