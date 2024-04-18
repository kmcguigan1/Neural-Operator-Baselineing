import torch 
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange

from models.holder.operator_layers import SpectralConv2d, DenseNet, NNConv
from models.layers.base_layers import ConvMLP

class FNOBlock(nn.Module):
    def __init__(self, latent_dims, modes1, modes2):
        super().__init__()
        # save the variables
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        # generate the layers
        self.conv = SpectralConv2d(self.latent_dims, self.latent_dims, self.modes1, self.modes2)
        self.mlp = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * 2)
        self.w = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * 2)
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
    def __init__(self, latent_dims, modes1, modes2):
        super().__init__()
        # save the variables
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        # generate the layers
        self.conv = SpectralConv2d(self.latent_dims, self.latent_dims, self.modes1, self.modes2)
        self.mlp = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * 2)
        self.w = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * 2)
        self.norm = nn.InstanceNorm2d(self.latent_dims)

    def forward(self, x):
        # fourier branch
        x1 = self.norm(self.conv(self.norm(x)))
        x1 = self.mlp(x1)
        # spatial branch
        x2 = self.w(x)
        # add and activate
        return x1 + x2

class TokenFNOBranch(nn.Module):
    def __init__(self, latent_dims:int, modes1:int, modes2:int):
        super().__init__()
        # save the variables
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        # generate the layers
        self.conv = SpectralConv2d(self.latent_dims, self.latent_dims, self.modes1, self.modes2)
        self.mlp = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * 2)
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
    
class GNOBlockSingleConv(nn.Module):
    def __init__(self, latent_dims:int, kernel_dims:int, edge_dims:int, depth:int, activation=F.gelu, activation_first:bool=False):
        # save some variables
        self.latent_dims = latent_dims
        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.depth = depth
        self.activation = activation
        self.activation_first = activation_first
        # layers
        kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.GELU)
        self.conv = NNConv(self.latent_dims, self.latent_dims, kernel)

    def forward(self, nodes, edge_index, edge_attr):
        for idx in range(self.depth):
            if(self.activation_first):
                nodes = self.activation(nodes)
            nodes = self.conv(nodes, edge_index, edge_attr)
            if(not self.activation_first and idx < self.depth - 1):
                nodes = self.activation(nodes)
        return nodes

class GNOBlock(nn.Module):
    def __init__(self, latent_dims:int, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, activation:object=nn.GELU):
        # save some variables
        self.latent_dims = latent_dims
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.depth = depth
        # layers
        kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.GELU)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            in_dims, out_dims = self.latent_dims, self.latent_dims
            if(idx == 0):
                in_dims = self.in_dims
            if(idx == self.depth - 1):
                out_dims = self.out_dims
            self.blocks.append(NNConv(in_dims, out_dims, kernel, activation=activation))
            if(idx < self.depth - 1):
                out_dims = self.out_dims

    def forward(self, nodes, edge_index, edge_attr):
        for block in self.blocks:
            nodes = block(nodes, edge_index, edge_attr)
    
