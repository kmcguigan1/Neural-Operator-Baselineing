import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

import math

from einops import rearrange

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
class ConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, padding_mode=None):
        super().__init__()
        # vars
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
        x_ft = torch.fft.rfft2(x, norm="ortho")
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize,  x.size(1), x.size(2)//2 + 1, self.out_channels, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[..., :self.modes1, :self.modes2], self.weights1)
        out_ft[..., -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[..., -self.modes1:, :self.modes2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x
    
class FNOBlock(nn.Module):
    def __init__(self, image_size:tuple, latent_dims:int, modes1:int, modes2:int):
        super().__init__()
        # save the variables
        self.image_size = image_size
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        # generate the layers
        self.conv = SpectralConv2d(self.latent_dims, self.latent_dims, self.modes1, self.modes2)
        self.mlp = ConvMLP(self.latent_dims, self.latent_dims, self.latent_dims * 2)
        self.norm = nn.InstanceNorm2d(self.latent_dims)

    def forward(self, x, batchsize):
        B, C = x.shape
        # reshape the inputs
        x = rearrange(x, "(b h w) c -> b h w c", b=batchsize, c=C, h=self.image_size[0], w=self.image_size[1])
        x = rearrange(x, "b h w c -> b c h w")
        # fourier branch
        x = self.norm(self.conv(self.norm(x)))
        x = self.mlp(x)
        # reshape the outputs
        x = rearrange(x, "b c h w -> b h w c")
        x = rearrange(x, "b h w c -> (b h w) c", b=batchsize, c=C, h=self.image_size[0], w=self.image_size[1])
        return x

class NNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, kernel, aggr='add', activation='gelu', root_weight=True, bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.aggr = aggr
        self.activation = nn.GELU() if activation == 'gelu' else nn.Identity()
        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.kernel)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.kernel(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return self.activation(aggr_out)

class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x
    
class GNOBlock(nn.Module):
    def __init__(self, latent_dims:int, kernel_dims:int, edge_dims:int, graph_passes:int):
        super().__init__()
        # save some variables
        self.latent_dims = latent_dims
        self.graph_passes = graph_passes
        self.edge_dims = edge_dims
        self.kernel_dims = kernel_dims
        # layers
        kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.GELU)
        self.blocks = nn.ModuleList()
        for idx in range(self.graph_passes):
            activation = 'gelu'
            if(idx == self.graph_passes - 1):
                activation = 'none'
            self.blocks.append(NNConv(self.latent_dims, self.latent_dims, kernel, activation=activation))

    def forward(self, nodes, edge_index, edge_attr):
        for block in self.blocks:
            nodes = block(nodes, edge_index, edge_attr)
        return nodes

class OperatorBlock(nn.Module):
    def __init__(self, image_size:tuple, latent_dims:int, modes1:int, modes2:int, kernel_dims:int, edge_dims:int, graph_passes:int, activation:str='gelu'):
        super().__init__()
        # save the variables
        self.image_size = image_size
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.graph_passes = graph_passes
        # save the layers
        self.fno_block = FNOBlock(self.image_size, self.latent_dims, self.modes1, self.modes2)
        self.gno_block = GNOBlock(self.latent_dims, self.kernel_dims, self.edge_dims, self.graph_passes)
        self.fc_block = MLP(self.latent_dims, self.latent_dims, self.latent_dims)
        self.activation = nn.GELU() if activation =='gelu' else nn.Identity()

    def forward(self, nodes, edge_index, edge_attrs, batchsize):
        x1 = self.fno_block(nodes, batchsize)
        x2 = self.gno_block(nodes, edge_index, edge_attrs)
        x3 = self.fc_block(nodes)
        return self.activation(x1 + x2 + x3)

class CustomOPP(nn.Module):
    def __init__(self, config:dict, image_size:tuple):
        super().__init__()
        # save needed vars
        self.image_size = image_size
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        # fourier based information
        self.depth = config["DEPTH"]
        self.modes = (config['MODES1'], config['MODES2'])
        # graph based information
        self.kernel_dims = config['KERNEL_DIMS']
        self.edge_dims = 5
        self.graph_passes = config['GRAPH_PASSES']
        # setup layers
        self.project = MLP(self.in_dims, self.latent_dims, self.latent_dims//2)
        self.decode = MLP(self.latent_dims, 1, self.latent_dims//2)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            self.blocks.append(OperatorBlock(self.image_size, self.latent_dims, self.modes[0], self.modes[1], self.kernel_dims, self.edge_dims, self.graph_passes, activation='gelu' if idx < self.depth - 1 else 'none'))

    def forward(self, xx, grid, edge_index, edge_attrs, batch, ptr):
        # grid = self.get_grid(xx.shape, xx.device)
        # save the starting solution
        # a = xx[..., -1]
        # add the grid to the data
        x = torch.cat((xx, grid), dim=-1)
        batchsize = ptr.size(0) - 1
        B, C = x.shape
        assert C == self.in_dims and B == batchsize * self.image_size[0] * self.image_size[1]
        # project the data
        x = self.project(x)
        # go thorugh the blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_attrs, batchsize)
        # decode the prediction
        x = self.decode(x)
        # return the predictions
        return x