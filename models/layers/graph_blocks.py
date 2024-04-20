import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

from einops import rearrange

class NNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, kernel, aggr='mean', root_weight=True, bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.aggr = aggr
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

    def forward(self, x, edge_index, edge_attr, a=None):
        return self.propagate(edge_index, x=x, pseudo=edge_attr)

    def message(self, x_j, pseudo):
        weight = self.kernel(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class NNConvEdges(NNConv):
    def __init__(self, in_channels, out_channels, kernel, aggr='mean', root_weight=True, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, kernel, aggr=aggr, root_weight=root_weight, bias=bias, **kwargs)
    
    def forward(self, x, edge_index, edge_attr, a=None):
        return self.propagate(edge_index, x=x, pseudo=edge_attr) 
    
    def message(self, x_i, x_j, pseudo):
        pseudo = torch.cat((pseudo, x_i, x_j), dim=-1)
        weight = self.kernel(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

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
    
"""SINGLE CONV GNO BLOCKS"""
class GNOBlockSigleConvBase(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int):
        super().__init__()
        # save some variables
        assert in_dims == out_dims
        self.latent_dims = in_dims
        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.depth = depth
        self.activation = F.gelu

    def forward(self, *args):
        raise NotImplementedError('')

class GNOBlockSingleConv(GNOBlockSigleConvBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth)
        kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.GELU)
        self.conv = NNConv(self.latent_dims, self.latent_dims, kernel)

    def forward(self, nodes, edge_index, edge_attr, a=None):
        for idx in range(self.depth):
            nodes = self.conv(nodes, edge_index, edge_attr, a=a)
            if(idx < self.depth - 1):
                nodes = self.activation(nodes)
        return nodes
    
class GNOBlockSingleConvAddNodesToEdge(GNOBlockSigleConvBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth)
        self.edge_dims += 2 * self.latent_dims
        kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.GELU)
        self.conv = NNConvEdges(self.latent_dims, self.latent_dims, kernel)

    def forward(self, nodes, edge_index, edge_attr, a=None):
        for idx in range(self.depth):
            nodes = self.conv(nodes, edge_index, edge_attr, a=a)
            if(idx < self.depth - 1):
                nodes = self.activation(nodes)
        return nodes
    
"""MULTI CONV GNO BLOCKS"""
class GNOBlockBase(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.depth = depth
        self.activation = F.gelu
        self.shorten_kernel = shorten_kernel
    
    def forward(self, *args):
        raise NotImplementedError('')

class GNOBlock(GNOBlockBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool=False):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth, shorten_kernel)
        if(self.shorten_kernel):
            kernel = DenseNet([self.edge_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        else:
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            if(idx == 0):
                in_dims = self.in_dims
            else:
                in_dims = self.out_dims
            self.blocks.append(NNConv(in_dims, self.out_dims, kernel))

    def forward(self, nodes, edge_index, edge_attr, a=None):
        for idx, block in enumerate(self.blocks):
            nodes = block(nodes, edge_index, edge_attr, a=a)
            if(idx < len(self.blocks) - 1):
                nodes = self.activation(nodes)
        return nodes
    
class GNOBlockAddNodesToEdge(GNOBlockBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool=False):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth, shorten_kernel)
        assert self.in_dims == self.out_dims or self.depth == 1
        self.edge_dims += 2 * self.in_dims
        if(self.shorten_kernel):
            kernel = DenseNet([self.edge_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        else:
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            if(idx == 0):
                in_dims = self.in_dims
            else:
                in_dims = self.out_dims
            self.blocks.append(NNConvEdges(in_dims, self.out_dims, kernel))

    def forward(self, nodes, edge_index, edge_attr, a=None):
        for idx, block in enumerate(self.blocks):
            nodes = block(nodes, edge_index, edge_attr, a=a)
            if(idx < len(self.blocks) - 1):
                nodes = self.activation(nodes)
        return nodes
    
"""Utilities"""
class CombineInitAndEdges(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge_index, edge_attr, init):
        src = init[edge_index[0], ...]
        dst = init[edge_index[1], ...]
        return torch.cat((edge_attr, src, dst), dim=-1)
