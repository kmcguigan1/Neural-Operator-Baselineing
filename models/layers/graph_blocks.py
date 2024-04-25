import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

from models.layers.base_layers import MLP

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

    def forward(self, x, edge_index, edge_attr):
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
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, pseudo=edge_attr) 
    
    def message(self, x_i, x_j, pseudo):
        pseudo = torch.cat((pseudo, x_i, x_j), dim=-1)
        weight = self.kernel(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=True):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                if normalize:
                    # self.layers.append(nn.BatchNorm1d(layers[j+1]))
                    self.layers.append(nn.LayerNorm(layers[j+1]))
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x
    
"""SINGLE CONV GNO BLOCKS"""
class GNOBlockSigleConvBase(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool, apply_to_output:bool):
        super().__init__()
        # save some variables
        assert in_dims == out_dims
        self.latent_dims = in_dims
        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.depth = depth
        self.activation = F.gelu
        self.norm = nn.LayerNorm(self.latent_dims)
        self.shorten_kernel = shorten_kernel
        self.apply_to_output = apply_to_output

    def forward(self, *args):
        raise NotImplementedError('')

class GNOBlockSingleConv(GNOBlockSigleConvBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool=False, apply_to_output:bool=False):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth, shorten_kernel, apply_to_output)
        if(self.shorten_kernel):
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.latent_dims**2], torch.nn.ReLU)
        else:
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.ReLU)
        self.conv = NNConv(self.latent_dims, self.latent_dims, kernel)

    def forward(self, nodes, edge_index, edge_attr):
        for idx in range(self.depth):
            nodes = self.conv(nodes, edge_index, edge_attr)
            if(idx < self.depth - 1 or self.apply_to_output):
                nodes = self.activation(nodes)
                nodes = self.norm(nodes)
        return nodes
    
class GNOBlockSingleConvAddNodesToEdge(GNOBlockSigleConvBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool=False, apply_to_output:bool=False):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth, shorten_kernel, apply_to_output)
        self.edge_dims += 2 * self.latent_dims
        if(self.shorten_kernel):
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.latent_dims**2], torch.nn.ReLU)
        else:
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.ReLU)
        self.conv = NNConvEdges(self.latent_dims, self.latent_dims, kernel)

    def forward(self, nodes, edge_index, edge_attr):
        for idx in range(self.depth):
            nodes = self.conv(nodes, edge_index, edge_attr)
            if(idx < self.depth - 1 or self.apply_to_output):
                nodes = self.activation(nodes)
                nodes = self.norm(nodes)
        return nodes
    
"""MULTI CONV GNO BLOCKS"""
class GNOBlockBase(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool, apply_to_output:bool):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.depth = depth
        self.activation = F.gelu
        self.shorten_kernel = shorten_kernel
        self.norm = nn.LayerNorm(self.out_dims)
        self.apply_to_output = apply_to_output
    
    def forward(self, *args):
        raise NotImplementedError('')

class GNOBlock(GNOBlockBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool=False, apply_to_output:bool=False):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth, shorten_kernel, apply_to_output)
        if(self.shorten_kernel):
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        else:
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            if(idx == 0):
                in_dims = self.in_dims
            else:
                in_dims = self.out_dims
            self.blocks.append(NNConv(in_dims, self.out_dims, kernel))

    def forward(self, nodes, edge_index, edge_attr):
        for idx, block in enumerate(self.blocks):
            nodes = block(nodes, edge_index, edge_attr)
            if(idx < len(self.blocks) - 1 or self.apply_to_output):
                nodes = self.activation(nodes)
                nodes = self.norm(nodes)
        return nodes
    
class GNOBlockAddNodesToEdge(GNOBlockBase):
    def __init__(self, in_dims:int, out_dims:int, kernel_dims:int, edge_dims:int, depth:int, shorten_kernel:bool=False, apply_to_output:bool=False):
        super().__init__(in_dims, out_dims, kernel_dims, edge_dims, depth, shorten_kernel, apply_to_output)
        assert self.in_dims == self.out_dims or self.depth == 1
        self.edge_dims += 2 * self.in_dims
        if(self.shorten_kernel):
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        else:
            kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.in_dims*self.out_dims], torch.nn.ReLU)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            if(idx == 0):
                in_dims = self.in_dims
            else:
                in_dims = self.out_dims
            self.blocks.append(NNConvEdges(in_dims, self.out_dims, kernel))

    def forward(self, nodes, edge_index, edge_attr):
        for idx, block in enumerate(self.blocks):
            nodes = block(nodes, edge_index, edge_attr)
            if(idx < len(self.blocks) - 1 or self.apply_to_output):
                nodes = self.activation(nodes)
                nodes = self.norm(nodes)
        return nodes
    
"""Efficient Implementation"""
class GNOBlockEfficient(MessagePassing):
    def __init__(self, latent_dims, edge_dims, mlp_ratio:int=None, aggr:str='mean'):
        super().__init__(aggr=aggr)
        if(mlp_ratio is not None):
            raise Exception('')
            self.src_func = MLP(latent_dims, latent_dims, latent_dims*mlp_ratio)
            self.dst_func = MLP(latent_dims, latent_dims, latent_dims*mlp_ratio)
            self.edge_func = MLP(edge_dims, latent_dims, latent_dims*mlp_ratio)
            self.out_func = MLP(latent_dims, latent_dims, latent_dims*mlp_ratio)
        else:
            self.src_func = nn.Linear(latent_dims, latent_dims)
            self.dst_func = nn.Linear(latent_dims, latent_dims)
            self.edge_func = nn.Linear(edge_dims, latent_dims)
            self.out_func = nn.Linear(latent_dims, latent_dims)
    
        self.norm = nn.LayerNorm(latent_dims)

    def forward(self, x, edge_index, edge_attr):
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_new = self.norm(F.gelu(x_new))
        x_new = self.propagate(edge_index, x=x_new, edge_attr=edge_attr)
        x = x + x_new
        return x
    
    def message(self, x_i, x_j, edge_attr):
        x_i = self.dst_func(x_i)
        x_j = self.src_func(x_j)
        edge_attr = self.edge_func(edge_attr)
        return self.out_func(x_i + x_j + edge_attr)
    
"""Utilities"""
class CombineInitAndEdges(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge_index, edge_attr, init):
        src = init[edge_index[0], ...]
        dst = init[edge_index[1], ...]
        return torch.cat((edge_attr, src, dst), dim=-1)