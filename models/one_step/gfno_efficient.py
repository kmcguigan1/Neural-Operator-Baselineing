import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

from models.layers.base_layers import MLP
from models.layers.fourier_blocks_dim_last import TokenFNOBranch
from models.layers.graph_blocks import GNOBlockEfficient

class GFNOBlock(nn.Module):
    def __init__(self, latent_dims, modes1, modes2, edge_dims:int, graph_passes:int, mlp_ratio:int=None):
        super().__init__()
        # vars
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        self.edge_dims = edge_dims
        self.graph_passes = graph_passes
        self.mlp_ratio = mlp_ratio
        # layers
        self.fno_block = TokenFNOBranch(self.latent_dims, self.modes1, self.modes2, mlp_ratio=mlp_ratio)
        self.gno_block = GNOBlockEfficient(self.latent_dims, self.edge_dims, self.graph_passes, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(self.latent_dims)

    def forward(self, nodes, edge_index, edge_attrs, batch_size, image_size):
        x1 = self.fno_block(nodes, batch_size, image_size)
        x2 = self.gno_block(nodes, edge_index, edge_attrs)
        return self.norm(F.gelu(x1 + x2))

class GFNOEfficient(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        # fourier based information
        self.depth = config["DEPTH"]
        self.modes = (config['MODES1'], config['MODES2'])
        # graph based information
        self.edge_dims = 5
        self.graph_passes = config['GRAPH_PASSES']
        # setup layers
        self.project = MLP(self.in_dims, self.latent_dims, self.latent_dims//2)
        self.decode = MLP(self.latent_dims, 1, self.latent_dims//2)
        self.blocks = nn.ModuleList([GFNOBlock(self.latent_dims, self.modes[0], self.modes[1], self.edge_dims, self.graph_passes) for _ in range(self.depth)])   

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # add the grid to the data
        nodes = torch.cat((nodes, grid), dim=-1)
        # project the data
        nodes = self.project(nodes)
        # go thorugh the blocks
        for block in self.blocks:
            nodes = block(nodes, edge_index, edge_attr, batch_size, image_size)
        # decode the prediction
        nodes = self.decode(nodes)
        # return the predictions
        return nodes