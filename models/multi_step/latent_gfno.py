import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

from models.layers.base_layers import MLP
from models.layers.fourier_blocks_dim_last import TokenFNOBranch
from models.layers.graph_blocks import GNOBlock, GNOBlockAddNodesToEdge

class GFNOBlock(nn.Module):
    def __init__(self, latent_dims, modes1, modes2, kernel_dims:int, edge_dims:int, graph_passes:int, padding_mode:str=None):
        super().__init__()
        # vars
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        self.padding_mode = padding_mode

        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.graph_passes = graph_passes

        # layers
        self.fno_block = TokenFNOBranch(self.latent_dims, self.modes1, self.modes2, mlp_ratio=None, padding_mode=self.padding_mode)
        self.gno_block = GNOBlockAddNodesToEdge(self.latent_dims, self.latent_dims, self.kernel_dims, self.edge_dims, self.graph_passes)
        self.norm = nn.LayerNorm(self.latent_dims)

    def forward(self, nodes, edge_index, edge_attrs, batch_size, image_size):
        x1 = self.fno_block(nodes, batch_size, image_size)
        x2 = self.gno_block(nodes, edge_index, edge_attrs)
        return self.norm(F.gelu(x1 + x2))

class LatentGFNO(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.steps_out = config['TIME_STEPS_OUT']
        self.in_dims = self.steps_in + 2
        # fourier based information
        self.depth = config["DEPTH"]
        self.modes = (config['MODES1'], config['MODES2'])
        self.padding_mode = config.get('PADDING_MODE', None)
        # graph based information
        self.kernel_dims = config['KERNEL_DIMS']
        self.edge_dims = 5
        self.graph_passes = config['GRAPH_PASSES']
        # setup layers
        self.project = MLP(self.in_dims, self.latent_dims, self.latent_dims//2)
        self.decode = MLP(self.latent_dims, 1, self.latent_dims//2)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            self.blocks.append(GFNOBlock(self.latent_dims, self.modes[0], self.modes[1], self.kernel_dims, self.edge_dims, self.graph_passes, padding_mode=self.padding_mode))

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # add the grid to the data
        nodes = torch.cat((nodes, grid), dim=-1)
        # project the data
        nodes = self.project(nodes)
        # setup the predictions
        predictions = torch.zeros(batch_size*image_size[0]*image_size[1], self.steps_out, self.latent_dims, device=nodes.device, dtype=torch.float32)
        # go thorugh the blocks
        for step in range(self.steps_out):
            for block in self.blocks:
                nodes = block(nodes, edge_index, edge_attr, batch_size, image_size)
            predictions[:, step, :] = nodes
        # decode the prediction
        predictions = self.decode(predictions)
        predictions = torch.squeeze(predictions, dim=-1)
        # return the predictions
        return predictions