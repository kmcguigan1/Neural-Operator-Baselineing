import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

from models.layers.base_layers import MLP
from models.layers.fourier_blocks_dim_last import TokenFNOBranch
from models.layers.graph_blocks import GNOBlockSingleConv, GNOBlockSingleConvAddNodesToEdge, CombineInitAndEdges, GNOBlock

class GFNOBlock(nn.Module):
    def __init__(self, latent_dims, modes1, modes2, kernel_dims:int, edge_dims:int, graph_passes:int, add_init_to_edge:bool, add_node_to_edge:bool, padding_mode:str=None):
        super().__init__()
        # vars
        self.latent_dims = latent_dims
        self.modes1 = modes1
        self.modes2 = modes2
        self.padding_mode = padding_mode

        self.kernel_dims = kernel_dims
        self.edge_dims = edge_dims
        self.graph_passes = graph_passes

        self.add_init_to_edge = add_init_to_edge
        self.add_node_to_edge = add_node_to_edge
        # layers
        self.fno_block = TokenFNOBranch(self.latent_dims, self.modes1, self.modes2, mlp_ratio=1, padding_mode=self.padding_mode)
        if(self.add_node_to_edge):
            self.gno_block = GNOBlockSingleConvAddNodesToEdge(self.latent_dims, self.latent_dims, self.kernel_dims, self.edge_dims, self.graph_passes)
        else:
            self.gno_block = GNOBlockSingleConv(self.latent_dims, self.latent_dims, self.kernel_dims, self.edge_dims, self.graph_passes)

    def forward(self, nodes, edge_index, edge_attrs, batch_size, image_size):
        x1 = self.fno_block(nodes, batch_size, image_size)
        x2 = self.gno_block(nodes, edge_index, edge_attrs)
        return F.gelu(x1 + x2)

class GFNO(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        # fourier based information
        self.depth = config["DEPTH"]
        self.modes = (config['MODES1'], config['MODES2'])
        self.padding_mode = config.get('PADDING_MODE', None)
        # graph based information
        self.kernel_dims = config['KERNEL_DIMS']
        self.edge_dims = 5
        self.graph_passes = config['GRAPH_PASSES']
        # check how we will treat the edges
        self.add_nodes_to_edge = config.get("ADD_NODES_TO_EDGE", False) and False
        self.add_init_to_edge = config.get("ADD_INIT_TO_EDGE", False) and False
        # assert (self.add_nodes_to_edge and self.add_init_to_edge) == False
        # if(self.add_init_to_edge):
        #     self.edge_dims += 2
        #     self.edge_attr_updater = CombineInitAndEdges()
        # setup layers
        self.project = MLP(self.in_dims, self.latent_dims, self.latent_dims//2)
        self.decode = MLP(self.latent_dims, 1, self.latent_dims//2)
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            self.blocks.append(GFNOBlock(self.latent_dims, self.modes[0], self.modes[1], self.kernel_dims, self.edge_dims, self.graph_passes, self.add_init_to_edge, self.add_nodes_to_edge, padding_mode=self.padding_mode))

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # if(self.add_init_to_edge):
        #     edge_attr = self.edge_attr_updater(edge_index, edge_attr, nodes[..., -1:])
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