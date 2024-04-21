import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.layers.base_layers import MLP
from models.layers.graph_blocks import GNOBlock, GNOBlockAddNodesToEdge, GNOBlockSingleConv, GNOBlockSingleConvAddNodesToEdge, CombineInitAndEdges

class GNO(torch.nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        self.out_dims = 1
        # graph based information
        self.depth = config["DEPTH"]
        self.use_single_conv = config.get("USE_SINGLE_CONV", False)
        self.add_nodes_to_edge = config.get("ADD_NODES_TO_EDGE", False)
        self.add_init_to_edge = config.get("ADD_INIT_TO_EDGE", False)
        self.edge_dims = 5
        self.kernel_dims = config['KERNEL_DIMS']
        # check that we only pick one edge thing
        assert (self.add_nodes_to_edge and self.add_init_to_edge) == False
        if(self.add_init_to_edge):
            self.edge_dims += 2 # * self.steps_in
        # create the layers
        self.projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.decoder = MLP(self.latent_dims, 1, self.latent_dims // 2)

        if(self.use_single_conv):
            if(self.add_nodes_to_edge):
                gno_block_class = GNOBlockSingleConvAddNodesToEdge
            else:
                gno_block_class = GNOBlockSingleConv
        else:
            if(self.add_nodes_to_edge):
                gno_block_class = GNOBlockAddNodesToEdge
            else:
                gno_block_class = GNOBlock

        self.block = gno_block_class(
            in_dims=self.latent_dims, 
            out_dims=self.latent_dims, 
            kernel_dims=self.kernel_dims, 
            edge_dims=self.edge_dims, 
            depth=self.depth
        )

        if(self.add_init_to_edge):
            self.edge_attr_updater = CombineInitAndEdges()
        

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # check if we save the init data
        if(self.add_init_to_edge):
            edge_attr = self.edge_attr_updater(edge_index, edge_attr, nodes[..., -1:])
        
        # project the data
        nodes = torch.cat((nodes, grid), dim=-1)
        nodes = self.projector(nodes)
        nodes = F.gelu(nodes)

        # run the gno blocks
        nodes = self.block(nodes, edge_index, edge_attr)
        
        nodes = self.decoder(nodes)
        return nodes