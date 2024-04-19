import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.layers.base_layers import MLP
from models.layers.graph_blocks import GNOBlock, GNOBlockAddNodesToEdge, GNOBlockAddInitNodesToEdge, GNOBlockSingleConv, GNOBlockSingleConvAddNodesToEdge, GNOBlockSingleConvAddInitNodesToEdge

class GNO(torch.nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        self.out_dims = 1
        # fourier based information
        self.depth = config["DEPTH"]
        # graph based information
        self.use_single_conv = config.get("USE_SINGLE_CONV", False)
        self.add_nodes_to_edge = config.get("ADD_NODES_TO_EDGE", False)
        self.add_init_to_edge = config.get("ADD_INIT_TO_EDGE", False)
        self.graph_passes = config['GRAPH_PASSES']
        self.edge_dims = 5
        self.kernel_dims = config['KERNEL_DIMS']
        # check that we only pick one edge thing
        assert (self.add_nodes_to_edge and self.add_init_to_edge) == False
        if(self.add_init_to_edge):
            self.edge_dims += 2 * self.steps_in
        # create the layers
        self.projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.decoder = MLP(self.latent_dims, 1, self.latent_dims // 2)

        if(self.use_single_conv):
            if(self.add_nodes_to_edge):
                gno_block_class = GNOBlockSingleConvAddNodesToEdge
            elif(self.add_init_to_edge):
                gno_block_class = GNOBlockSingleConvAddInitNodesToEdge
            else:
                gno_block_class = GNOBlockSingleConv
        else:
            if(self.add_nodes_to_edge):
                gno_block_class = GNOBlockAddNodesToEdge
            elif(self.add_init_to_edge):
                gno_block_class = GNOBlockAddInitNodesToEdge
            else:
                gno_block_class = GNOBlock

        print(gno_block_class)

        self.blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.blocks.append(
                gno_block_class(
                    in_dims=self.latent_dims, 
                    out_dims=self.latent_dims, 
                    kernel_dims=self.kernel_dims, 
                    edge_dims=self.edge_dims, 
                    depth=self.graph_passes
                )
            )
        

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # check if we save the init data
        init = None
        if(self.add_init_to_edge):
            init = nodes[..., -1:].clone()
            # init = nodes.clone()
        
        # project the data
        nodes = torch.cat((nodes, grid), dim=-1)
        nodes = self.projector(nodes)
        nodes = F.gelu(nodes)

        # run the gno blocks
        for idx, block in enumerate(self.blocks):
            nodes = block(nodes, edge_index, edge_attr, a=init)
            if(idx < self.depth - 1):
                nodes = F.gelu(nodes)
        
        nodes = self.decoder(nodes)
        return nodes