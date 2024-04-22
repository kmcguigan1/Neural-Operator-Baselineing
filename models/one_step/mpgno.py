from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.layers.base_layers import MLP
from models.layers.graph_blocks import GNOBlock, GNOBlockAddNodesToEdge, GNOBlockSingleConv, GNOBlockSingleConvAddNodesToEdge, CombineInitAndEdges

class MPGNO(torch.nn.Module):
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
            self.edge_attr_updater = CombineInitAndEdges()
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
            
        gno_block_class_partial = partial(gno_block_class, in_dims=self.latent_dims, out_dims=self.latent_dims, kernel_dims=self.kernel_dims, edge_dims=self.edge_dims, depth=self.depth, shorten_kernel=True)

        self.k11 = gno_block_class_partial()
        self.k22 = gno_block_class_partial()
        self.k33 = gno_block_class_partial()

        self.k12 = gno_block_class_partial()
        self.k23 = gno_block_class_partial()

        self.k32 = gno_block_class_partial()
        self.k21 = gno_block_class_partial()

    def forward(self, nodes, grid, edge_index_1, edge_index_2, edge_index_3, edge_attr_1, edge_attr_2, edge_attr_3, batch_size, image_size):
        # check if we save the init data
        if(self.add_init_to_edge):
            edge_attr_1 = self.edge_attr_updater(edge_index_1, edge_attr_1, nodes[..., -1:])
            edge_attr_2 = self.edge_attr_updater(edge_index_2, edge_attr_2, nodes[..., -1:])
            edge_attr_3 = self.edge_attr_updater(edge_index_3, edge_attr_3, nodes[..., -1:])
        
        # project the data
        nodes = torch.cat((nodes, grid), dim=-1)
        nodes = self.projector(nodes)
        nodes = F.gelu(nodes)

        # run the downsmaple blocks
        nodes_11 = F.gelu(self.k11(nodes, edge_index_1, edge_attr_1))
        nodes_12 = F.gelu(self.k12(nodes, edge_index_1, edge_attr_1))

        nodes_22 = F.gelu(self.k22(nodes_12, edge_index_2, edge_attr_2))
        nodes_23 = F.gelu(self.k23(nodes_12, edge_index_2, edge_attr_2))

        nodes_33 = F.gelu(self.k33(nodes_23, edge_index_3, edge_attr_3))

        nodes_32 = F.gelu(self.k32(nodes_33, edge_index_2, edge_attr_2))
        
        nodes_21 = F.gelu(self.k21(nodes_32 + nodes_22, edge_index_1, edge_attr_1))

        nodes = nodes_21 + nodes_11
        
        nodes = self.decoder(nodes)
        return nodes
    
class MPGNOCust(torch.nn.Module):
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
        if(self.add_init_to_edge and False):
            self.edge_dims += 2 # * self.steps_in
            self.edge_attr_updater = CombineInitAndEdges()
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
            
        gno_block_class_partial = partial(gno_block_class, in_dims=self.latent_dims, out_dims=self.latent_dims, kernel_dims=self.kernel_dims, edge_dims=self.edge_dims, depth=self.depth, shorten_kernel=True, apply_to_output=True)

        self.k11 = gno_block_class_partial()
        self.k22 = gno_block_class_partial()
        self.k33 = gno_block_class_partial()

        self.k12 = gno_block_class_partial()
        self.k23 = gno_block_class_partial()

        self.k32 = gno_block_class_partial()
        self.k21 = gno_block_class_partial()

    def forward(
            self, 
            nodes, 
            grid, 
            edge_index_11, 
            edge_index_12, 
            edge_index_22, 
            edge_index_23, 
            edge_index_33,
            edge_index_32, 
            edge_index_21,
            edge_attr_11, 
            edge_attr_12, 
            edge_attr_22, 
            edge_attr_23, 
            edge_attr_33,
            edge_attr_32, 
            edge_attr_21, 
            batch_size, 
            image_size
        ):
        # check if we save the init data
        if(self.add_init_to_edge and False):
            edge_attr_11 = self.edge_attr_updater(edge_index_11, edge_attr_11, nodes[..., -1:])
            edge_attr_12 = self.edge_attr_updater(edge_index_12, edge_attr_12, nodes[..., -1:])
            edge_attr_22 = self.edge_attr_updater(edge_index_22, edge_attr_22, nodes[..., -1:])
            edge_attr_23 = self.edge_attr_updater(edge_index_23, edge_attr_23, nodes[..., -1:])
            edge_attr_33 = self.edge_attr_updater(edge_index_33, edge_attr_33, nodes[..., -1:])
            edge_attr_32 = self.edge_attr_updater(edge_index_32, edge_attr_32, nodes[..., -1:])
            edge_attr_21 = self.edge_attr_updater(edge_index_21, edge_attr_21, nodes[..., -1:])
        
        # project the data
        nodes = torch.cat((nodes, grid), dim=-1)
        nodes = self.projector(nodes)
        nodes = F.gelu(nodes)

        # run the downsmaple blocks
        nodes_11 = self.k11(nodes, edge_index_11, edge_attr_11)
        nodes_12 = self.k12(nodes, edge_index_12, edge_attr_12)

        nodes_22 = self.k22(nodes_12, edge_index_22, edge_attr_22)
        nodes_23 = self.k23(nodes_12, edge_index_23, edge_attr_23)

        nodes_33 = self.k33(nodes_23, edge_index_33, edge_attr_33)

        nodes_32 = self.k32(nodes_33, edge_index_32, edge_attr_32)
        
        nodes_21 = self.k21(nodes_32 + nodes_22, edge_index_21, edge_attr_21)

        nodes = nodes_21 + nodes_11
        
        nodes = self.decoder(nodes)
        return nodes