import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.layers.graph_blocks import GNOBlock, GNOBlockAddNodesToEdge, GNOBlockAddInitNodesToEdge
from models.layers.fourier_blocks import FNOBlockWithW

class GINO(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        self.out_dims = 1
        # fourier based information
        self.depth = config["DEPTH"]
        self.mlp_ratio = 1
        self.modes = (config['MODES1'], config['MODES2'])
        # graph based information
        self.add_nodes_to_edge = config.get("ADD_NODES_TO_EDGE", False)
        self.add_init_to_edge = config.get("ADD_INIT_TO_EDGE", False)
        self.graph_passes = config['GRAPH_PASSES']
        self.edge_dims = 5
        self.kernel_dims = config['KERNEL_DIMS']
        # check that we only pick one edge thing
        assert (self.add_nodes_to_edge and self.add_init_to_edge) == False
        if(self.add_init_to_edge):
            self.edge_dims += 2
        # setup layers
        if(self.add_nodes_to_edge):
            gno_block_class = GNOBlockAddNodesToEdge
        elif(self.add_init_to_edge):
            self.gno_block_class = GNOBlockAddInitNodesToEdge
        else:
            gno_block_class = GNOBlock

        self.project1 = gno_block_class(self.in_dims, self.latent_dims, self.kernel_dims, self.edge_dims, 1)
        self.project2 = gno_block_class(self.latent_dims, self.latent_dims, self.kernel_dims, self.edge_dims, self.graph_passes)
        
        self.decode1 = gno_block_class(self.latent_dims, self.latent_dims, self.kernel_dims, self.edge_dims, self.graph_passes)
        self.decode2 = gno_block_class(self.latent_dims, self.out_dims, self.kernel_dims, self.edge_dims, 1)
        
        self.blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.blocks.append(FNOBlockWithW(self.latent_dims, self.modes[0], self.modes[1], self.mlp_ratio, padding_mode=None))
            self.blocks.append(nn.GELU())

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # check if we save the init data
        init = None
        if(self.add_init_to_edge):
            init = nodes[..., -1:].clone()
        # add the grid to the data
        nodes = torch.cat((nodes, grid), dim=-1)
        B, C = nodes.shape
        assert C == self.in_dims and B == batch_size * image_size[0] * image_size[1]
        # project the data
        nodes = self.project1(nodes, edge_index, edge_attr, a=init)
        nodes = F.gelu(nodes)
        nodes = self.project2(nodes, edge_index, edge_attr, a=init)
        # go thorugh the fno blocks
        nodes = rearrange(nodes, "(b h w) c -> b h w c", b=batch_size, h=image_size[0], w=image_size[1], c=self.latent_dims)
        nodes = rearrange(nodes, "b h w c -> b c h w", b=batch_size, h=image_size[0], w=image_size[1], c=self.latent_dims)
        for block in self.blocks:
            nodes = block(nodes)
        nodes = rearrange(nodes, "b c h w -> b h w c", b=batch_size, h=image_size[0], w=image_size[1], c=self.latent_dims)
        nodes = rearrange(nodes, "b h w c -> (b h w) c", b=batch_size, h=image_size[0], w=image_size[1], c=self.latent_dims)
        # decode the prediction
        nodes = self.decode1(nodes, edge_index, edge_attr, a=init)
        nodes = F.gelu(nodes)
        nodes = self.decode2(nodes, edge_index, edge_attr, a=init)
        # return the predictions
        return nodes