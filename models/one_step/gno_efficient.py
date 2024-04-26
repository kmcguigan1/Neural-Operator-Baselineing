import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.layers.base_layers import MLP
from models.layers.graph_blocks import GNOBlockEfficient

class GNOEfficient(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        self.out_dims = 1
        # graph based information
        self.depth = config["DEPTH"]
        self.graph_passes = config['GRAPH_PASSES']
        self.edge_dims = 5
        self.mlp_ratio = config.get("MLP_RATIO", 2)
        # create the layers
        self.projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.decoder = MLP(self.latent_dims, 1, self.latent_dims // 2)
        self.blocks = nn.ModuleList([
            GNOBlockEfficient(
                self.latent_dims,
                self.edge_dims,
                self.graph_passes,
                mlp_ratio=self.mlp_ratio,
                apply_to_output=True
            )
        for _ in range(self.depth)])

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # project the data
        nodes = torch.cat((nodes, grid), dim=-1)
        nodes = self.projector(nodes)
        nodes = F.gelu(nodes)

        # run the gno blocks
        for block in self.blocks:
            nodes = block(nodes, edge_index, edge_attr)
        
        nodes = self.decoder(nodes)
        return nodes