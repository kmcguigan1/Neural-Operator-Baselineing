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
        self.edge_dims = 5
        self.mlp_ratio = config.get("MLP_RATIO", None)
        # create the layers
        self.projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.decoder = MLP(self.latent_dims, 1, self.latent_dims // 2)
        self.block = GNOBlockEfficient(
            self.latent_dims,
            self.edge_dims,
            self.depth,
            mlp_ratio=self.mlp_ratio
        )

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        # project the data
        nodes = torch.cat((nodes, grid), dim=-1)
        nodes = self.projector(nodes)
        nodes = F.gelu(nodes)

        # run the gno blocks
        nodes = self.block(nodes, edge_index, edge_attr)
        nodes = F.gelu(nodes)
        
        nodes = self.decoder(nodes)
        return nodes