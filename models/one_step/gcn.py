import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.base_layers import MLP
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # variables
        self.in_dims = config['TIME_STEPS_IN'] + 2
        self.latent_dims = config['LATENT_DIMS']
        self.out_dims = 1
        self.depth = config['DEPTH']
        # layers
        self.projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.decoder = MLP(self.latent_dims, self.out_dims, self.latent_dims // 2)
        
        self.blocks = nn.ModuleList([
            GCNConv(self.latent_dims, self.latent_dims) for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.latent_dims)

    def forward(self, nodes, grid, edge_index, edge_attr, batch_size, image_size):
        nodes = torch.cat((nodes, grid), dim=-1)
        nodes = self.projector(nodes)
        nodes = F.gelu(nodes)

        for idx, block in enumerate(self.blocks):
            nodes = block(nodes, edge_index)
            if(idx < self.depth - 1):
                nodes = F.gelu(nodes)
                nodes = self.norm(nodes)
        
        nodes = self.decoder(nodes)
        return nodes