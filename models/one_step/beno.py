import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GAT

from models.layers.base_layers import MLP
from models.layers.graph_blocks import GNOBlock, GNOBlockAddNodesToEdge, GNOBlockSingleConv, GNOBlockSingleConvAddNodesToEdge, CombineInitAndEdges

class BoundaryEncoder(nn.Module):
    def __init__(self, latent_dims, edge_dim:int, depth:int=4):
        super().__init__()
        self.edge_func = MLP(latent_dims, latent_dims, latent_dims)
        self.gat = GAT(
            latent_dims,
            latent_dims,
            depth,
            edge_dim=edge_dim, 
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.norm = nn.LayerNorm(latent_dims)

    def forward(self, bnd_nodes, bnd_edge_index, bnd_edge_attr):
        # run the attention and pool
        bnd_nodes = self.gat(bnd_nodes, bnd_edge_index, bnd_edge_attr)
        bnd_nodes = torch.permute(bnd_nodes, dims=[1, 0])
        bnd_nodes = self.pool(bnd_nodes)
        bnd_nodes = torch.permute(bnd_nodes, dims=[1, 0])
        return self.norm(bnd_nodes)

class InteractionNetwork(MessagePassing):
    def __init__(self, latent_dims, aggr='mean'):
        super().__init__(aggr=aggr)
        self.latent_dims = latent_dims
        self.aggr = aggr
        # setup the layers
        self.node_func = MLP(self.latent_dims*3, self.latent_dims, self.latent_dims)
        self.edge_func = MLP(self.latent_dims*3, self.latent_dims, self.latent_dims)
        self.node_norm = nn.LayerNorm(self.latent_dims)
        self.edge_norm = nn.LayerNorm(self.latent_dims)

    def forward(self, x, edge_index, edge_attr, boundary):
        x_new, edge_attr_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, boundary=boundary)
        x = self.node_norm(F.gelu(x + x_new))
        edge_attr = self.edge_norm(F.gelu(edge_attr + edge_attr_new))
        return x, edge_attr
    
    def message(self, x_i, x_j, edge_attr, boundary):
        edge_attr = torch.cat([x_i, x_j, edge_attr], dim=-1)
        edge_attr = self.edge_func(edge_attr)
        return edge_attr

    def update(self, x_updated, x, edge_attr, boundary):
        boundary = boundary.repeat(x.shape[0], 1)
        x_updated = torch.cat([x_updated, x, boundary], dim=-1)
        x_updated = self.node_func(x_updated)
        return x_updated, edge_attr

class BENO(torch.nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        self.out_dims = 1
        # graph based information
        self.depth = config["DEPTH"]
        self.bnd_depth = config["BND_DEPTH"]
        self.use_single_conv = config.get("USE_SINGLE_CONV", False)
        self.add_nodes_to_edge = config.get("ADD_NODES_TO_EDGE", False)
        self.add_init_to_edge = config.get("ADD_INIT_TO_EDGE", False)
        self.edge_dims = 7
        self.boundary_edge_dims = 5
        # check that we only pick one edge thing
        assert (self.add_nodes_to_edge and self.add_init_to_edge) == False
        if(self.add_init_to_edge):
            raise Exception('Not Implemented')
            self.edge_dims += 2 # * self.steps_in
        # create the layers
        self.projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.edge_projector = MLP(self.edge_dims, self.latent_dims, self.latent_dims // 2)
        
        self.external_projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.external_edge_projector = MLP(self.edge_dims, self.latent_dims, self.latent_dims // 2)

        self.boundary_projector = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.boundary_edge_projector = MLP(self.boundary_edge_dims, self.latent_dims, self.latent_dims // 2)
        
        self.decoder = MLP(self.latent_dims, 1, self.latent_dims // 2)
        self.external_decoder = MLP(self.latent_dims, 1, self.latent_dims // 2)

        self.boundary_encoder = BoundaryEncoder(self.latent_dims, self.latent_dims, depth=self.bnd_depth)

        self.internal_nodes_blocks = nn.ModuleList([
            InteractionNetwork(self.latent_dims) for _ in range(self.depth)
        ])

        self.external_nodes_blocks = nn.ModuleList([
            InteractionNetwork(self.latent_dims) for _ in range(self.depth)
        ])

        self.internal_norm = nn.LayerNorm(self.latent_dims)
        self.internal_edge_norm = nn.LayerNorm(self.latent_dims)
        self.external_norm = nn.LayerNorm(self.latent_dims)
        self.external_edge_norm = nn.LayerNorm(self.latent_dims)

    def forward(self, nodes, grid, edge_index, edge_attr, boundary_edge_index, boundary_edge_attr, boundary_node_index, boundary_node_mask, batch_size, image_size):
        nodes = torch.cat([nodes, grid], dim=-1)
        # project all data to latent space
        internal_nodes = F.gelu(self.projector(nodes))
        internal_edge_attr = F.gelu(self.edge_projector(edge_attr))

        external_nodes = F.gelu(self.external_projector(nodes))
        external_nodes = external_nodes * boundary_node_mask
        external_edge_attr = F.gelu(self.external_edge_projector(edge_attr))

        boundary_nodes = nodes[boundary_node_index, ...]
        boundary_nodes = F.gelu(self.boundary_projector(boundary_nodes))
        boundary_edge_attr = F.gelu(self.boundary_edge_projector(boundary_edge_attr))
        boundary = self.boundary_encoder(boundary_nodes, boundary_edge_index, boundary_edge_attr)

        # run the gno blocks
        for idx, block in enumerate(self.internal_nodes_blocks):
            internal_nodes, internal_edge_attr = block(internal_nodes, edge_index, internal_edge_attr, boundary)
            internal_nodes = self.internal_norm(internal_nodes)
            internal_edge_attr = self.internal_edge_norm(internal_edge_attr)

        for idx, block in enumerate(self.external_nodes_blocks):
            external_nodes, external_edge_attr = block(external_nodes, edge_index, external_edge_attr, boundary)
            internal_nodes = self.external_norm(external_nodes)
            internal_edge_attr = self.external_edge_norm(external_edge_attr)
        
        internal_nodes = self.decoder(internal_nodes)
        external_nodes = self.external_decoder(external_nodes)
        return internal_nodes + external_nodes