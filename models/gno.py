import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MLP(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, mid_dims:int=None):
        super().__init__()
        if(mid_dims is not None):
            self.double = True
            self.enc = nn.Linear(in_dims, mid_dims)
            self.dec = nn.Linear(mid_dims, out_dims)
        else:
            self.double = False
            self.linear = nn.Linear(in_dims, out_dims)
    
    def forward(self, x):
        if(self.double):
            x = self.enc(x)
            x = F.gelu(x)
            x = self.dec(x)
        else:
            x = self.linear(x)
        return x

class CustomMessagePassing(MessagePassing):
    def __init__(self, node_features:int, edge_features:int, mlp_ratio:int=1, output_mlp:str='none'):
        super().__init__(aggr='add')
        self.W = MLP(node_features, node_features, mid_dims=node_features*mlp_ratio)
        self.K = MLP(node_features * 2 + edge_features, node_features, mid_dims=node_features)
        if(output_mlp == 'none'):
            self.mode = 'none'
        elif(output_mlp == 'add'):
            self.mode = 'add'
            self.O = MLP(node_features, node_features, mid_dims=node_features*mlp_ratio)
        elif(output_mlp == 'concat'):
            self.mode = 'concat'
            self.O = MLP(node_features*2, node_features, mid_dims=node_features*mlp_ratio)
        else:
            raise Exception(f'Invalid output mlp mode {output_mlp}')
    
    def forward(self, x, edge_index, edge_features):
        x = self.propagate(edge_index, x=x, edge_features=edge_features)
        return x

    def message(self, x_i, x_j, edge_index, index):
        return x_j
        distances = torch.sqrt(torch.sum(torch.square(grid_i - grid_j), dim=-1, keepdims=True))
        edge_features = torch.cat((distances, grid_i, grid_j, x_i, x_j), dim=-1)
        return self.K(edge_features)

    def update(self, aggr_out, x):
        if(self.mode == 'none'):
            return F.gelu(self.W(x) + aggr_out)
        elif(self.mode == 'add'):
            return F.gelu(self.O(self.W(x) + aggr_out))
        elif(self.mode == 'concat'):
            return F.gelu(self.O(torch.concatenate((self.W(x), aggr_out), dim=-1)))
        else:
            raise Exception('')

class GNO(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.latent_dims = config["LATENT_DIMS"]
        self.steps_in = config["TIME_STEPS_IN"]
        self.steps_out = config["TIME_STEPS_OUT"]
        self.in_dims = self.steps_in + 2
        self.mlp_ratio = config['MLP_RATIO']
        self.depth = config['DEPTH']
        self.output_mode = config['OUTPUT_MODE'] if 'OUTPUT_MODE' in config.keys() else 'none'
        # dropout information
        self.drop_rate = config["DROP_RATE"]
        # setup layers
        self.project = MLP(self.in_dims, config['LATENT_DIMS'], config['LATENT_DIMS'] * self.mlp_ratio)
        self.decode = MLP(config['LATENT_DIMS'], 1, config['LATENT_DIMS'] * self.mlp_ratio)
        # create the graph layers
        self.blocks = nn.ModuleList([
            CustomMessagePassing(self.latent_dims, 5, mlp_ratio=self.mlp_ratio, output_mlp=self.output_mode)
        ])

    def forward(self, x, grid, edge_index, edge_features):
        edge_index = edge_index.to(torch.int64)
        x = torch.cat((x, grid), dim=-1)
        x = self.project(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_features)

        x = self.decode(x)
        return x


def test():
    layer = CustomMessagePassing(1, 2)
    # features
    x = torch.Tensor([[0],[1],[2],[3]])
    # edges
    edges = torch.tensor([
        [0, 1, 0, 2, 2, 3, 0],
        [1, 0, 2, 0, 3, 2, 3]
    ])
    edge_features = torch.rand((edges.shape[1], 4))

    layer.forward(x, edges, edge_features)

if __name__ == '__main__':
    test()
