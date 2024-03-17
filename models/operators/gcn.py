import torch
from torch_geometric.nn import GCNConv, NNConv

class GCN_Net(torch.nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.depth = config['DEPTH']
        self.in_dims = config['TIME_STEPS_IN'] + 2
        self.out_dims = 1
        self.latent_dims = config['LATENT_DIMS']
        self.kernel_dims = config['KERNEL_DIMS']

        self.fc_in = torch.nn.Linear(self.in_dims, self.latent_dims)

        self.conv1 = GCNConv(self.latent_dims, self.latent_dims)
        self.conv2 = GCNConv(self.latent_dims, self.latent_dims)
        self.conv3 = GCNConv(self.latent_dims, self.latent_dims)
        self.conv4 = GCNConv(self.latent_dims, self.latent_dims)


        self.fc_out1 = torch.nn.Linear(self.latent_dims, self.kernel_dims)
        self.fc_out2 = torch.nn.Linear(self.kernel_dims, self.out_dims)

    def forward(self, x, grid, edge_index, edge_attr):
        x = torch.cat((x, grid), dim=-1)
        x = self.fc_in(x)

        for t in range(self.depth):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = self.conv4(x, edge_index)
            x = F.relu(x)

        x = F.relu(self.fc_out1(x))
        x = self.fc_out2(x)
        return x