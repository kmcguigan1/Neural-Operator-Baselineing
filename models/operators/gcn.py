from torch_geometric.nn import GCNConv, NNConv

class GCN_Net(torch.nn.Module):
    def __init__(self, width, ker_width, depth, in_width=1, out_width=1):
        super(GCN_Net, self).__init__()
        self.depth = depth
        self.width = width

        self.fc_in = torch.nn.Linear(in_width, width)

        self.conv1 = GCNConv(width, width)
        self.conv2 = GCNConv(width, width)
        self.conv3 = GCNConv(width, width)
        self.conv4 = GCNConv(width, width)


        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.fc_in(data.x)

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