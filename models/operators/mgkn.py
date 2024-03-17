import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, NNConv

class MKGN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, points, level, in_width=1, out_width=1):
        super(MKGN, self).__init__()
        self.depth = depth
        self.width = width
        self.level = level

        index = 0
        self.points = [0]
        for point in points:
            index = index + point
            self.points.append(index)
        print(level, self.points)

        self.points_total = np.sum(points)

        # in (P)
        self.fc_in = torch.nn.Linear(in_width, width)

        # K12 K23 K34 ...
        self.conv_down_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_down_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_down_list = torch.nn.ModuleList(self.conv_down_list)

        # K11 K22 K33
        self.conv_list = []
        for l in range(level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=True, bias=False))
        self.conv_list = torch.nn.ModuleList(self.conv_list)

        # K21 K32 K43
        self.conv_up_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_up_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_up_list = torch.nn.ModuleList(self.conv_up_list)

        # out (Q)
        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)


    def forward(self, data):
        edge_index_down, edge_attr_down, range_down = data.edge_index_down, data.edge_attr_down, data.edge_index_down_range
        edge_index_mid, edge_attr_mid, range_mid = data.edge_index_mid, data.edge_attr_mid, data.edge_index_range
        edge_index_up, edge_attr_up, range_up = data.edge_index_up, data.edge_attr_up, data.edge_index_up_range

        x = self.fc_in(data.x)

        for t in range(self.depth):
            #downward
            for l in range(self.level-1):
                x = x + self.conv_down_list[l](x, edge_index_down[:,range_down[l,0]:range_down[l,1]], edge_attr_down[range_down[l,0]:range_down[l,1],:])
                x = F.relu(x)

            #upward
            for l in reversed(range(self.level)):
                x[self.points[l]:self.points[l+1]] = self.conv_list[l](x[self.points[l]:self.points[l+1]].clone(),
                                                                       edge_index_mid[:,range_mid[l,0]:range_mid[l,1]]-self.points[l],
                                                                        edge_attr_mid[range_mid[l,0]:range_mid[l,1],:])

                if l > 0:
                    x = x + self.conv_up_list[l-1](x, edge_index_up[:,range_up[l-1,0]:range_up[l-1,1]], edge_attr_up[range_up[l-1,0]:range_up[l-1,1],:])
                    x = F.relu(x)

        x = F.relu(self.fc_out1(x[:self.points[1]]))
        x = self.fc_out2(x)
        return x

class KernelInduced(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, points, level, in_width=1, out_width=1):
        super(KernelInduced, self).__init__()
        self.depth = depth
        self.width = width
        self.level = level
        self.points = points
        self.points_total = np.sum(points)

        # in
        self.fc_in = torch.nn.Linear(in_width, width)
        # self.fc_in_list = []
        # for l in range(level):
        #     self.fc_in_list.append(torch.nn.Linear(in_width, width))
        # self.fc_in_list = torch.nn.ModuleList(self.fc_in_list)

        # K12 K23 K34 ...
        self.conv_down_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_down_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_down_list = torch.nn.ModuleList(self.conv_down_list)

        # K11 K22 K33
        self.conv_list = []
        for l in range(level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_list = torch.nn.ModuleList(self.conv_list)

        # K21 K32 K43
        self.conv_up_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_up_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_up_list = torch.nn.ModuleList(self.conv_up_list)

        # out
        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)


    def forward(self, data):
        edge_index_down, edge_attr_down, range_down = data.edge_index_down, data.edge_attr_down, data.edge_index_down_range
        edge_index_mid, edge_attr_mid, range_mid = data.edge_index_mid, data.edge_attr_mid, data.edge_index_range
        edge_index_up, edge_attr_up, range_up = data.edge_index_up, data.edge_attr_up, data.edge_index_up_range

        x = self.fc_in(data.x)

        for t in range(self.depth):
            #downward
            for l in range(self.level-1):
                x = x + self.conv_down_list[l](x, edge_index_down[:,range_down[l,0]:range_down[l,1]], edge_attr_down[range_down[l,0]:range_down[l,1],:])
                x = F.relu(x)

            #upward
            for l in reversed(range(self.level)):
                x = x + self.conv_list[l](x, edge_index_mid[:,range_mid[l,0]:range_mid[l,1]], edge_attr_mid[range_mid[l,0]:range_mid[l,1],:])
                x = F.relu(x)
                if l > 0:
                    x = x + self.conv_up_list[l-1](x, edge_index_up[:,range_up[l-1,0]:range_up[l-1,1]], edge_attr_up[range_up[l-1,0]:range_up[l-1,1],:])
                    x = F.relu(x)


        x = F.relu(self.fc_out1(x[:self.points[0]]))
        x = self.fc_out2(x)
        return x