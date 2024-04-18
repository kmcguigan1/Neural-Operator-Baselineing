import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

import math

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, padding_mode=None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.padding_mode = padding_mode

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        if(self.padding_mode == 'EVERY_SINGLE'):
            padding = int(math.sqrt(x.size(1)))
            x = F.pad(x, [0, padding, 0, padding, 0, 0])
        elif(self.padding_mode == 'EVERY_DUAL'):
            padding = int(math.sqrt(x.size(1)) // 2)
            x = F.pad(x, [padding, padding, padding, padding, 0, 0])
            
        x_ft = torch.fft.rfftn(x, dim=(1, 2), norm="ortho")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize,  x.size(-2), x.size(-1)//2 + 1, self.out_channels, dtype=torch.cfloat, device=x.device)
        out_ft[:, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :self.modes1, :self.modes2, :], self.weights1)
        out_ft[:, -self.modes1:, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, -self.modes1:, :self.modes2, :], self.weights2)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, dim=(1, 2), s=(x.size(-2), x.size(-1)), norm="ortho")

        if(self.padding_mode == 'EVERY_SINGLE'):
            x = x[:, :-padding, :-padding, :]
        elif(self.padding_mode == 'EVERY_DUAL'):
            x = x[:, padding:-padding, padding:-padding, :]
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
class NNConv_old(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, aggr='add', root_weight=True, bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x
    
class NeuralOperator(nn.Module):
    def __init__(self, modes1, modes2, width, kernel, image_size, activation, padding_mode=None):
        super(NeuralOperator, self).__init__()
        self.image_size = image_size
        # FNO branch
        self.conv = SpectralConv2d(width, width, modes1, modes2, padding_mode=padding_mode)
        self.w = MLP(width, width, width * 2)
        # GNO branch
        self.gno = NNConv_old(width, width, kernel)
        # mlp output
        self.mlp = MLP(width, width, width * 2)
        # global important info
        self.norm = nn.InstanceNorm2d(width)
        if(activation == 'gelu'):
            self.activation = nn.GELU()
        elif(activation == 'none'):
            self.activation = nn.Identity()
        else:
            raise Exception(f'Invalid activation specified {activation}')

    def forward(self, nodes, edge_index, edge_attr):
        B, H, W, C = nodes.shape
        assert H == self.image_size[0] and W == self.image_size[1]
        # FNO branch
        x = self.norm(nodes)
        x1 = self.conv(x)
        x1 = self.norm(x1)
        x1 = self.mlp(x1)
        # plain branch
        x2 = self.w(x)
        x2 = self.norm(x2)
        # GNO branch
        x3 = x.reshape(B, H * W, C)
        x3 = self.gno(x3, edge_index, edge_attr)
        x = self.activation(x1 + x2 + x3)
        return x
    
class FNOGNO(nn.Module):
    def __init__(self, config:dict, image_size:tuple):
        super().__init__()
        # save needed vars
        self.image_size = image_size
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.in_dims = self.steps_in + 2
        self.depth = config['DEPTH']
        self.modes = (config['MODES1'], config['MODES2'])
        self.padding_mode = config.get('PADDING_MODE', None)
        # graph based information
        self.edge_dims = 5
        self.kernel_dims = config['KERNEL_DIMS']
        # dropout information
        self.drop_rate = 0.0
        # setup layers
        self.project = MLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.decode = MLP(self.latent_dims, 1, self.latent_dims // 2)
        # add the neural operator blocks
        kernel = DenseNet([self.edge_dims, self.kernel_dims, self.kernel_dims, self.latent_dims**2], torch.nn.ReLU)
        activations = ['gelu' for _ in range(self.depth - 1)]
        activations.append('none')
        self.blocks = nn.ModuleList([
            NeuralOperator(self.modes[0], self.modes[1], self.latent_dims, kernel, self.image_size, activation=activation, padding_mode=self.padding_mode) for activation in activations
        ])

    def forward(self, xx, grid):
        # grid = self.get_grid(xx.shape, xx.device)
        # add the grid to the data
        x = torch.cat((xx, grid), dim=-1)
        B, T, C = x.shape
        assert C == self.in_dims and T == self.image_size[0] * self.image_size[1]
        # we should now make the data into a grid, we can flatten in the gno block
        x = x.reshape(B, self.image_size[0], self.image_size[1], C)
        # project the data
        x = self.project(x)
        # pad the inputs if that is what we are doing
        if(self.padding_mode == 'ONCE_SINGLE'):
            padding = int(math.sqrt(x.size(1)))
            x = F.pad(x, [0, padding, 0, padding, 0, 0])
        elif(self.padding_mode == 'ONCE_DUAL'):
            padding = int(math.sqrt(x.size(1)) // 2)
            x = F.pad(x, [padding, padding, padding, padding, 0, 0])
        # go thorugh the blocks
        for block in self.blocks:
            x = block(x)
        # decode the prediction
        if(self.padding_mode == 'ONCE_SINGLE'):
            x = x[:, :-padding, :-padding, :]
        elif(self.padding_mode == 'ONCE_DUAL'):
            x = x[:, padding:-padding, padding:-padding, :]
        x = self.decode(x)
        x = x.reshape(B, self.image_size[0] * self.image_size[1], 1)
        # return the predictions
        return x