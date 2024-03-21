import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
class NeuralOperator(nn.Module):
    def __init__(self, modes1, modes2, width, activation, padding_mode=None):
        super(NeuralOperator, self).__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2, padding_mode=padding_mode)
        self.mlp = MLP(width, width, width)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)
        if(activation == 'gelu'):
            self.activation = nn.GELU()
        elif(activation == 'none'):
            self.activation = nn.Identity()
        else:
            raise Exception(f'Invalid activation specified {activation}')

    def forward(self, x):
        x1 = self.norm(self.conv(self.norm(x)))
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x = self.activation(x1 + x2)
        return x
    
class FNO2d(nn.Module):
    def __init__(self, config:dict):
        super(FNO2d, self).__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.steps_in = config['TIME_STEPS_IN']
        self.steps_out = config['TIME_STEPS_OUT']
        self.in_dims = self.steps_in + 2
        self.mlp_ratio = 1
        self.depth = config['DEPTH']
        self.modes = (config['MODES1'], config['MODES2'])
        self.padding_mode = config.get('PADDING', None)
        # dropout information
        self.drop_rate = 0.0
        # setup layers
        self.project = nn.Linear(self.in_dims, self.latent_dims)
        self.decode = self.q = MLP(self.latent_dims, 1, self.latent_dims * 2)
        # add the neural operator blocks
        activations = ['gelu' for _ in range(self.depth - 1)]
        activations.append('none')
        self.blocks = nn.ModuleList([
            NeuralOperator(self.modes[0], self.modes[1], self.latent_dims, activation=activation, padding_mode=padding_mode) for activation in activations
        ])

    def forward(self, xx, grid):
        # grid = self.get_grid(xx.shape, xx.device)
        # add the grid to the data
        x = torch.cat((xx, grid), dim=-1)
        # project the data
        x = self.project(x)
        x = x.permute(0, 3, 1, 2)
        # pad the inputs if that is what we are doing
        if(self.padding_mode == 'ONCE'):
            padding = int(math.sqrt(x.size(-1)))
            x = F.pad(x, [0, padding, 0, padding])
        # go thorugh the blocks
        for block in self.blocks:
            x = block(x)
        # decode the prediction
        if(self.padding_mode == 'ONCE'):
            x = x[..., :-padding, :-padding]
        x = self.decode(x)
        x = x.permute(0, 2, 3, 1)
        # return the predictions
        return x

# from functools import partial
# NormLayerPartial = partial(nn.LayerNorm, eps=1e-6)

# class SpectralConv2d(nn.Module):
#     """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#     """
#     def __init__(self, in_channels, out_channels, modes1, modes2, layer_name):
#         super(SpectralConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

#         self.layer_name = layer_name

#     # Complex multiplication
#     def compl_mul2d(self, input, weights):
#         # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
#         result = torch.einsum("bixy,ioxy->boxy", input, weights)
#         return result

#     def forward(self, x):
#         batchsize = x.shape[0]
#         #Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfft2(x)
#         # x_ft = torch.fft.rfftn(x, dim=(-2, -1))

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

#         #Return to physical space
#         x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
#         return x

# class ConvMLP(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels, layer_name):
#         super(ConvMLP, self).__init__()
#         self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
#         self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
#         self.layer_name = layer_name 

#     def forward(self, x):
#         x = self.mlp1(x)
#         x = F.gelu(x)
#         x = self.mlp2(x)
#         return x

# class MLP(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels, layer_name):
#         super(MLP, self).__init__()
#         self.mlp1 = nn.Linear(in_channels, mid_channels)
#         self.mlp2 = nn.Linear(mid_channels, out_channels)
#         self.layer_name = layer_name 

#     def forward(self, x):
#         x = self.mlp1(x)
#         x = F.gelu(x)
#         x = self.mlp2(x)
#         return x

# class NeuralOperator(nn.Module):
#     def __init__(self, modes1, modes2, width, layer_number, activation='none'):
#         super(NeuralOperator, self).__init__()
#         self.layer_name = f"Neural-Operator-Block{layer_number}"
#         self.conv = SpectralConv2d(width, width, modes1, modes2, f"{self.layer_name}-Conv")
#         self.mlp = ConvMLP(width, width, width, f"{self.layer_name}-MLP")
#         self.w = nn.Conv2d(width, width, 1)
#         self.norm = nn.InstanceNorm2d(width)
#         # self.norm1 = NormLayerPartial()
#         # self.norm2 = NormLayerPartial()
#         self.activation = activation

#     def forward(self, x):
#         x1 = self.norm(self.conv(self.norm(x)))
#         x1 = self.mlp(x1)
#         x2 = self.w(x)
#         x = x1 + x2
#         if(self.activation == 'gelu'):
#             x = F.gelu(x)
#         return x

# class FNO2d(nn.Module):
#     def __init__(self, config: dict):
#         super(FNO2d, self).__init__()
#         # save needed vars
#         self.latent_dims = config["LATENT_DIMS"]
#         self.steps_in = config["TIME_STEPS_IN"]
#         self.steps_out = config["TIME_STEPS_OUT"]
#         self.in_dims = self.steps_in + 2
#         self.mlp_ratio = config['MLP_RATIO']
#         self.depth = config['DEPTH']
#         self.modes = (config['MODES1'], config['MODES2'])
#         # dropout information
#         self.drop_rate = config["DROP_RATE"]
#         # setup layers
#         self.project = MLP(self.in_dims, config['LATENT_DIMS'], config['LATENT_DIMS'] * self.mlp_ratio, "Projection Layer")
#         self.decode = MLP(config['LATENT_DIMS'], 1, config['LATENT_DIMS'] * self.mlp_ratio, "UnProjection Layer")
#         # add the neural operator blocks
#         activations = ['gelu' if idx < self.depth - 1 else 'none' for idx in range(self.depth)]
#         self.blocks = nn.ModuleList([
#             NeuralOperator(self.modes[0], self.modes[1], self.latent_dims, idx+1, activation=activation) for idx, activation in enumerate(activations)
#         ])

#     def forward(self, xx, grid):
#         # add the grid to the data
#         x = torch.cat((xx, grid), dim=-1)
#         # project the data
#         x = self.project(x)
#         x = x.permute(0, 3, 1, 2)
#         # go thorugh the blocks
#         for block in self.blocks:
#             x = block(x)
#         # decode the prediction
#         x = x.permute(0, 2, 3, 1)
#         x = self.decode(x)
#         # return the predictions
#         return x