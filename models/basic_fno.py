import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from functools import partial
NormLayerPartial = partial(nn.LayerNorm, eps=1e-6)

class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, layer_name):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        self.layer_name = layer_name

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        result = torch.einsum("bixy,ioxy->boxy", input, weights)
        return result

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # x_ft = torch.fft.rfftn(x, dim=(-2, -1))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class ConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, layer_name):
        super(ConvMLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.layer_name = layer_name 

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, layer_name):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.layer_name = layer_name 

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class NeuralOperator(nn.Module):
    def __init__(self, modes1, modes2, width, layer_number, activation='none'):
        super(NeuralOperator, self).__init__()
        self.layer_name = f"Neural-Operator-Block{layer_number}"
        self.conv = SpectralConv2d(width, width, modes1, modes2, f"{self.layer_name}-Conv")
        self.mlp = ConvMLP(width, width, width, f"{self.layer_name}-MLP")
        self.w = nn.Conv2d(width, width, 1)
        # self.norm = nn.InstanceNorm2d(width)
        # self.norm1 = NormLayerPartial()
        # self.norm2 = NormLayerPartial()
        self.activation = activation

    def forward(self, x):
        x1 = self.conv(x) #self.norm2(self.conv(self.norm1(x)))
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x = x1 + x2
        if(self.activation == 'gelu'):
            x = F.gelu(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, config: dict, img_size: tuple):
        super(FNO2d, self).__init__()
        # save needed vars
        self.img_size = img_size
        self.latent_dims = config["LATENT_DIMS"]
        self.steps_in = config["TIME_STEPS_IN"]
        self.steps_out = config["TIME_STEPS_OUT"]
        self.in_dims = self.steps_in + 2
        self.mlp_ratio = config['MLP_RATIO']
        self.depth = config['DEPTH']
        self.modes = (config['MODES1'], config['MODES2'])
        # dropout information
        self.drop_rate = config["DROP_RATE"]
        # setup layers
        self.project = MLP(self.in_dims, config['LATENT_DIMS'], config['LATENT_DIMS'] * self.mlp_ratio, "Projection Layer")
        self.decode = MLP(config['LATENT_DIMS'], 1, config['LATENT_DIMS'] * self.mlp_ratio, "UnProjection Layer")
        # add the neural operator blocks
        activations = ['gelu' if idx < self.depth - 1 else 'none' for idx in range(self.depth)]
        self.blocks = nn.ModuleList([
            NeuralOperator(self.modes[0], self.modes[1], self.latent_dims, idx+1, activation=activation) for idx, activation in enumerate(activations)
        ])

    def forward(self, xx, grid):
        # we get x in the shape
        B, H, W, C = xx.shape
        # save the predictions
        predictions = torch.zeros((B, H, W, self.steps_out), dtype=xx.dtype, device=xx.device)
        # iterate over the steps we have to forecast ahead
        for idx in range(self.steps_out):
            # add the new predictions to the x data
            if(idx > 0):
                xx = torch.cat((xx[..., 1:], predictions[..., idx-1:idx]), dim=-1)
            # add the grid to the data
            x = torch.cat((xx, grid), dim=-1)
            # project the data
            x = self.project(x)
            x = x.permute(0, 3, 1, 2)
            # go thorugh the blocks
            for block in self.blocks:
                x = block(x)
            # decode the prediction
            x = x.permute(0, 2, 3, 1)
            x = self.decode(x)
            x = torch.squeeze(x, axis=-1)
            # save the prediction
            predictions[..., idx] = x
        # return the predictions
        return predictions