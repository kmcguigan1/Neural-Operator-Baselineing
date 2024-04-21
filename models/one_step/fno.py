import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

from models.layers.base_layers import ConvMLP
from models.layers.fourier_blocks import FNOBlockWithW

class FNO2d(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # save needed vars
        self.latent_dims = config['LATENT_DIMS']
        self.in_dims = config['TIME_STEPS_IN'] + 2
        self.out_dims = 1
        self.modes = (config['MODES1'], config['MODES2'])
        self.mlp_ratio = config["MLP_RATIO"]
        self.depth = config['DEPTH']
        self.padding_mode = config.get('PADDING_MODE', None)
        # setup layers
        self.project = ConvMLP(self.in_dims, self.latent_dims, self.latent_dims // 2)
        self.decode = ConvMLP(self.latent_dims, self.out_dims, self.latent_dims // 2)
        # add the neural operator blocks
        self.blocks = nn.ModuleList()
        for idx in range(self.depth):
            self.blocks.append(FNOBlockWithW(self.latent_dims, self.modes[0], self.modes[1], self.mlp_ratio, padding_mode=self.padding_mode))
            if(idx < self.depth - 1):
                self.blocks.append(nn.GELU())

    def forward(self, x, grid):
        # add the grid to the data
        x = torch.cat((x, grid), dim=-1)
        B, H, W, C = x.shape
        x = rearrange(x, "b h w c -> b c h w", b=B, h=H, w=W, c=C)
        # project the data
        x = self.project(x)
        # pad the inputs if that is what we are doing
        if(self.padding_mode == 'ONCE_SINGLE'):
            padding = int(math.sqrt(x.size(-1)))
            x = F.pad(x, [0, padding, 0, padding])
        elif(self.padding_mode == 'ONCE_DUAL'):
            padding = int(math.sqrt(x.size(-1)) // 2)
            x = F.pad(x, [padding, padding, padding, padding])
        # go thorugh the blocks
        for block in self.blocks:
            x = block(x)
        # decode the prediction
        if(self.padding_mode == 'ONCE_SINGLE'):
            x = x[..., :-padding, :-padding]
        elif(self.padding_mode == 'ONCE_DUAL'):
            x = x[..., padding:-padding, padding:-padding]
        x = self.decode(x)
        x = rearrange(x, "b c h w -> b h w c", b=B, h=H, w=W, c=self.out_dims)
        return x