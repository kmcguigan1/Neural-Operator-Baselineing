import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.layers.base_layers import FCNLSTMCell

class FCNLSTMModel(nn.Module):
    def __init__(self, config:dict):
        # save needed vars
        super().__init__()
        self.latent_dims = config["LATENT_DIMS"]
        self.input_steps = config["TIME_STEPS_IN"]
        self.forecast_steps = config["TIME_STEPS_OUT"]
        self.in_dims = 1
        self.out_dims = 1
        self.depth = config['DEPTH']
        self.kernel_size = config['KERNEL_SIZE']
        # generate the conv lstms
        blocks = []
        for idx in range(self.depth):
            in_dims, out_dims = self.latent_dims, self.latent_dims
            if(idx == 0):
                in_dims = self.in_dims
            blocks.append(FCNLSTMCell(in_dims, out_dims, kernel_size=self.kernel_size))
        blocks.append(nn.Conv2d(self.latent_dims, self.out_dims, kernel_size=self.kernel_size, padding=self.kernel_size//2))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, grid, image_size):
        B, H, W, I = x.shape
        x = rearrange(x,
            "b h w (i c) -> b i c h w",
            b=B, h=H, w=W, i=I, c=1
        )
        # print("Init shape: ", x.shape)
        assert H == image_size[0] and W == image_size[1]
        # setup the block states
        for block in self.blocks[:-1]:
            block.setup_states(B, image_size, x.device)
        # warmup the lstm
        for step in range(self.input_steps-1):
            last_step = x[:, step, ...]
            for block in self.blocks:
                last_step = block(last_step)

        # run the conv lstm
        last_step = x[:, -1, ...]
        predictions = torch.zeros(B, self.out_dims, image_size[0], image_size[1], self.forecast_steps, device=x.device)
        for step in range(self.forecast_steps):
            for block in self.blocks:
                last_step = block(last_step)
            predictions[..., step] = last_step

        predictions = rearrange(predictions,
            "b c h w o -> b h w (c o)",
            b=B, h=H, w=W, c=1, o=self.forecast_steps
        )
        return predictions