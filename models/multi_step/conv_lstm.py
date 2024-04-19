import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.layers.base_layers import ConvLSTMCell

class ConvLSTMModel(nn.Module):
    def __init__(self, config:dict, image_size:tuple):
        # save needed vars
        super().__init__()
        self.img_size = image_size
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
            # if(idx == self.depth - 1):
            #     out_dims = self.out_dims
            blocks.append(ConvLSTMCell(in_dims, out_dims, kernel_size=self.kernel_size, img_shape=self.img_size))
        blocks.append(nn.Conv2d(self.latent_dims, self.out_dims, kernel_size=self.kernel_size, padding=self.kernel_size//2))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, grid, image_size):
        B, H, W, I = x.shape
        x = rearrange(x,
            "b h w (i c) -> b i c h w",
            b=B, h=H, w=W, i=I, c=1
        )
        # print("Init shape: ", x.shape)
        assert H == self.img_size[0] and W == self.img_size[1]
        # setup the block states
        for block in self.blocks[:-1]:
            block.setup_states(B, x.device)
        # warmup the lstm
        for step in range(self.input_steps-1):
            last_step = x[:, step, ...]
            for block in self.blocks:
                last_step = block(last_step)

        # run the conv lstm
        last_step = x[:, -1, ...]
        predictions = torch.zeros(B, self.out_dims, self.img_size[0], self.img_size[1], self.forecast_steps, device=x.device)
        for step in range(self.forecast_steps):
            for block in self.blocks:
                last_step = block(last_step)
            predictions[..., step] = last_step

        predictions = rearrange(predictions,
            "b c h w o -> b h w (c o)",
            b=B, h=H, w=W, c=1, o=self.forecast_steps
        )
        return predictions

if __name__ == '__main__':
    mod = ConvLSTMModel({
        "LATENT_DIMS": 8,
        "TIME_STEPS_IN": 10,
        "TIME_STEPS_OUT": 12,
        'DEPTH': 2,
        'KERNEL_SIZE': 5,
    }, (32,32))
    mod(torch.randn(2, 32, 32, 10), None)