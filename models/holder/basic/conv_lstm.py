import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
class ConvLSTMCell(nn.Module):
    def __init__(self, in_dims:int, hidden_dims:int, kernel_size:int, img_shape:tuple):
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.img_shape = img_shape
        # input gate
        self.wxi = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whi = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wci = nn.Parameter(torch.zeros(1, self.hidden_dims, self.img_shape[0], self.img_shape[1]))
        # forget gate
        self.wxf = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whf = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wcf = nn.Parameter(torch.zeros(1, self.hidden_dims, self.img_shape[0], self.img_shape[1]))
        # context gate
        self.wxc = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whc = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        # self.wcc = nn.Parameter(torch.zeros(1, self.hidden_dims, self.img_shape[0], self.img_shape[1]))
        # output gate
        self.wxo = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.who = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wco = nn.Parameter(torch.zeros(1, self.hidden_dims, self.img_shape[0], self.img_shape[1]))

    def setup_states(self, batch_size, device):
        self.hidden_state = torch.zeros(batch_size, self.hidden_dims, self.img_shape[0], self.img_shape[1], device=device)
        self.cell_state = torch.zeros(batch_size, self.hidden_dims, self.img_shape[0], self.img_shape[1], device=device)
        
    def update_states(self, hidden_state, cell_state):
        self.hidden_state = self.hidden_state.clone()
        self.cell_state = self.cell_state.clone()
        self.hidden_state = hidden_state
        self.cell_state = cell_state

    
    def forward(self, x):
        # print("in gate")
        input_gate = F.sigmoid(self.wxi(x) + self.whi(self.hidden_state) + self.cell_state * self.wci)
        # print("f gate")

        forget_gate = F.sigmoid(self.wxf(x) + self.whf(self.hidden_state) + self.cell_state * self.wcf)
        # print("ncs gate")

        new_cell_state = forget_gate * self.cell_state + input_gate * F.tanh(self.wxc(x) + self.whc(self.hidden_state))
        # print("o gate")
    
        output_gate = F.sigmoid(self.wxo(x) + self.who(self.hidden_state) + new_cell_state * self.wco)
        # print("nhs gate")
        
        new_hidden_state = output_gate * F.tanh(new_cell_state)

        self.update_states(new_hidden_state, new_cell_state)
        return new_hidden_state

class ConvLSTMModel(nn.Module):
    def __init__(self, config:dict, img_size:tuple):
        # save needed vars
        super().__init__()
        self.img_size = img_size
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
        blocks.append(nn.Conv2d(self.latent_dims, self.out_dims, kernel_size=self.kernel_size, padding='same'))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, grid):
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