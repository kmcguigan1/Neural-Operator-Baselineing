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
        print(self.img_shape)
        # input gate
        self.wxi = nn.Conv2d(self.in_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whi = nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wci = nn.Parameter(torch.zeros(1, self.hidden_dims, self.img_shape[0], self.img_shape[1]))
        # forget gate
        self.wxf = nn.Conv2d(self.in_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whf = nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wcf = nn.Parameter(torch.zeros(1, self.hidden_dims, self.img_shape[0], self.img_shape[1]))
        # context gate
        self.wxc = nn.Conv2d(self.in_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whc = nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        # self.wcc = nn.Parameter(torch.zeros(1, self.hidden_dims, self.img_shape[0], self.img_shape[1]))
        # output gate
        self.wxo = nn.Conv2d(self.in_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.who = nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
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
        input_gate = F.sigmoid(self.wxi(x) + self.whi(self.hidden_state) + self.cell_state * self.wci)

        forget_gate = F.sigmoid(self.wxf(x) + self.whf(self.hidden_state) + self.cell_state * self.wcf)

        new_cell_state = forget_gate * self.cell_state + input_gate * F.tanh(self.wxc(x) + self.whc(self.hidden_state))
    
        output_gate = F.sigmoid(self.wxo(x) + self.who(self.hidden_state) + new_cell_state * self.wco)
        
        new_hidden_state = output_gate * F.tanh(new_cell_state)

        self.update_states(new_hidden_state, new_cell_state)
        return new_hidden_state

class ConvLSTMModel(nn.Module):
    def __init__(self, config:dict, img_size:tuple):
        # save needed vars
        super().__init__()
        self.img_size = img_size
        print(self.img_size)
        self.latent_dims = config["LATENT_DIMS"]
        self.input_steps = config["TIME_STEPS_IN"]
        self.forecast_steps = config["TIME_STEPS_OUT"]
        self.in_dims = 1
        self.out_dims = 1
        self.depth = config['DEPTH']
        self.projection_kernel_size = config['PROJ_KERNEL_SIZE']
        self.kernel_size = config['KERNEL_SIZE']
        # generate the conv lstms
        self.projector = nn.Conv2d(self.in_dims, self.latent_dims, kernel_size=self.projection_kernel_size, padding="same")
        self.blocks = nn.ModuleList([
            ConvLSTMCell(self.latent_dims, self.latent_dims, self.kernel_size, self.img_size) for _ in range(self.depth)
        ])
        # generate the head model
        self.head = nn.Linear(self.latent_dims, self.out_dims)

    def forward(self, x, grid):
        B, H, W, I = x.shape
        x = rearrange(x,
            "b h w (i c) -> b i c h w",
            b=B, h=H, w=W, i=I, c=1
        )
        # print("Init shape: ", x.shape)
        assert H == self.img_size[0] and W == self.img_size[1]
        # get the latent states for all the inputs
        input_latent_states = torch.zeros(B, self.input_steps, self.latent_dims, self.img_size[0], self.img_size[1], device=x.device)
        for step in range(self.input_steps):
            input_latent_states[:, step, ...] = self.projector(x[:, step, ...])

        # warmup the lstm cells
        for step in range(self.input_steps):
            if(step > 0):
                latent_input = latent_input.clone()
            latent_input = input_latent_states[:, 0, ...]
            for block_idx, block in enumerate(self.blocks):
                if(step == 0):
                    block.setup_states(B, x.device)
                latent_input = block(latent_input)

        # run the forecasts on the lstm cells
        latent_states = torch.zeros(B, self.forecast_steps, self.latent_dims, self.img_size[0], self.img_size[1], device=x.device)        
        # print("latent states shape: ", latent_states.shape)
        # print("latent inputs shape: ", latent_input.shape)
        for step in range(self.forecast_steps):
            # print(f"step {step} input shape: {latent_input.shape}")
            for block_idx, block in enumerate(self.blocks):
                latent_input = block(latent_input)
            # save the latent
            latent_states[:, step, ...] = latent_input

        # project the latent variables into real space
        latent_states = rearrange(latent_states,
            "b f c h w -> b h w f c",
            b=B, f=self.forecast_steps, c=self.latent_dims, h=H, w=W    
        )
        x = self.head(latent_states)
        x = torch.squeeze(x, dim=-1)
        return x