import torch 
import torch.nn as nn 
import torch.nn.functional as F

"""MLP LAYERS"""
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
class ConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

"""EMBEDDOR LAYERS"""
class PatchEmbed(nn.Module):
    def __init__(self, img_size:tuple, patch_size:tuple, in_chans:int, embed_dim:int, dropout_rate:float=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # get the patch count information
        self.patch_count_x = img_size[0] // patch_size[0] 
        self.patch_count_y = img_size[1] // patch_size[1]
        assert img_size[0] % patch_size[0] == 0, f"x has bad dim combo im: {img_size[0]}, patch: {patch_size[0]}"
        assert img_size[1] % patch_size[1] == 0, f"y has bad dim combo im: {img_size[1]}, patch: {patch_size[1]}"
        self.patch_count = self.patch_count_x * self.patch_count_y
        # get the projection layer
        self.proj = nn.Conv2d(in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        # get the position embeddings information
        self.pos_embeddings = nn.Parameter(torch.zeros(1, self.patch_count, self.embed_dim))
        self.pos_dropout = nn.Dropout(dropout_rate)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # get the patches from the image
        x = self.proj(x)
        x = x.reshape(B, self.embed_dim, self.patch_count)
        x = x.transpose(1, 2)
        # add the position encoddings
        x = x + self.pos_embeddings
        x = self.pos_dropout(x)
        # reshape the data
        # x = x.reshape(B, self.patch_count_x, self.patch_count_y, self.embed_dim)
        return x
    
"""CONV LSTM CELLS"""
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
        input_gate = F.sigmoid(self.wxi(x) + self.whi(self.hidden_state) + self.cell_state * self.wci)
        forget_gate = F.sigmoid(self.wxf(x) + self.whf(self.hidden_state) + self.cell_state * self.wcf)
        new_cell_state = forget_gate * self.cell_state + input_gate * F.tanh(self.wxc(x) + self.whc(self.hidden_state))
        output_gate = F.sigmoid(self.wxo(x) + self.who(self.hidden_state) + new_cell_state * self.wco)        
        new_hidden_state = output_gate * F.tanh(new_cell_state)
        self.update_states(new_hidden_state, new_cell_state)
        return new_hidden_state
    
class FCNLSTMCell(nn.Module):
    def __init__(self, in_dims:int, hidden_dims:int, kernel_size:int):
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        # input gate
        self.wxi = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whi = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wci = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        # forget gate
        self.wxf = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whf = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wcf = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        # context gate
        self.wxc = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.whc = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        # output gate
        self.wxo = nn.Conv2d(in_channels=self.in_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=True)
        self.who = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)
        self.wco = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, kernel_size=self.kernel_size, padding="same", bias=False)

    def setup_states(self, batch_size, image_size, device):
        self.hidden_state = torch.zeros(batch_size, self.hidden_dims, image_size[0], image_size[1], device=device)
        self.cell_state = torch.zeros(batch_size, self.hidden_dims, image_size[0], image_size[1], device=device)
        
    def update_states(self, hidden_state, cell_state):
        self.hidden_state = self.hidden_state.clone()
        self.cell_state = self.cell_state.clone()
        self.hidden_state = hidden_state
        self.cell_state = cell_state
    
    def forward(self, x):
        input_gate = F.sigmoid(self.wxi(x) + self.whi(self.hidden_state) + self.wci(self.cell_state))
        forget_gate = F.sigmoid(self.wxf(x) + self.whf(self.hidden_state) + self.wcf(self.cell_state))
        new_cell_state = forget_gate * self.cell_state + input_gate * F.tanh(self.wxc(x) + self.whc(self.hidden_state))
        output_gate = F.sigmoid(self.wxo(x) + self.who(self.hidden_state) + self.wco(new_cell_state))        
        new_hidden_state = output_gate * F.tanh(new_cell_state)
        self.update_states(new_hidden_state, new_cell_state)
        return new_hidden_state