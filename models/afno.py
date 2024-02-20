import numpy as np 

import math
from functools import partial

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        x = x.reshape(B, self.patch_count_x, self.patch_count_y, self.embed_dim)
        return x

class SpectralConvBlockDiag(nn.Module):
    # we should experiment with more hard thresholding to make the noisier components zero
    def __init__(self, hidden_size, num_blocks=4, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        assert self.hidden_size % self.num_blocks == 0, f"hidden_size {self.hidden_size} should be divisble by num_blocks {self.num_blocks}"

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        # this sets things near zero to just be zero
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dropout=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.fc1 = nn.Linear(self.in_channels, self.mid_channels)
        self.act = act_layer()
        self.norm_layer = norm_layer(self.mid_channels)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.mid_channels, self.out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.norm_layer(x)
        # x = self.drop(x)
        x = self.fc2(x)
        return x

class AFNOBoundariesBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mlp_ratio=4,
        channel_mixer_dropout=0.0,
        num_blocks=1,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        channel_mixer_norm=nn.Identity,
        norm_layer=nn.Identity,
        output_activation=nn.Identity
    ):
        super().__init__()
        # initialize things for the fno section
        self.fno_norm1 = channel_mixer_norm(in_channels)
        self.fno_norm2 = channel_mixer_norm(in_channels)
        self.fno = SpectralConvBlockDiag(
            in_channels,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction
        )
        self.channel_mixer = MLP(
            in_channels=in_channels, 
            out_channels=in_channels, 
            mid_channels=in_channels*mlp_ratio,
            dropout=channel_mixer_dropout
        )
        # initialize the boundary conditions path
        # self.boundary_transform = MLP(
        #     in_channels=in_channels, 
        #     out_channels=in_channels, 
        #     mid_channels=in_channels*mlp_ratio,
        #     dropout=channel_mixer_dropout
        # )
        self.output_activation = output_activation()

    def forward(self, x):
        # get the layer inputs
        residual = x
        # apply the fno block
        # x = self.fno_norm1(x)
        x = self.fno(x)
        # x = self.fno_norm2(x)
        x = self.channel_mixer(x)
        # apply the MLP block for boundary conditions
        # xb = self.boundary_transform(residual)
        # add them together and transform
        # x = x + xb
        # apply the output activation
        return self.output_activation(x)

class AFNO(nn.Module):
    def __init__(self, config:dict, img_size:tuple):
        super().__init__()
        # save needed vars
        self.img_size = img_size
        self.latent_dims = config["LATENT_DIMS"]
        self.forecast_steps = config["TIME_STEPS_OUT"]
        self.patch_size = [config['PATCH_SIZE'], config['PATCH_SIZE']]
        self.in_dims = config['TIME_STEPS_IN'] + 2
        self.out_dims = 1
        self.num_blocks = config["NUM_BLOCKS"] 
        self.mlp_ratio = config['MLP_RATIO']
        self.depth = config['DEPTH']
        # dropout information
        self.drop_rate = config["DROP_RATE"]
        # self.drop_path_rate = config["DROP_PATH_RATE"]
        self.sparsity_threshold = 0.01
        self.hard_thresholding_fraction = 1.0
        # patch embeddings
        self.patch_embeddor = PatchEmbed(self.img_size, self.patch_size, self.in_dims, self.latent_dims, self.drop_rate)
        self.patch_count_x = self.patch_embeddor.patch_count_x
        self.patch_count_y = self.patch_embeddor.patch_count_y
        # get the norm layer
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # FNO blocks
        self.blocks = nn.ModuleList([
            AFNOBoundariesBlock(
                self.latent_dims,
                mlp_ratio=self.mlp_ratio,
                channel_mixer_dropout=self.drop_rate,
                # dropout=self.drop_path_rate,
                num_blocks=self.num_blocks,
                sparsity_threshold=self.sparsity_threshold,
                hard_thresholding_fraction=self.hard_thresholding_fraction,
                channel_mixer_norm=norm_layer,
                norm_layer=norm_layer
            ) for idx in range(config['DEPTH'])
        ])
        # prediction layer
        self.head = MLP(
            in_channels=self.latent_dims, 
            out_channels=self.out_dims*self.patch_size[0]*self.patch_size[1], 
            mid_channels=self.latent_dims*2,
            dropout=self.drop_rate
        )
        #nn.Linear(self.latent_dims, self.out_dims*self.patch_size[0]*self.patch_size[1], bias=False)
        # get the output activation function
        # get the sample input shape
        self._sample_input_shape = (self.in_dims, *self.img_size)

    def get_sample_input_shape(self):
        return self._sample_input_shape

# we have the dims wrong,
# conv needs to be C, H, W
# but linear needs H, W, C 
    def forward(self, x, grid):
        B, H, W, C = x.shape
        x = torch.cat((x, grid), dim=-1)
        x = rearrange(x,
            "b h w c -> b c h w",
            b=B, h=H, w=W, c=self.in_dims
        )
        x = self.patch_embeddor(x)
        # new shape is B, p1, p2, C
        # get the latent states
        latent_states = torch.zeros((B, self.forecast_steps, self.patch_count_x, self.patch_count_y, self.latent_dims), device=x.device)
        for idx in range(self.forecast_steps):
            # get the next step
            for block in self.blocks:
                x = block(x)
            # save the value
            latent_states[:, idx, :, :, :] = x
        # get the outputs
        x = self.head(latent_states)
        x = rearrange(
            x,
            "b f h w (p1 p2 c_out) -> b c_out (h p1) (w p2) f",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.patch_count_x,
            w=self.patch_count_y,
            f=self.forecast_steps
        )
        x = torch.squeeze(x, axis=1)
        return x