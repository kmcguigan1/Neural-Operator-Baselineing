import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dropout, norm_layer=nn.LayerNorm):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.norm = norm_layer(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        # x = self.norm(x)
        # x = self.drop(x)
        x = self.mlp2(x)
        return x

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
        return x

class VIT(nn.Module):
    def __init__(self, config:dict, img_size:tuple):
        super().__init__()
        # save needed vars
        self.img_size = img_size
        self.latent_dims = config["LATENT_DIMS"]
        self.forecast_steps = config["TIME_STEPS_OUT"]
        self.patch_size = [config['PATCH_SIZE'], config['PATCH_SIZE']]
        self.in_dims = config['TIME_STEPS_IN'] + 2
        self.out_dims = 1
        # self.mlp_ratio = config['MLP_RATIO']
        self.depth = config['DEPTH']
        self.nhead = config["NHEAD"]
        # dropout information
        self.drop_rate = config["DROPOUT"]
        # patch embeddings
        self.patch_embeddor = PatchEmbed(self.img_size, self.patch_size, self.in_dims, self.latent_dims, self.drop_rate)
        self.patch_count_x = self.patch_embeddor.patch_count_x
        self.patch_count_y = self.patch_embeddor.patch_count_y
        self.patch_count = self.patch_embeddor.patch_count
        # get the norm layer
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            self.latent_dims, self.nhead, self.latent_dims, self.drop_rate, activation="gelu", batch_first=True, layer_norm_eps=1e-6
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, self.depth
        )
        # prediction layer
        self.head = nn.Linear(self.latent_dims, self.out_dims*self.patch_size[0]*self.patch_size[1], bias=False)

    def forward(self, x, grid):
        # print(x.shape)
        x = torch.concatenate((x, grid), dim=-1)
        B, H, W, C = x.shape
        x = rearrange(
            x,
            "b h w c -> b c h w",
            b=B, h=H, w=W, c=C
        )
        x = self.patch_embeddor(x)
        # get the latent states
        latent_states = torch.zeros((B, self.forecast_steps, self.patch_count, self.latent_dims), device=x.device)
        for idx in range(self.forecast_steps):
            # get the next step
            x = self.transformer_encoder(x)
            # save the value
            latent_states[:, idx, :, :] = x
        # get the outputs
        x = self.head(latent_states)
        x = F.sigmoid(x)
        x = rearrange(
            x,
            "b f (h w) (p1 p2 c_out) -> b (h p1) (w p2) f c_out",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.patch_count_x,
            w=self.patch_count_y,
            f=self.forecast_steps,
            c_out=1
        )
        x = torch.squeeze(x, axis=-1)
        return x