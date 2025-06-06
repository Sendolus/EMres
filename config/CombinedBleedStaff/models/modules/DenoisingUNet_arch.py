"""
This file is adapted from: https://github.com/Algolzw/image-restoration-sde.
Original license: MIT (Copyright © 2023 Ziwei Luo)
Modifications: Altered the conditioning mechanism and added classical single-head cross attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, LayerNorm, Residual)


class RVCross(nn.Module):
    '''
    Recto-Verso Cross-Attention Module
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm(c)
        self.norm_r = LayerNorm(c)
        self.l_projQ = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_projQ = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.l_projK = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_projK = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_projV = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_projV = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_l, x_r = x.chunk(2, dim=0)

        Q_l = self.l_projQ(self.norm_l(x_l)).permute(0, 2, 3, 1)    # B, H, W, c
        K_r_T = self.r_projK(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W
        V_r = self.r_projV(x_r).permute(0, 2, 3, 1)                 # B, H, W, c

        Q_r = self.r_projQ(self.norm_r(x_r)).permute(0, 2, 3, 1)    # B, H, W, c
        K_l_T = self.l_projK(self.norm_l(x_l)).permute(0, 2, 1, 3)  # B, H, c, W
        V_l = self.l_projV(x_l).permute(0, 2, 3, 1)                 # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention_r2l = torch.matmul(Q_l, K_r_T) * self.scale
        attention_l2r = torch.matmul(Q_r, K_l_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention_r2l, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention_l2r, dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        xl = x_l + F_r2l
        xr = x_r + F_l2r
        return torch.cat([xl, xr], dim=0)


class ConditionalUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth=4, upscale=1, fusion=False):
        super().__init__()
        self.depth = depth
        self.upscale = upscale # not used
        self.fusion = fusion

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2+1, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))), RVCross(dim_in),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))), RVCross(dim_out),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_fusion = RVCross(mid_dim)
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def pix_unshuffle(self, tensor):
        return F.pixel_unshuffle(tensor, downscale_factor=self.upscale)

    def pix_shuffle(self, tensor):
        return F.pixel_shuffle(tensor, upscale_factor=self.upscale)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def tuple_fn(self, fn, feats):
        return tuple([fn(x) for x in feats])

    def forward(self, xt, cond, time):

        xt_res = xt[:, :6, :, :].clone()

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        xtl = xt[:, :3, :, :]
        xtr = xt[:, 3:6, :, :]
        maskl = xt[:, 6:7, :, :]
        maskr = xt[:, 7:8, :, :]

        condl, condr = torch.chunk(cond, 2, dim=1)

        x_exp_l = xtl - condl
        x_exp_r = xtr - condr

        xl = torch.cat([x_exp_l, condl, maskl], dim=1)
        xr = torch.cat([x_exp_r, condr, maskr], dim=1)

        x = torch.cat([xl, xr], dim=0)
        time = torch.cat([time, time], dim=0)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time)

        h = []

        for b1, b2, attn, fusion, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            x = fusion(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_fusion(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, fusion, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)
            x = fusion(x)
            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W]
        x_l, x_r = x.chunk(2, dim=0)
        x = xt_res + torch.cat([x_l, x_r], dim=1)

        return x



