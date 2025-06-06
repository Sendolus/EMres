"""
This file is adapted from: https://github.com/Algolzw/image-restoration-sde.
Original license: MIT (Copyright © 2023 Ziwei Luo)
Modifications: Added an auxiliary context processing branch, from which features are injected into the main U-Net body
through cross-attention layers that are added in the U-Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

import functools

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None):
        super().__init__()
        self.context_dim = context_dim or dim
        self.to_q = nn.Conv2d(self.context_dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, context):
        # x: [B, C, H, W]
        # context: [B, C, H, W] or similar

        q = self.to_q(context)
        k = self.to_k(x)
        v = self.to_v(x)

        B, C, H, W = q.shape
        q = q.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        k = k.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        v = v.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        attn = torch.softmax(q @ k.transpose(-2, -1) / (C ** 0.5), dim=-1)
        out = attn @ v  # [B, HW, C]

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return self.to_out(out), attn


class ConditionalUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth=4, upscale=1):
        super().__init__()
        self.depth = depth
        self.upscale = upscale # not used

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2, nf, 7)
        self.context_proj = default_conv(9, nf, kernel_size=1)
        
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
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                CrossAttention(dim_in),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out),
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                CrossAttention(dim_out),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in),
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_cross_attn = CrossAttention(mid_dim)
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        xtt, c1, c2, c3 = torch.chunk(xt, 4, dim=1)
        context = torch.cat([c1, c2, c3], dim=1)

        x = xtt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        context = self.check_image_size(context, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        context = self.context_proj(context)

        t = self.time_mlp(time.view(1))

        h = []
        h2 = []

        for b1, b2, attn, crossattn, downsample, downsample_context in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)

            out, attn = crossattn(x, context)
            x = x + out
            h.append(x)
            h2.append(context)

            x = downsample(x)
            context = downsample_context(context)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        out, attn = self.mid_cross_attn(x, context)

        x = x + out
        h2.append(context)
        x = self.mid_block2(x, t)

        for b1, b2, attn, crossattn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            context = h2.pop()
            context = F.interpolate(context, size=x.shape[2:], mode='bilinear', align_corners=False)
            out, attn = crossattn(x, context)

            x = x + out
            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        
        return x



