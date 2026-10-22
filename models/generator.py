import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from models.restoration_transformer import RestorationTransformerBlock, ConvResBlock
from models.spatial_modulation import CondEncoder


class PatchEmbed4x(nn.Module):
    def __init__(self, in_c=3, dim=48):
        super().__init__()
        mid = max(dim // 2, 16)
        self.conv1 = nn.Conv2d(in_c, mid, kernel_size=3, stride=2, padding=1,
                               padding_mode="reflect", bias=False)
        self.conv2 = nn.Conv2d(mid, dim, kernel_size=3, stride=2, padding=1,
                               padding_mode="reflect", bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class _Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class _Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class FinalUpsample4x(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(dim // 2, (dim // 2) * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.up2(self.up1(x))

def _build_gaussian_kernel(kernel_size=3, pad=2, sigma=1.0, channels=3):
    coord = torch.arange(kernel_size).float()
    grid_x = coord.repeat(kernel_size).view(kernel_size, kernel_size)
    grid_y = grid_x.t()
    xy = torch.stack([grid_x, grid_y], dim=-1)
    mean = (kernel_size - 1) / 2.0
    var = sigma ** 2
    k = (1.0 / (2.0 * math.pi * var)) * torch.exp(
        -((xy - mean) ** 2).sum(-1) / (2.0 * var)
    )
    k = k / k.sum()
    k = k.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    conv = nn.Conv2d(channels, channels, kernel_size,
                     groups=channels, padding=kernel_size - pad, bias=False)
    conv.weight.data.copy_(k)
    conv.weight.requires_grad = False
    return conv


class RTCGenerator(nn.Module):

    def __init__(
        self,
        cond_channels=4,
        cond_dim=128,
        dim=64,
        n_latent=8,
        n_heads_latent=8,
        ffn_expansion_factor=2.0,
        bias=False,
        inception=False, 
        sigma=1.0,
    ):
        super().__init__()
        self.inception = inception
        self.cond_channels = cond_channels
        self.cond_dim = cond_dim

        # ---- Conditioning encoder: 2D -> 2D, lifts to cond_dim channels ----
        self.cond_encoder = CondEncoder(cond_channels, cond_dim)

        # ---- Patch embed (4x stride; immediately gets to H/4, W/4) ----
        self.patch_embed = PatchEmbed4x(in_c=3, dim=dim)

        # ---- Encoder L1: cheap conv blocks at dim, H/4 ----
        self.enc1_a = ConvResBlock(dim, cond_dim)
        self.enc1_b = ConvResBlock(dim, cond_dim)
        self.down1 = _Downsample(dim)              # dim -> 2*dim, H/8

        # ---- Encoder L2: cheap conv blocks at 2*dim, H/8 ----
        self.enc2_a = ConvResBlock(dim * 2, cond_dim)
        self.enc2_b = ConvResBlock(dim * 2, cond_dim)
        self.down2 = _Downsample(dim * 2)           # 2*dim -> 4*dim, H/16

        # ---- Latent: heavy transformer work at 4*dim, H/16 ----
        self.latent = nn.ModuleList([
            RestorationTransformerBlock(
                dim * 4, num_heads=n_heads_latent,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, cond_dim=cond_dim,
            )
            for _ in range(n_latent)
        ])

        # ---- Decoder L2 ----
        self.up2 = _Upsample(dim * 4)               # 4*dim -> 2*dim, H/8
        self.fuse2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.dec2_a = ConvResBlock(dim * 2, cond_dim)
        self.dec2_b = ConvResBlock(dim * 2, cond_dim)

        # ---- Decoder L1 ----
        self.up1 = _Upsample(dim * 2)               # 2*dim -> dim, H/4
        self.fuse1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.dec1_a = ConvResBlock(dim, cond_dim)
        self.dec1_b = ConvResBlock(dim, cond_dim)

        # ---- 4x upsample back to input resolution ----
        self.final_up = FinalUpsample4x(dim)        # H/4 -> H
        self.final_norm = nn.GroupNorm(
            num_groups=8 if (dim // 2) % 8 == 0 else 1,
            num_channels=dim // 2,
        )
        self.output = nn.Conv2d(dim // 2, 3, kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

        # Smoothing kernel
        self.alf_layer = _build_gaussian_kernel(kernel_size=3, pad=2, sigma=sigma)
        # self.block_mask = BlockMask(mask_prob=1.0)

        # ----- Padding for U-Net divisibility ------------------------------
    @staticmethod
    def _pad_to_multiple(x, factor=16):
        """Pad bottom/right to be divisible by factor (4x patch_embed + 2
        strided downsamples = factor 16). e.g. 299 -> 304."""
        h, w = x.shape[-2:]
        new_h = ((h + factor - 1) // factor) * factor
        new_w = ((w + factor - 1) // factor) * factor
        pad_h = new_h - h
        pad_w = new_w - w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w)

  
    def forward(self, x, cond, eps=0):
        # 2D conditioning ONLY. cond must be 4D [B, cond_channels, H_c, W_c].
        assert cond.dim() == 4, (
            f"CondRestormerSpatial expects a 4D spatial cond tensor "
            f"[B, {self.cond_channels}, H_c, W_c], got shape {tuple(cond.shape)}."
        )
        cond_map = self.cond_encoder(cond.float())   # [B, cond_dim, H_c, W_c]

        x_in, (H, W) = self._pad_to_multiple(x, factor=16)

        # Encoder
        x = self.patch_embed(x_in)                          # [B, dim, H/4, W/4]
        x = self.enc1_a(x, cond_map)
        x = self.enc1_b(x, cond_map)
        skip1 = x
        x = self.down1(x)                                   # [B, 2*dim, H/8]
        x = self.enc2_a(x, cond_map)
        x = self.enc2_b(x, cond_map)
        skip2 = x
        x = self.down2(x)                                   # [B, 4*dim, H/16]

        # Latent (heavy transformers at small spatial size)
        for blk in self.latent:
            x = blk(x, cond_map)

        # Decoder
        x = self.up2(x)                                     # [B, 2*dim, H/8]
        x = self.fuse2(torch.cat([x, skip2], dim=1))
        x = self.dec2_a(x, cond_map)
        x = self.dec2_b(x, cond_map)

        x = self.up1(x)                                     # [B, dim, H/4]
        x = self.fuse1(torch.cat([x, skip1], dim=1))
        x = self.dec1_a(x, cond_map)
        x = self.dec1_b(x, cond_map)

        # Restore full resolution
        x = self.final_up(x)                                # [B, dim/2, H, W]
        x = self.final_norm(x)
        out = self.output(x)
        out = out[..., :H, :W]                              # crop padding
        out = torch.tanh(out)
        out = self.alf_layer(out)
        return out * eps

