import torch
import torch.nn as nn
import torch.nn.functional as F

class CondEncoder(nn.Module):
    def __init__(self, cond_channels, cond_dim):
        super().__init__()
        h = max(cond_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, h, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if h % 8 == 0 else 1, h),
            nn.GELU(),
            nn.Conv2d(h, cond_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if cond_dim % 8 == 0 else 1, cond_dim),
            nn.GELU(),
        )

    def forward(self, z):
        return self.net(z)


def _resize_cond(cond_map, target_hw):
    if cond_map.shape[-2:] == tuple(target_hw):
        return cond_map
    return F.interpolate(
        cond_map, size=target_hw, mode="bilinear", align_corners=False,
    )


class SpatialFiLM(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.gamma = nn.Conv2d(cond_dim, dim, kernel_size=3, padding=1, bias=True)
        self.beta = nn.Conv2d(cond_dim, dim, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, c):
        # c: [B, cond_dim, H, W] -- already resized to target spatial size
        return self.gamma(c), self.beta(c)


class SpatialAdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.conv = nn.Conv2d(cond_dim, 6 * dim, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, c):
        return self.conv(c).chunk(6, dim=1)


def _modulate_spatial(x, gamma, beta):
    # All inputs are [B, C, H, W] at the same spatial size.
    return x * (1.0 + gamma) + beta


def _gate_spatial(x, alpha):
    return x * alpha