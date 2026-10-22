import numbers
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from models.spatial_modulation import SpatialAdaLN, SpatialFiLM, _gate_spatial, _modulate_spatial, _resize_cond

def _to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.eps = eps

    def forward(self, x):
        var = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(var + self.eps)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = _BiasFreeLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, padding=1,
            groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) y x -> b head c (y x)", head=self.num_heads)
        k = rearrange(k, "b (head c) y x -> b head c (y x)", head=self.num_heads)
        v = rearrange(v, "b (head c) y x -> b head c (y x)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (y x) -> b (head c) y x", head=self.num_heads, y=h, x=w)
        return self.project_out(out)

class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.0, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1,
            groups=hidden * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class RestorationTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.0,
                 bias=False, cond_dim=128):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias=bias)
        self.mod = SpatialAdaLN(dim, cond_dim)

    def forward(self, x, cond_map):
        c = _resize_cond(cond_map, x.shape[-2:])
        g1, b1, a1, g2, b2, a2 = self.mod(c)
        x = x + _gate_spatial(self.attn(_modulate_spatial(self.norm1(x), g1, b1)), a1)
        x = x + _gate_spatial(self.ffn(_modulate_spatial(self.norm2(x), g2, b2)), a2)
        return x


class ConvResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode="reflect", bias=False)
        self.norm1 = nn.GroupNorm(8 if dim % 8 == 0 else 1, dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode="reflect", bias=False)
        self.norm2 = nn.GroupNorm(8 if dim % 8 == 0 else 1, dim)
        self.act = nn.GELU()
        self.mod1 = SpatialFiLM(dim, cond_dim)
        self.mod2 = SpatialFiLM(dim, cond_dim)

    def forward(self, x, cond_map):
        c = _resize_cond(cond_map, x.shape[-2:])
        g1, b1 = self.mod1(c)
        g2, b2 = self.mod2(c)
        h = self.conv1(x)
        h = _modulate_spatial(self.norm1(h), g1, b1)
        h = self.act(h)
        h = self.conv2(h)
        h = _modulate_spatial(self.norm2(h), g2, b2)
        return x + h
