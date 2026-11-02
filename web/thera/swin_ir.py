"""MLX implementation of SwinIR (Swin Transformer for Image Restoration)."""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def window_partition(x, window_size):
    """Partition into non-overlapping windows.
    x: (B, H, W, C) → (B * nH * nW, ws, ws, C)
    """
    B, H, W, C = x.shape
    nH, nW = H // window_size, W // window_size
    x = x.reshape(B, nH, window_size, nW, window_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # B, nH, nW, ws, ws, C
    return x.reshape(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """Reverse window_partition.
    windows: (B * nH * nW, ws, ws, C) → (B, H, W, C)
    """
    nH, nW = H // window_size, W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.reshape(B, nH, nW, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # B, nH, ws, nW, ws, C
    return x.reshape(B, H, W, -1)


def make_attn_mask(shift_size, window_size, H, W):
    """Create attention mask for shifted window self-attention."""
    if shift_size == 0:
        return None

    mask = np.zeros((1, H, W, 1))
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(mx.array(mask), window_size)  # nW, ws, ws, 1
    mask_windows = mask_windows.reshape(-1, window_size * window_size)  # nW, ws*ws
    attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]  # nW, ws*ws, ws*ws
    attn_mask = mx.where(attn_mask != 0, -100.0, 0.0)
    return attn_mask


def make_relative_position_index(window_size):
    """Compute relative position index for each pair in a window."""
    coords_h = np.arange(window_size)
    coords_w = np.arange(window_size)
    # Note: meshgrid with indexing='ij' matches the Flax implementation
    coords = np.stack(np.meshgrid(coords_w, coords_h, indexing='ij'))  # 2, ws, ws
    coords_flat = coords.reshape(2, -1)  # 2, ws*ws
    relative = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, ws*ws, ws*ws
    relative = relative.transpose(1, 2, 0)  # ws*ws, ws*ws, 2
    relative[:, :, 0] += window_size - 1
    relative[:, :, 1] += window_size - 1
    relative[:, :, 0] *= 2 * window_size - 1
    return relative.sum(-1)  # ws*ws, ws*ws


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table: (2*ws-1)^2 entries, num_heads
        num_rel = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = mx.zeros((num_rel, num_heads))

        # Precompute relative position index (as flat int array for indexing)
        idx = make_relative_position_index(window_size)
        self._rel_pos_index = mx.array(idx.reshape(-1), dtype=mx.int32)

    def __call__(self, x, mask=None):
        B, N, C = x.shape  # N = ws*ws

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 3, 2)  # B, heads, N, N

        # Add relative position bias
        rel_pos_bias = self.relative_position_bias_table[
            self._rel_pos_index
        ].reshape(N, N, -1)
        rel_pos_bias = rel_pos_bias.transpose(2, 0, 1)  # heads, N, N
        attn = attn + rel_pos_bias[None]

        # Apply mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B // nW, nW, self.num_heads, N, N)
            attn = attn + mask[None, :, None, :, :]
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional shifted windows."""

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def __call__(self, x, H, W):
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        ws = self.window_size
        shift = self.shift_size

        # Cyclic shift
        if shift > 0:
            x = mx.roll(x, (-shift, -shift), axis=(1, 2))

        # Partition windows
        x_windows = window_partition(x, ws)  # nW*B, ws, ws, C
        x_windows = x_windows.reshape(-1, ws * ws, C)

        # Window attention with mask
        mask = make_attn_mask(shift, ws, H, W) if shift > 0 else None
        attn_windows = self.attn(x_windows, mask)

        # Merge windows
        attn_windows = attn_windows.reshape(-1, ws, ws, C)
        x = window_reverse(attn_windows, ws, H, W)

        # Reverse cyclic shift
        if shift > 0:
            x = mx.roll(x, (shift, shift), axis=(1, 2))

        x = x.reshape(B, H * W, C)

        # Residual + FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block."""

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.0):
        super().__init__()
        self.blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(dim, num_heads, window_size, shift, mlp_ratio)
            )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def __call__(self, x, H, W):
        res = x
        for block in self.blocks:
            x = block(x, H, W)

        # PatchUnEmbed → Conv → PatchEmbed
        B, L, C = x.shape
        x = x.reshape(B, H, W, C)
        x = self.conv(x)
        x = x.reshape(B, H * W, C)
        return x + res


class SwinIRTail(nn.Module):
    """SwinIR tail module for Thera-pro.

    Takes 64-channel RDN output and refines it, outputting 64 channels.
    """

    def __init__(self, in_channels=64, embed_dim=180, depths=(7, 6),
                 num_heads=(6, 6), window_size=8, mlp_ratio=2.0, num_feat=64):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim

        # conv_first
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # PatchEmbed norm
        self.patch_embed_norm = nn.LayerNorm(embed_dim, eps=1e-5)

        # RSTB layers
        self.layers = [
            RSTB(embed_dim, depth, nh, window_size, mlp_ratio)
            for depth, nh in zip(depths, num_heads)
        ]

        # Final norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

        # conv_after_body
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # conv_before_upsample → output 64 channels
        self.conv_last = nn.Conv2d(embed_dim, num_feat, kernel_size=3, padding=1)

    def __call__(self, x):
        _, h_orig, w_orig, _ = x.shape

        # Pad to multiple of window_size
        ws = self.window_size
        pad_h = (ws - h_orig % ws) % ws
        pad_w = (ws - w_orig % ws) % ws
        if pad_h > 0 or pad_w > 0:
            # Reflect padding
            x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], mode='edge')

        _, H, W, _ = x.shape

        # conv_first
        x = self.conv_first(x)
        res = x

        # PatchEmbed (reshape + norm)
        B = x.shape[0]
        x = x.reshape(B, H * W, self.embed_dim)
        x = self.patch_embed_norm(x)

        # RSTB layers
        for layer in self.layers:
            x = layer(x, H, W)

        # Final norm
        x = self.norm(x)

        # PatchUnEmbed
        x = x.reshape(B, H, W, self.embed_dim)

        # conv_after_body + residual
        x = self.conv_after_body(x) + res

        # conv_before_upsample
        x = nn.leaky_relu(self.conv_last(x))

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :h_orig, :w_orig, :]

        return x
