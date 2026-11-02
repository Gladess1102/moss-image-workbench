"""MLX implementation of Thera super-resolution models (air/pro variants)."""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


# --- Utility functions ---

def make_grid(h, w):
    """Create coordinate grid in [-0.5, 0.5] with pixel centers."""
    offset_h = 1.0 / (2 * h)
    offset_w = 1.0 / (2 * w)
    ys = np.linspace(-0.5 + offset_h, 0.5 - offset_h, h, dtype=np.float32)
    xs = np.linspace(-0.5 + offset_w, 0.5 - offset_w, w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    return np.stack([grid_y, grid_x], axis=-1)  # (H, W, 2)


def interpolate_nearest(coords, grid):
    """
    Nearest-neighbor sampling of a grid at given coordinates.
    Args:
        coords: mx.array (B, H, W, 2) coordinates in [-0.5, 0.5]
        grid: mx.array (B, H', W', C) grid to sample from
    Returns:
        mx.array (B, H, W, C)
    """
    B, Hp, Wp, C = grid.shape
    _, H, W, _ = coords.shape

    y = coords[..., 0] * Hp + (Hp - 1) / 2.0
    x = coords[..., 1] * Wp + (Wp - 1) / 2.0

    y_idx = mx.clip(mx.round(y).astype(mx.int32), 0, Hp - 1)
    x_idx = mx.clip(mx.round(x).astype(mx.int32), 0, Wp - 1)

    flat_idx = y_idx * Wp + x_idx  # (B, H, W)
    batch_offset = mx.arange(B).reshape(B, 1, 1) * (Hp * Wp)
    global_idx = (flat_idx + batch_offset).reshape(-1)  # (B*H*W,)

    grid_flat = grid.reshape(-1, C)  # (B*Hp*Wp, C)
    result = grid_flat[global_idx]  # (B*H*W, C)
    return result.reshape(B, H, W, C)


# --- RDN Backbone ---

class RDBConv(nn.Module):
    """Single convolution layer within a Residual Dense Block."""
    def __init__(self, in_channels: int, growth_rate: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size,
                              padding=(kernel_size - 1) // 2)

    def __call__(self, x):
        out = nn.relu(self.conv(x))
        return mx.concatenate([x, out], axis=-1)


class RDB(nn.Module):
    """Residual Dense Block."""
    def __init__(self, g0: int, growth_rate: int, n_conv_layers: int):
        super().__init__()
        self.convs = [
            RDBConv(g0 + i * growth_rate, growth_rate)
            for i in range(n_conv_layers)
        ]
        total_ch = g0 + n_conv_layers * growth_rate
        self.local_fusion = nn.Conv2d(total_ch, g0, kernel_size=1)

    def __call__(self, x):
        res = x
        for conv in self.convs:
            x = conv(x)
        x = self.local_fusion(x)
        return x + res


class RDN(nn.Module):
    """Residual Dense Network backbone (config B)."""
    def __init__(self, n_colors: int = 3, g0: int = 64):
        super().__init__()
        D, C, G = 16, 8, 64  # config B

        self.sfe1 = nn.Conv2d(n_colors, g0, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(g0, g0, kernel_size=3, padding=1)
        self.rdbs = [RDB(g0, G, C) for _ in range(D)]
        self.gff_1x1 = nn.Conv2d(D * g0, g0, kernel_size=1)
        self.gff_3x3 = nn.Conv2d(g0, g0, kernel_size=3, padding=1)

    def __call__(self, x):
        f1 = self.sfe1(x)
        x = self.sfe2(f1)

        rdb_outs = []
        for rdb in self.rdbs:
            x = rdb(x)
            rdb_outs.append(x)

        x = mx.concatenate(rdb_outs, axis=-1)
        x = self.gff_1x1(x)
        x = self.gff_3x3(x)
        return x + f1


# --- Thera Model ---

class Thera(nn.Module):
    """
    Thera: arbitrary-scale super-resolution using neural heat fields.

    Stages:
    1. Encoder (RDN backbone) produces features at source resolution
    2. Optional refinement tail (identity for air, SwinIR for pro)
    3. Hypernetwork (1x1 conv) predicts per-pixel field parameters
    4. Heat field decoder produces RGB residuals
    """
    OUT_DIM = 3
    W0 = 1.0
    MEAN = np.array([0.4488, 0.4371, 0.4040], dtype=np.float32)
    VAR = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    def __init__(self, size='air'):
        super().__init__()
        self.size = size
        self.hidden_dim = 32 if size == 'air' else 512

        # Field params: Dense kernel + Thermal phase (alphabetical order)
        n_field_params = self.hidden_dim * self.OUT_DIM + self.hidden_dim

        self.encoder = RDN(n_colors=3, g0=64)

        # Refinement tail
        if size == 'pro':
            from swin_ir import SwinIRTail
            self.refine = SwinIRTail(
                in_channels=64, embed_dim=180,
                depths=(7, 6), num_heads=(6, 6),
                window_size=8, mlp_ratio=2.0, num_feat=64)
        # For 'air', no refine module (identity)

        self.out_conv = nn.Conv2d(64, n_field_params, kernel_size=1)

        self.k = mx.array(0.0)
        self.components = mx.zeros((2, self.hidden_dim))

    def encode(self, source_norm):
        """Run encoder + optional refinement tail."""
        x = self.encoder(source_norm)
        if self.size == 'pro':
            x = self.refine(x)
        return x

    def decode(self, encoding, target_coords, t):
        """Predict RGB residuals at target coordinates."""
        sampled = interpolate_nearest(target_coords, encoding)
        phi = self.out_conv(sampled)

        hd = self.hidden_dim
        kernel = phi[..., :hd * self.OUT_DIM].reshape(
            *phi.shape[:-1], hd, self.OUT_DIM)
        phase = phi[..., hd * self.OUT_DIM:]

        Hs, Ws = encoding.shape[1], encoding.shape[2]
        source_grid = mx.array(make_grid(Hs, Ws))
        source_coords = mx.broadcast_to(
            source_grid[None], (encoding.shape[0],) + source_grid.shape)
        nearest_src = interpolate_nearest(target_coords, source_coords)
        rel_coords = target_coords - nearest_src
        rel_coords_scaled = mx.concatenate([
            rel_coords[..., 0:1] * Hs,
            rel_coords[..., 1:2] * Ws,
        ], axis=-1)

        x = rel_coords_scaled @ self.components
        norm = mx.linalg.norm(self.components, axis=0)
        t_4d = t[:, :, None, None] if t.ndim == 2 else t.reshape(-1, 1, 1, 1)
        decay = mx.exp(-((self.W0 * norm) ** 2) * self.k * t_4d)
        x = mx.sin(self.W0 * x + phase) * decay
        out = mx.sum(x[..., None] * kernel, axis=-2)

        return out

    def upscale(self, source, target_h, target_w, ensemble=False):
        mean = mx.array(self.MEAN)
        var = mx.array(self.VAR)
        std = mx.sqrt(var)

        if ensemble:
            outs = []
            for k_rot in range(4):
                src = mx.array(np.rot90(np.array(source), k=k_rot))
                th = target_w if k_rot % 2 else target_h
                tw = target_h if k_rot % 2 else target_w
                out = self._upscale_single(src, th, tw, mean, var, std)
                mx.eval(out)
                out_np = np.rot90(np.array(out), k=-k_rot)
                outs.append(out_np)
            result = np.stack(outs).mean(0).clip(0.0, 1.0)
            return mx.array((result * 255).round().astype(np.uint8))
        else:
            out = self._upscale_single(source, target_h, target_w, mean, var, std)
            out = mx.clip(out, 0.0, 1.0)
            return (out * 255 + 0.5).astype(mx.uint8)

    def _upscale_single(self, source, target_h, target_w, mean, var, std):
        Hs, Ws = source.shape[0], source.shape[1]
        t = mx.array([(target_h / Hs) ** -2], dtype=mx.float32)[None]

        target_grid = mx.array(make_grid(target_h, target_w))[None]
        source_4d = source[None]
        source_up = interpolate_nearest(target_grid, source_4d)

        source_norm = (source_4d - mean) / std
        encoding = self.encode(source_norm)

        coords = mx.array(make_grid(target_h, target_w))[None]
        residual = self.decode(encoding, coords, t)

        out = residual * std + mean + source_up
        return out[0]


# Backwards compatibility alias
TheraRDNAir = Thera
