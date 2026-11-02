#!/usr/bin/env python3
"""Core upscale logic for Thera MLX."""

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from model import Thera

WEIGHTS_DIR = Path(__file__).parent / "weights"


def load_weights(model, weights_path):
    """Load converted weights into the MLX model."""
    weights_path = str(weights_path)
    if weights_path.endswith('.safetensors'):
        from safetensors.numpy import load_file
        raw = load_file(weights_path)
        weights = {k: mx.array(v) for k, v in raw.items()}
    elif weights_path.endswith('.npz'):
        raw = np.load(weights_path)
        weights = {k: mx.array(raw[k]) for k, v in raw.items()}
    else:
        raise ValueError(f"Unknown weight format: {weights_path}")

    weight_list = list(weights.items())
    model.load_weights(weight_list)
    return model


def get_weights_path(model_size):
    """Resolve weights path for a model variant."""
    return WEIGHTS_DIR / f"weights-{model_size}.safetensors"


def upscale_tiled(model, source_np, target_h, target_w, tiles, ensemble=False):
    """Upscale an image using NxN tiles to reduce peak RAM.

    Splits the source image into tiles with overlap, upscales each tile
    individually, then blends them back together using linear feathering.

    Args:
        model: Loaded Thera model.
        source_np: numpy array (H, W, 3) float32 in [0, 1].
        target_h: Target height.
        target_w: Target width.
        tiles: Number of tiles per axis (2, 3, or 4).
        ensemble: Use geometric self-ensemble.

    Returns:
        numpy uint8 array (target_h, target_w, 3).
    """
    h, w = source_np.shape[:2]
    scale_h = target_h / h
    scale_w = target_w / w

    # Overlap in source pixels (10% of tile size, minimum 8px)
    tile_h = h / tiles
    tile_w = w / tiles
    overlap_h = max(8, int(tile_h * 0.1))
    overlap_w = max(8, int(tile_w * 0.1))

    # Build output canvas (float32 for blending)
    output = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_map = np.zeros((target_h, target_w, 1), dtype=np.float32)

    total_tiles = tiles * tiles
    done = 0

    for row in range(tiles):
        for col in range(tiles):
            # Source tile bounds with overlap
            sy0 = round(row * h / tiles) - (overlap_h if row > 0 else 0)
            sy1 = round((row + 1) * h / tiles) + (overlap_h if row < tiles - 1 else 0)
            sx0 = round(col * w / tiles) - (overlap_w if col > 0 else 0)
            sx1 = round((col + 1) * w / tiles) + (overlap_w if col < tiles - 1 else 0)

            sy0 = max(0, sy0)
            sy1 = min(h, sy1)
            sx0 = max(0, sx0)
            sx1 = min(w, sx1)

            tile_src = source_np[sy0:sy1, sx0:sx1]
            th = round((sy1 - sy0) * scale_h)
            tw = round((sx1 - sx0) * scale_w)

            # Upscale tile
            result = model.upscale(mx.array(tile_src), th, tw, ensemble=ensemble)
            mx.eval(result)
            tile_out = np.array(result).astype(np.float32) / 255.0

            # Target tile bounds
            ty0 = round(sy0 * scale_h)
            tx0 = round(sx0 * scale_w)
            ty1 = ty0 + tile_out.shape[0]
            tx1 = tx0 + tile_out.shape[1]

            # Clamp to output bounds
            ty1 = min(ty1, target_h)
            tx1 = min(tx1, target_w)
            tile_out = tile_out[:ty1 - ty0, :tx1 - tx0]

            # Linear feather weight for blending overlaps
            fh, fw = tile_out.shape[:2]
            wy = np.ones(fh, dtype=np.float32)
            wx = np.ones(fw, dtype=np.float32)

            # Feather top/bottom edges in overlap regions
            ovl_top = round(overlap_h * scale_h) if row > 0 else 0
            ovl_bot = round(overlap_h * scale_h) if row < tiles - 1 else 0
            ovl_left = round(overlap_w * scale_w) if col > 0 else 0
            ovl_right = round(overlap_w * scale_w) if col < tiles - 1 else 0

            if ovl_top > 0:
                ramp = np.linspace(0, 1, min(ovl_top, fh), dtype=np.float32)
                wy[:len(ramp)] = ramp
            if ovl_bot > 0:
                ramp = np.linspace(1, 0, min(ovl_bot, fh), dtype=np.float32)
                wy[-len(ramp):] = np.minimum(wy[-len(ramp):], ramp)
            if ovl_left > 0:
                ramp = np.linspace(0, 1, min(ovl_left, fw), dtype=np.float32)
                wx[:len(ramp)] = ramp
            if ovl_right > 0:
                ramp = np.linspace(1, 0, min(ovl_right, fw), dtype=np.float32)
                wx[-len(ramp):] = np.minimum(wx[-len(ramp):], ramp)

            w2d = wy[:, None] * wx[None, :]  # (fh, fw)
            w3d = w2d[:, :, None]  # (fh, fw, 1)

            output[ty0:ty1, tx0:tx1] += tile_out * w3d
            weight_map[ty0:ty1, tx0:tx1] += w3d

            done += 1
            print(f"  tile {done}/{total_tiles}")

    # Normalize by weight
    weight_map = np.maximum(weight_map, 1e-8)
    output = (output / weight_map * 255 + 0.5).clip(0, 255).astype(np.uint8)
    return output


def upscale_file(input_path, output_path, scale=None, size=None,
                 model_size='air', weights_path=None, ensemble=False,
                 tiles=None):
    """Upscale a single image file."""
    img = Image.open(input_path).convert('RGB')
    source = np.asarray(img, dtype=np.float32) / 255.0
    h, w = source.shape[:2]

    if scale is not None:
        target_h = round(h * scale)
        target_w = round(w * scale)
    elif size is not None:
        target_h, target_w = size
    else:
        raise ValueError("Must specify either scale or size")

    scale_actual = target_h / h
    if weights_path is None:
        weights_path = get_weights_path(model_size)

    model = Thera(size=model_size)
    model = load_weights(model, weights_path)
    mx.eval(model.parameters())

    t0 = time.perf_counter()

    if tiles and tiles > 1:
        print(f"Tiled upscale: {tiles}x{tiles} ({tiles*tiles} tiles)")
        result_np = upscale_tiled(model, source, target_h, target_w,
                                  tiles, ensemble=ensemble)
        Image.fromarray(result_np).save(output_path)
    else:
        result = model.upscale(mx.array(source), target_h, target_w, ensemble=ensemble)
        mx.eval(result)
        Image.fromarray(np.array(result)).save(output_path)

    elapsed = time.perf_counter() - t0

    suffix = " (ensemble)" if ensemble else ""
    tile_info = f" [{tiles}x{tiles} tiles]" if tiles and tiles > 1 else ""
    print(f"[{model_size}]{suffix}{tile_info} {w}x{h} -> {target_w}x{target_h} ({scale_actual:.4g}x)  {elapsed:.1f}s  ->  {output_path}")
