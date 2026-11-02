#!/usr/bin/env python3
"""Thera 任意尺度超分执行器 —— 独立进程（MLX，不碰 torch）。

为什么单独一个 runner：Thera 用 MLX + 神经热场，代码在 web/thera/（上游 thera-mlx 移植）。
主进程只做调度，绝不 import mlx 模型本身。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# thera/ 内部用绝对导入（from model import Thera），所以 thera/ 也要进 path
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "thera"))

# 必须限制 MLX 缓冲池：默认策略会把大量临时 buffer 留在 cache 里不归还，
# 4x 放大（640x480 -> 2560x1920）时峰值顶穿 Metal 显存，报
# "Command buffer execution failed: Insufficient Memory
#  (kIOGPUCommandBufferCallbackErrorOutOfMemory)"。
# mlx_runner.py 一直有这条限制，thera_runner 之前漏了 —— 2026-10-23 补上。
import mlx.core as mx  # noqa: E402

mx.set_cache_limit(1536 * 1024 * 1024)  # 1.5 GiB

from thera.upscale import upscale_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--scale", type=float, default=4.0)
    # Thera 默认整图一次放大，输出像素一多就顶穿 Metal 内存（2048² 源 x4 = 67MP 必爆）。
    # thera/upscale.py 本来就有 upscale_tiled（N×N 分块 + 羽化融合），这里透传出来。
    ap.add_argument("--tiles", type=int, default=1, help="N×N 分块，1=整图")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()

    weights = a.weights or os.path.join(HERE, "models", "weights-air.npz")
    if not os.path.isfile(weights):
        print("missing weights: " + weights, flush=True)
        return 2

    upscale_file(a.src, a.dst, scale=a.scale, model_size="air", weights_path=weights,
                 tiles=(a.tiles if a.tiles > 1 else None))
    print("完成 %s -> %s (%.2fx)" % (a.src, a.dst, a.scale), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
