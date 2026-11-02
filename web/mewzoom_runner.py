#!/usr/bin/env python3
"""MewZoom 超分执行器 —— 独立进程（torch 绝不进工作台主进程）。

工作台主进程 import torch 会在退出时崩在 at::native::mps::MetalShaderLibrary 析构，
所以 spandrel 与 mewzoom 两条路都必须走独立进程 + pysafe 包装。

为什么不用 spandrel：MewZoom 是 TrunkNet 架构，spandrel 0.4.2 的 42 个内置架构里没有它，
必须用官方 mewzoom 库（PyPI mewzoom 1.0.0，实测在 torch 2.12 上可用，需 --no-deps 安装）。
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

MODEL_CFG = dict(exciter_hidden_ratio=8, hidden_ratio=2, num_channels=96,
                 num_layers=64, skip_type="forward-gated", upscale_ratio=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = a.weights or os.path.join(here, "models", "mewzoom_v1_4x.safetensors")
    if not os.path.isfile(weights):
        print("missing weights: " + weights, flush=True)
        return 2

    import time
    import torch
    from safetensors.torch import load_file
    from mewzoom.model import MewZoomTrunkNet

    t0 = time.time()
    model = MewZoomTrunkNet(**MODEL_CFG).eval()
    sd = load_file(weights)
    # 官方权重以 model. 为前缀，模型本身没有这层，必须剥掉
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 只容忍零星不匹配；大面积不匹配说明权重和架构对不上，继续跑只会产出垃圾图。
    if len(missing) > 5 or len(unexpected) > 5:
        print("FATAL weights mismatch: missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)
        return 3
    if missing or unexpected:
        print("WARN weights mismatch: missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)
    scale = int(MODEL_CFG["upscale_ratio"])
    print("模型就绪 %.1fs · %s · x%d · %.2fM" % (
        time.time() - t0, dev.type, scale,
        sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    src = Image.open(a.src).convert("RGB")
    w, h = src.size
    arr = np.asarray(src).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
    print("整幅推理 %dx%d" % (w, h), flush=True)
    t1 = time.time()
    with torch.inference_mode():
        out = model.upscale(ten)
    if dev.type == "mps":
        torch.mps.synchronize()
    o = out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    oh, ow = int(o.shape[0]), int(o.shape[1])
    Image.fromarray((o * 255.0).round().astype(np.uint8)).save(a.dst)
    # 用实际输出尺寸汇报，不靠 w*scale 推算（模型内部若做 padding 会不一致）
    print("完成 %dx%d -> %dx%d · %.1fs" % (w, h, ow, oh, time.time() - t1), flush=True)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
