#!/usr/bin/env python3
"""ESRGAN 超分执行器 —— 独立进程。

为什么单独一个进程：工作台主进程绝不能 import torch/torchvision，
否则退出时会崩在 at::native::mps::MetalShaderLibrary 析构（macOS 27 + torch 2.14）。
本进程由 app.py 经 pysafe 包装调用，pysafe 会在 C 层析构前 os._exit。

用法:
  upscale_runner.py --in A.png --out B.png [--model remacri] [--tile 512] [--overlap 32]
  --model 的取值与权重路径见 upscale_registry.py（单一数据源，本文件不再自带模型表）。

进度行（供 app.py 解析）:
  UPSCALE <cur>/<total> <sec>
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

# pysafe 用 runpy.run_path 执行本文件：__file__ 正常，但脚本目录不在 sys.path 里
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upscale_registry  # 纯数据模块，不引 torch/mlx

# 小于这个像素数就整幅推理；超过则分块，避免 swap
WHOLE_LIMIT = 500_000


def plan_tiles(w, h, tile, overlap):
    """返回 (xs, ys)，块大小不超过 tile，覆盖整图。"""
    if w <= tile and h <= tile:
        return [0], [0]
    step = max(tile - overlap, 1)

    def axis(n):
        if n <= tile:
            return [0]
        pos = list(range(0, n - tile + 1, step))
        if pos[-1] + tile < n:
            pos.append(n - tile)
        return pos

    return axis(w), axis(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--model", default="remacri")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)
    a = ap.parse_args()

    try:
        path, _mdef = upscale_registry.resolve(a.model)
    except KeyError:
        print("unknown model: " + a.model, flush=True)
        return 2
    if not path or not os.path.isfile(path):
        print("missing weights: " + str(path), flush=True)
        return 2

    import torch
    from spandrel import ModelLoader

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("加载 " + os.path.basename(path), flush=True)
    t0 = time.time()
    model = ModelLoader().load_from_file(path)
    model.eval()
    model.to(dev)
    scale = int(getattr(model, "scale", 4) or 4)
    print("模型就绪 %.1fs · %s · x%d" % (time.time() - t0, dev.type, scale), flush=True)

    src = Image.open(a.src).convert("RGB")
    w, h = src.size
    arr = np.asarray(src).astype(np.float32) / 255.0

    if w * h >= WHOLE_LIMIT:
        xs, ys = plan_tiles(w, h, a.tile, a.overlap)
    else:
        xs, ys = [0], [0]
    total = len(xs) * len(ys)
    if total == 1:
        print("整幅推理 %dx%d" % (w, h), flush=True)
    else:
        print("分块推理 %d 块 · tile=%d overlap=%d" % (total, a.tile, a.overlap), flush=True)

    canvas = Image.new("RGB", (w * scale, h * scale))
    cur = 0
    t1 = time.time()
    # 整幅推理时不能被 tile 截断：像 640×480 这种"像素数低于 WHOLE_LIMIT 但边长超过 tile"
    # 的图，会被 min(tile, w-x) 切掉右侧/底部，在输出里留下纯黑边。
    # 2026-10-21 修：整幅路径直接用实际剩余宽高。
    whole = (total == 1)
    for y in ys:
        for x in xs:
            tw = (w - x) if whole else min(a.tile, w - x)
            th = (h - y) if whole else min(a.tile, h - y)
            patch = arr[y:y + th, x:x + tw]
            ten = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                out = model(ten)
            out = out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            canvas.paste(Image.fromarray((out * 255.0).round().astype(np.uint8)),
                         (x * scale, y * scale))
            cur += 1
            print("UPSCALE %d/%d %.2f" % (cur, total, time.time() - t1), flush=True)

    canvas.save(a.dst)
    print("完成 %dx%d -> %dx%d · %.1fs" % (w, h, w * scale, h * scale, time.time() - t0), flush=True)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
