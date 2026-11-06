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
import math
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
    """返回 (xs, ys)，块大小不超过 tile，覆盖整图。

    2026-11-07 重写：原实现是「按 step 步进，末尾不够再补一块 n-tile」，
    在 1024² + (512,32) 上会得到 [0, 480, 512] —— 第二块 480..992 与第三块
    512..1024 重叠 480px（几乎整块重复计算），9 块里有 2 块是白算的。
    改成「先算块数、再在 [0, n-tile] 上均分起点」，重叠变得均匀（1024² 上
    384/16 的相邻重叠 64px 而不是 480px），块数不变但每块都在干有效活。
    """
    if w <= tile and h <= tile:
        return [0], [0]
    step = max(tile - overlap, 1)

    def axis(n):
        if n <= tile:
            return [0]
        count = int(math.ceil((n - tile) / step)) + 1
        if count <= 1:
            return [0]
        starts = [int(round(i * (n - tile) / (count - 1))) for i in range(count)]
        return sorted(set(starts))

    return axis(w), axis(h)


def patch_attn_mask_cache(model):
    """ATD 每次前向都在 CPU 现算 attn_mask，再整块 H2D。

    tile 512 时该张量是 (1024, 256, 256) = 268MB fp32，而它只依赖 (h, w)，
    同一次 job 里所有块的 (h,w) 完全相同 —— 缓存一次就够，块数越多省得越多。
    其它架构（ESRGAN 等）没有 calculate_mask，直接跳过。
    """
    try:
        cls = type(model.model)
        orig = cls.calculate_mask
    except AttributeError:
        return False

    cache = {}

    def _cached(self, x_size):
        key = (int(x_size[0]), int(x_size[1]))
        hit = cache.get(key)
        if hit is None:
            hit = orig(self, x_size)
            cache[key] = hit
        return hit

    cls.calculate_mask = _cached
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--model", default="remacri")
    # 2026-11-07：384/16 是 ATD 的安全区（AC_MSA 的 attn 张量 ≈0.9GB、attn_mask ≈151MB），
    # 且小块每像素更便宜（实测 256² 0.4534 ms/px vs 512² 0.5730 ms/px）。
    # 512/32 在 16GB 机器上会让 ATD 撞 swap，1024² 源要多算一倍多。
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=16)
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
    if patch_attn_mask_cache(model):
        print("attn_mask 已缓存（同形状块复用）", flush=True)
    print("模型就绪 %.1fs · %s · x%d" % (time.time() - t0, dev.type, scale), flush=True)
    # 结构化分段：app.py 的 _stream 解析后写进 job 与快照，用来回答「7.4s 里多少是启动」
    print("PHASE load %.2f" % (time.time() - t0), flush=True)

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
    print("PHASE infer %.2f" % (time.time() - t1), flush=True)
    print("完成 %dx%d -> %dx%d · %.1fs" % (w, h, w * scale, h * scale, time.time() - t0), flush=True)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
