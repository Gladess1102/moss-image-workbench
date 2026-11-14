#!/usr/bin/env python3
"""mflux 0.24 适配层的 CLI：支持两阶段（对应 fork 的 mlx_runner.py）。

  # 一站式
  mflux024_generate.py --prompt "..." -o out.png --width 512 --height 512 --steps 4 --guidance 1.0

  # 两阶段（16GB 机器推荐：编码阶段只装 TE，出图阶段只装 DiT）
  mflux024_generate.py --stage encode --prompt "..." --negative "..." --embeds job.embeds
  mflux024_generate.py --stage run --embeds job.embeds -o out.png --width 512 --height 512 --steps 4
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent

import mlx.core as mx

# 统一读 MOSS_MODELS（= 模型根目录），与 app.py / doctor.py 同源；
# 旧变量 MOSS_MODEL（单数，直接指 tier 目录）保留兼容。
MODELS = os.environ.get("MOSS_MODELS") or str(ROOT / "models")
MODEL = os.environ.get("MOSS_MODEL") or os.path.join(MODELS, "mlx_vi_8bit")


def _assert_finite_embeds(pos, neg, where):
    """文本 embedding 一旦出 NaN，采样出来就是一张 117 字节的纯黑图，而工作台
    只检查"文件存在 + 能读尺寸"，会把它标成成功。宁可在这里直接失败。

    实测（2026-10-29）：Qwen3.5 的 Gated DeltaNet 在长 prompt 上数值会炸——
    923 字符 Layout JSON（552 token）在第 25 层的第 5 个 chunk 开始出 NaN，
    而 500 字符（311 token）还正常。阈值大约在 300～550 token 之间。
    """
    import numpy as np

    for name, arr in (("pos", pos), ("neg", neg)):
        if arr is None:
            continue
        a = np.array(arr.astype(mx.float32))
        if np.isfinite(a).all():
            continue
        n_tok = int(a.shape[1]) if a.ndim == 3 else int(a.shape[0])
        bad = int((~np.isfinite(a)).sum())
        raise SystemExit(
            "文本编码溢出（%s）：%s 里有 %d 个非有限值（%d token）。\n"
            "prompt 过长时 Qwen3.5 的线性注意力会数值爆炸，出图会是纯黑。\n"
            "请把 prompt 缩到约 400 字符以内，或改用 msmlx 引擎跑长 Layout JSON。"
            % (where, name, bad, n_tok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--stage", default="all", choices=["all", "encode", "run"])
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", default="", help="从文件读 prompt（长 Layout JSON 用）")
    ap.add_argument("--negative", default="")
    ap.add_argument("--embeds", default="")
    ap.add_argument("--out", "-o", default="")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uncond-skip-layers", default="22")
    ap.add_argument("--lora", default="", help="逗号分隔可传多个")
    ap.add_argument("--lora-scale", default="1.0", help="逗号分隔或单值广播")
    ap.add_argument("--te", default="", help="覆盖 text_encoder 目录（默认 model/text_encoder）")
    ap.add_argument("--stream", action="store_true",
                    help="按层流式加载 DiT（常驻 5.5GB -> 0.5GB，单步约 +1.7%%，输出逐比特一致）")
    ap.add_argument("--time-embedding", default="")
    ap.add_argument("--skip", type=float, default=0.0, help="跳步比例，0=关")
    ap.add_argument("--skip-mode", default="adaptive",
                    choices=["adaptive", "fixed", "uniform", "official", "drift", "hybrid"])
    ap.add_argument("--cache-limit-gb", type=float, default=1.0,
                    help="MLX buffer 池上限（GB）；0=不限。16G 机器必须限，否则释放的 Metal buffer 不还")
    ap.add_argument("--skip-cos", type=float, default=0.972)
    ap.add_argument("--skip-drift", type=float, default=0.03,
                    help="drift 模式的漂移预算（相对单位，0.01~0.15）")
    ap.add_argument("--skip-warmup", type=int, default=3)
    ap.add_argument("--skip-extrap", action="store_true",
                    help="跳步时用最近两个真实 pred 做半额线性外推（替代直接复制）")
    ap.add_argument("--skip-tail", type=int, default=3)
    ap.add_argument("--decode-cache-gb", type=float, default=0.25,
                    help="VAE 解码期间 MLX buffer 池上限（GB）；只影响峰值，像素不变")
    ap.add_argument("--keep-dit", action="store_true",
                    help="解码前不释放 DiT（调试用；默认释放，16G 机器必须释放）")
    a = ap.parse_args()

    # 工作台传长 prompt 走文件，避开 argv 长度与引号坑
    if a.prompt_file:
        a.prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip()
    # 旧 mlx_runner 的 "uniform" 语义 = 固定比例跳步
    if a.skip_mode == "uniform":
        a.skip_mode = "fixed"
    # 必须在加载任何模型之前设，否则 buffer 池已经涨上去
    if a.cache_limit_gb and a.cache_limit_gb > 0:
        mx.set_cache_limit(int(a.cache_limit_gb * 1024 ** 3))

    import mflux024_adapter as A
    if a.te:
        # 让适配层用指定的 TE 目录（适配层已支持直接传 TE 目录）
        _orig_load_te = A.load_text_encoder
        A.load_text_encoder = lambda md, **kw: _orig_load_te(a.te, **kw)

    skip_layers = tuple(int(x) for x in str(a.uncond_skip_layers).replace(",", " ").split()) \
        if str(a.uncond_skip_layers).strip() else ()

    # ---------------- encode 阶段
    if a.stage == "encode":
        t0 = time.time()
        te = A.load_text_encoder(a.model)
        tok = A.load_tokenizer(a.model)
        neg_arg = a.negative if (a.negative or a.guidance != 1.0) else None
        pos, neg = A.encode_prompts(te, tok, a.prompt, neg_arg)
        _assert_finite_embeds(pos, neg, "encode")
        if not a.embeds:
            a.embeds = "embeds.safetensors"
        A.save_embeds(a.embeds, pos, neg)
        print("OK embeds -> %s  (%.1fs)" % (a.embeds, time.time() - t0), flush=True)
        return

    # ---------------- run / all
    t0 = time.time()
    if a.stage == "run":
        if not a.embeds:
            raise SystemExit("--stage run 需要 --embeds")
        pos, neg = A.load_embeds(a.embeds)
        _assert_finite_embeds(pos, neg, "run")
        print("embeds 已加载 %s" % a.embeds, flush=True)
    else:
        te = A.load_text_encoder(a.model)
        tok = A.load_tokenizer(a.model)
        neg_arg = a.negative if (a.negative or a.guidance != 1.0) else None
        pos, neg = A.encode_prompts(te, tok, a.prompt, neg_arg)
        # 与两阶段对齐（save_embeds 存 fp32）
        pos = pos.astype(mx.float32)
        if neg is not None:
            neg = neg.astype(mx.float32)
        del te
        mx.clear_cache()

    paths = [p.strip() for p in a.lora.split(",") if p.strip()] if a.lora else []
    scales = [float(s) for s in str(a.lora_scale).split(",") if s.strip()] if a.lora else []
    if scales and len(scales) == 1:
        scales = scales * len(paths)

    if a.stream:
        # 流式：42 层按层 mmap，常驻内存 5.5 GB -> ~0.5 GB。LoRA 必须在加载时传入，
        # 因为层是动态构建的，事后 apply_lora 找不到 layers。
        dit = A.load_transformer(a.model, streaming=True,
                                 lora_paths=paths or None, lora_scales=scales or None)
        print("流式加载：按层 mmap（常驻 ~0.5 GB）", flush=True)
    else:
        dit = A.load_transformer(a.model)
    vae = A.load_vae(a.model)

    if a.time_embedding:
        n = A.apply_time_embedding_override(dit, a.time_embedding)
        print("时间嵌入覆盖: %d 张量" % n, flush=True)
    if paths and not a.stream:
        A.apply_lora(dit, paths, scales)
    if paths:
        print("LoRA: %s" % ", ".join("%s@%s" % (p.rsplit("/", 1)[-1], s)
                                     for p, s in zip(paths, scales)), flush=True)

    n_steps = a.steps
    last = [time.time()]
    fwd = [0]

    def on_step(i, n, was_skip):
        now = time.time()
        if not was_skip:
            fwd[0] += 2 if (a.guidance != 1.0 and neg is not None) else 1
        print("第 %d/%d 步%s (%.1fs)" % (i + 1, n, " [跳]" if was_skip else "", now - last[0]),
              flush=True)
        last[0] = now

    latents, stats = A.sample(
        dit, pos, neg, seed=a.seed, height=a.height, width=a.width,
        steps=n_steps, guidance=a.guidance, shift=a.shift,
        uncond_skip_layers=skip_layers, skip=a.skip, skip_mode=a.skip_mode,
        skip_cos=a.skip_cos, skip_warmup=a.skip_warmup, skip_tail=a.skip_tail,
        skip_drift=a.skip_drift, extrapolate=a.skip_extrap, on_step=on_step)

    # 解码前释放 DiT：解码阶段它已经没用了，但它占着的驻留权重会让 VAE 的
    # fp32 中转 buffer 没有回收空间（mlx-vlm #2615 的第二个必要条件）。
    # stream 模式下常驻只有 ~0.5 GB，收益小，但行为一致。
    if not a.keep_dit:
        del dit
        mx.clear_cache()
        print("解码前已释放 DiT", flush=True)

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    t_dec = time.time()
    img = A.decode(vae, latents, cache_limit=int(a.decode_cache_gb * 1024 ** 3))
    if hasattr(mx, "get_peak_memory"):
        print("解码 %.1fs · 解码阶段 MLX 峰值 %.2f GB"
              % (time.time() - t_dec, mx.get_peak_memory() / 1024 ** 3), flush=True)

    if a.out:
        from PIL import Image
        Image.fromarray(img).save(a.out)
        print("OK image -> %s" % a.out, flush=True)
    print("耗时 %.1fs · 跳步 %d/%d · 前向 %d 次 · 位置 %s"
          % (time.time() - t0, len(stats["skipped"]), stats["total"], fwd[0],
             sorted(stats["skipped"])), flush=True)
    cl = stats.get("cos_log") or []
    if cl:
        print("cos 曲线: " + " ".join("%.3f" % (c if c is not None else 0) for c in cl), flush=True)


if __name__ == "__main__":
    main()