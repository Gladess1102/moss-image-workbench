#!/usr/bin/env python3
"""MOSS-Image runner backed by moss-image-mlx 0.2.0（独立 venv，非系统环境）。

与 mlx_runner.py 的参数对齐，但走的是 moss-image-mlx 的实现：
  * 单前向比 mflux 补丁版快约 7–8%
  * 内部按 low_memory 分阶段释放（TE → DiT → VAE），不需要外部两阶段编排
  * 支持 --turbo：15–16 次前向追 50 步质量（约 10 分钟 / 1MP）

依赖要点（改之前务必知道）：
  * venv: ~/moss-image/venv-msmlx（mlx 0.32.4 / mlx-lm 0.32.2 / moss-image-mlx 0.2.0）
  * venv 里的 text_encoder.py 打过补丁：本地 4bit TE 权重已是 MLX 布局，
    不能重复做 sanitize 的 conv1d moveaxis / (1+w) 偏移，否则出图完全不跟 prompt。
  * 模型目录 ~/moss-image/models/mlx_moss_sft 是符号/硬链接拼的干净目录，
    用来绕开 macOS AppleDouble 的 ._* 文件（会被 glob 当 safetensors 读）。
"""
import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = os.environ.get("MOSS_MODELS") or str(ROOT / "models")
BASE = os.environ.get("MOSS_MSMLX_MODEL") or (MODELS + "/mlx_moss_sft")
TURBO_STUDENT = ("VisualInference/MOSS-Image-SFT-Turbo-MLX", "turbo_student.safetensors")
DESTRIPE = ("nynxz/MOSS-Image", "loras/moss_sft_destripe_top64_bf16.safetensors")


def _install_skip(a, mx):
    """把 pipeline._sample 换成带自适应跳步的版本（不改库文件，运行时替换）。"""
    import numpy as np
    import moss_image_mlx.pipeline as P

    def _cos(x, y):
        na = mx.sqrt(mx.sum(x * x))
        nb = mx.sqrt(mx.sum(y * y))
        return float(mx.sum(x * y) / (na * nb + 1e-9))

    def _rel_drift(x, y):
        """相对漂移 mean|x-y|/mean|y|（与 mflux024_adapter._rel_drift 同口径）。"""
        return float(mx.mean(mx.abs(x - y))) / (float(mx.mean(mx.abs(y))) + 1e-8)

    def _sample(model, req, positive, negative, latents, on_step):
        req.validate(len(model.layers))
        if req.turbo:
            from moss_image_mlx import turbo
            return turbo.sample(model, req, positive, negative, latents, on_step)
        P.set_strength(model, 1.0, "default")
        if latents is None:
            latents = mx.array(P.randn(
                (1, model.config["latent_channels"], req.height // 16, req.width // 16), req.seed))
        sigmas = np.array(req.sigmas, dtype=np.float32) if req.sigmas else P.sigmas_for(req.steps, req.shift)
        timesteps = P.timesteps_for(sigmas)
        n = len(timesteps)
        lo = max(1, a.skip_warmup)
        hi = n - max(1, a.skip_tail)
        skipped, pred_prev, last_cos, just_skipped = [], None, None, False
        acc_drift, last_drift = 0.0, None
        drift_cap = max(1, round(a.skip * max(1, hi - lo))) if a.skip_mode == "drift" else 0
        for i, t in enumerate(timesteps):
            skip_active = req.slg_start <= i / n < req.slg_stop
            guidance = req.step_guidance[i] if req.step_guidance else req.guidance
            do_cfg = guidance != 1.0 and negative is not None
            if req.step_lora:
                P.set_strength(model, req.step_lora[i], "default")
            can_skip = pred_prev is not None and lo <= i < hi
            if a.skip_mode == "drift":
                est = (last_drift if last_drift is not None else 0.0) * 0.9
                do_skip = (can_skip and last_drift is not None and not just_skipped
                           and acc_drift + est <= a.skip_drift
                           and len(skipped) < drift_cap)
            elif a.skip_mode == "adaptive":
                do_skip = (can_skip and not just_skipped and last_cos is not None
                           and last_cos >= a.skip_cos)
            else:  # uniform：固定比例（之前漏实现，只有 adaptive 生效）
                do_skip = can_skip and not just_skipped and \
                    (i % max(1, round(1.0 / a.skip))) == 0
            if do_skip:
                prediction = pred_prev
                skipped.append(i)
                just_skipped = True
                if a.skip_mode == "drift" and last_drift is not None:
                    acc_drift += last_drift
            else:
                cond = model(latents, float(t), positive).astype(mx.float32)
                prediction = cond
                if do_cfg:
                    skp = req.uncond_skip_layers if (req.uncond_skip_layers and skip_active) else ()
                    uncond = model(latents, float(t), negative, skp).astype(mx.float32)
                    prediction = uncond + guidance * (cond - uncond)
                if req.slg_layers and req.slg_scale > 0 and skip_active:
                    sk = model(latents, float(t), positive, req.slg_layers).astype(mx.float32)
                    if not do_cfg:
                        prediction = sk
                    prediction = prediction + req.slg_scale * (cond - sk)
                if pred_prev is not None:
                    last_cos = _cos(prediction, pred_prev)
                    last_drift = _rel_drift(prediction, pred_prev)
                acc_drift = 0.0
                pred_prev = prediction
                just_skipped = False
            dt = mx.array(sigmas[i + 1] - sigmas[i], dtype=mx.float32)
            latents = latents.astype(mx.float32) + dt * prediction
            mx.eval(latents)
            if on_step:
                on_step(i, n)
        if skipped:
            print("实际跳过 %d/%d 步：%s" % (len(skipped), n, skipped), flush=True)
        return latents

    P._sample = _sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"], help="保留参数：moss-image-mlx 内部自己分阶段")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", default="", help="逗号分隔可传多个 LoRA（与 mlx_runner 同口径）")
    ap.add_argument("--lora-scale", default="1.0", help="逗号分隔；单值会广播到所有 LoRA")
    ap.add_argument("--turbo", action="store_true",
                    help="Turbo 采样：忽略 steps/guidance，走 5 student + 2 guided + 8 DPM++")
    # 跳步：monkeypatch moss_image_mlx.pipeline._sample，把「复用上一步 prediction」
    # 插进它的 Euler 循环。turbo 走 turbo.sample，固定 15 次前向，不参与跳步。
    ap.add_argument("--skip", type=float, default=0.0)
    ap.add_argument("--skip-mode", choices=["uniform", "adaptive", "drift"], default="uniform")
    ap.add_argument("--skip-cos", type=float, default=0.972)
    ap.add_argument("--skip-drift", type=float, default=0.03,
                    help="drift 模式的漂移预算（相对单位）；skip 表示最大跳步比例")
    ap.add_argument("--skip-warmup", type=int, default=3)
    ap.add_argument("--skip-tail", type=int, default=3)
    ap.add_argument("--cache-limit-gb", type=float, default=2.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not a.out:
        ap.error("--out is required")

    import mlx.core as mx
    if a.cache_limit_gb >= 0:
        mx.set_cache_limit(int(a.cache_limit_gb * 1024 ** 3))

    from moss_image_mlx import MOSSImagePipeline, Request, save

    if a.skip > 0 and not a.turbo:
        _install_skip(a, mx)

    prompt = a.prompt
    if a.prompt_file:
        prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip()

    loras = []
    if a.turbo:
        from huggingface_hub import hf_hub_download
        student = hf_hub_download(*TURBO_STUDENT)
        destripe = hf_hub_download(*DESTRIPE)
        # 顺序固定：先 destripe（全程 1.5），再 student（按步切换），最后才是用户 LoRA
        loras = [(destripe, 1.5, mx.bfloat16, "destripe"),
                 (student, 1.0, mx.float32, "student")]
        print("Turbo 适配器就绪（student 595MB + destripe）", flush=True)
    if a.lora:
        # 关键：akk / dmd2 的蒸馏 delta 是 fp32 存的，官方 README 写的是
        # "bf16 base + fp32 LoRA"。moss-image-mlx 的 apply_lora 默认把 LoRA 转 bf16 计算，
        # 而 DMD 的 delta 比 bf16 的舍入步长还小，转 bf16 会把它抹掉——4 步档会直接不跟 prompt。
        # 所以必须显式指定 fp32（底模仍是 8bit，只有低秩通路走 fp32）。
        paths = [p.strip() for p in str(a.lora).split(",") if p.strip()]
        scales = [float(s) for s in str(a.lora_scale).split(",") if s.strip()]
        if len(scales) == 1:
            scales = scales * len(paths)
        elif len(scales) != len(paths):
            raise SystemExit("--lora-scale 数量必须为 1，或与 --lora 数量一致")
        for _path, _scale in zip(paths, scales):
            loras.append((_path, _scale, mx.float32))

    t0 = time.time()
    pipe = MOSSImagePipeline(BASE, loras=loras, cache_dir=None)
    req = Request(
        prompt=prompt,
        negative_prompt=a.negative,
        width=a.width, height=a.height,
        steps=max(1, a.steps), guidance=a.guidance,
        seed=a.seed, turbo=a.turbo,
    )

    # 前端按 "第 N/M 步完成 (x.xs)" 解析进度，格式必须与 mlx_runner 一致；
    # 括号里必须是**单步耗时**。之前打印的是进程启动起的累计时间，app.py 把它当
    # 每步耗时算 ETA，导致 turbo 的"剩余时间"越跑越大（第 9 步能显示 800s+）。
    _last = [t0]

    # turbo 的真实前向数是 15（ACTORS="SSTSSS" → 5×1 + 1×2 = 7，finish 4×2 = 8），
    # 但 moss-image-mlx 只在 10 个调度点回调（6 个 actor + 4 个 finish）。
    # 直接打印 i/10 会让界面显示"5/10"，和档位名"15 步"对不上，所以映射回前向序号：
    #   actor i=0..5 → 1,2,4,5,6,7   （第 3 个 actor 是 T，带 CFG 记作 2 次）
    #   finish i=6..9 → 9,11,13,15
    _ACTOR_FWD = (1, 2, 4, 5, 6, 7)

    def on_step(i, n):
        now = time.time()
        per = now - _last[0]
        _last[0] = now
        if a.turbo:
            total = 15
            done = _ACTOR_FWD[i] if i < len(_ACTOR_FWD) else 7 + 2 * (i - (len(_ACTOR_FWD) - 1))
        else:
            total, done = n, i + 1
        print("第 %d/%d 步完成 (%.1fs)" % (done, total, per), flush=True)

    img = pipe(req, on_step=on_step)
    save(img, req, a.out, {
        "timings": dict(pipe.timings),
        "engine": "moss-image-mlx 0.2.0",
        "turbo": bool(a.turbo),
        "loras": [{"file": os.path.basename(p), "strength": s} for p, s, *_ in
                  (loras if not a.turbo else loras[:0])],
    })
    print("OK image -> " + a.out, flush=True)


if __name__ == "__main__":
    main()
