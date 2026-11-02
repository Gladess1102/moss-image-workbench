#!/usr/bin/env python3
"""下载本工作台需要的权重。

默认拉 16 GB 机器用的 compact tier；--tier 8bit 是 9.2 GB 峰值档（建议 >=24 GB 内存）。

产出：
  models/mlx_moss_sft/   tier 目录（transformer / text_encoder / tokenizer / vae / scheduler）
  models/mlx_vi_8bit     指向 mlx_moss_sft 的符号链接（mflux 适配层读这个目录名）
  models/akhaliq/        4 步蒸馏 LoRA（自动从 akhaliq/MOSS-Image-SFT-4-steps 下载）
  models/loras/          空目录，放你自己的角色 LoRA

可中断重跑：已下载的文件不会重复传输。
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("MOSS_MODELS") or (ROOT / "models"))

TIERS = {
    "compact": "VisualInference/MOSS-Image-SFT-MLX-compact",
    "8bit": "VisualInference/MOSS-Image-SFT-MLX-8bit",
}

# akhaliq 的 4 步蒸馏 LoRA：kohya 训练，键名与 mflux 的 756 个 diffusion_model.* 张量一一对应，
# 不需要转换。euler-* 是 Euler 原生版本（免 dataspace / time_embedding 两件套）。
AKK4_REPO = "akhaliq/MOSS-Image-SFT-4-steps"
AKK4_FILE = "euler-update-100/checkpoint-update-100/lora.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="compact")
    ap.add_argument("--no-4step", action="store_true", help="跳过 1.2 GB 的 4 步蒸馏 LoRA")
    a = ap.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("缺 huggingface_hub：先跑 install.sh，或用装了依赖的解释器执行本脚本")

    MODELS.mkdir(parents=True, exist_ok=True)
    repo = TIERS[a.tier]
    dst = MODELS / "mlx_moss_sft"
    if dst.exists() and (dst / "transformer").is_dir():
        print("[1/2] tier 已存在，跳过下载：%s" % dst)
    else:
        print("[1/2] 下载 %s -> %s（约 8 GB，可中断重跑）" % (repo, dst))
        snapshot_download(repo, local_dir=str(dst), max_workers=4)

    link = MODELS / "mlx_vi_8bit"
    if not link.exists():
        os.symlink("mlx_moss_sft", link)
        print("[2/2] 建链接：models/mlx_vi_8bit -> models/mlx_moss_sft")
    else:
        print("[2/2] 链接已存在：%s" % link)

    for sub in ("akhaliq", "loras", "turbo"):
        (MODELS / sub).mkdir(exist_ok=True)

    target = MODELS / "akhaliq" / "akk_4step_lora.safetensors"
    if a.no_4step:
        print("[3/3] 跳过 4 步 LoRA（--no-4step）")
    elif target.exists():
        print("[3/3] 4 步 LoRA 已存在：%s" % target)
    else:
        print("[3/3] 下载 4 步蒸馏 LoRA（约 1.2 GB）")
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(AKK4_REPO, AKK4_FILE)
        shutil.copyfile(got, target)
        print("      已放到 %s" % target)

    print()
    print("权重就绪。4 步档（默认）与 20/25/50 步档都可直接使用。")
    print()
    print("下一步：  .venv-mflux/bin/python scripts/doctor.py")


if __name__ == "__main__":
    main()
