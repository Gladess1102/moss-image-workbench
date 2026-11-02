#!/usr/bin/env python3
"""下载本工作台需要的权重。

默认拉 16 GB 机器用的 compact tier；--tier 8bit 是 9.2 GB 峰值档（建议 >=24 GB 内存）。

产出：
  models/mlx_moss_sft/   tier 目录（transformer / text_encoder / tokenizer / vae / scheduler）
  models/mlx_vi_8bit     指向 mlx_moss_sft 的符号链接（mflux 适配层读这个目录名）
  models/akhaliq/        空目录，等你放入 4 步蒸馏 LoRA
  models/loras/          空目录，放你自己的角色 LoRA

可中断重跑：已下载的文件不会重复传输。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("MOSS_MODELS") or (ROOT / "models"))

TIERS = {
    "compact": "VisualInference/MOSS-Image-SFT-MLX-compact",
    "8bit": "VisualInference/MOSS-Image-SFT-MLX-8bit",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="compact")
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

    print()
    print("权重就绪。还差一个可选件：4 步蒸馏 LoRA（akhaliq v3）")
    print("  放到 models/akhaliq/akk_4step_lora.safetensors 即可启用「4步·极速」档；")
    print("  不放也能跑，默认档会自动回落到 20 步。")
    print()
    print("下一步：  .venv-mflux/bin/python scripts/doctor.py")


if __name__ == "__main__":
    main()
