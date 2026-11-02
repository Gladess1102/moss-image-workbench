#!/usr/bin/env python3
"""环境自检：模型、引擎、端口、磁盘、Apple Silicon。有问题会明确告诉你怎么修。"""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("MOSS_MODELS") or (ROOT / "models"))
OUT = Path(os.environ.get("MOSS_OUT") or Path.home() / "Pictures" / "MOSS-Image")
PORT = int(os.environ.get("MOSS_PORT") or 7860)
RESULTS = []


def check(ok, name, detail="", fix=""):
    RESULTS.append({"ok": bool(ok), "name": name, "detail": detail, "fix": fix})
    mark = "OK " if ok else "FAIL"
    print("[%s] %s%s" % (mark, name, (" — " + detail) if detail else ""))


def warn(name, detail, fix=""):
    RESULTS.append({"ok": None, "name": name, "detail": detail, "fix": fix})
    print("[WARN] %s — %s" % (name, detail))


def python_of(env, default):
    p = os.environ.get(env)
    if p:
        return Path(p)
    cand = ROOT / default / "bin" / "python"
    return cand if cand.exists() else None


def main():
    print("MOSS-Image Workbench 自检")
    print("仓库: %s" % ROOT)
    print("模型: %s" % MODELS)
    print()

    check(sys.platform == "darwin", "操作系统", sys.platform, "只支持 macOS")
    check(platform.machine() == "arm64", "CPU 架构", platform.machine(), "需要 Apple Silicon")
    try:
        mem_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True).stdout) / 1024 ** 3
        check(mem_gb >= 16, "统一内存", "%.1f GB" % mem_gb, "16 GB 是下限，建议 24 GB+")
    except Exception:
        warn("统一内存", "读取失败")

    tier = MODELS / "mlx_moss_sft" / "transformer" / "config.json"
    check(tier.exists(), "DiT 权重", str(tier.parent.parent),
          "跑 scripts/download_models.py")
    for comp in ("text_encoder", "tokenizer", "vae", "scheduler"):
        p = MODELS / "mlx_moss_sft" / comp
        check(p.is_dir(), "组件 " + comp, str(p), "重新跑 scripts/download_models.py")
    link = MODELS / "mlx_vi_8bit"
    check(link.exists(), "mlx_vi_8bit 链接", str(link),
          "ln -s mlx_moss_sft models/mlx_vi_8bit")

    akk = MODELS / "akhaliq" / "akk_4step_lora.safetensors"
    if akk.exists():
        check(True, "4 步蒸馏 LoRA", "%.0f MB" % (akk.stat().st_size / 1e6))
    else:
        warn("4 步蒸馏 LoRA", "未安装，4 步档不可用（会自动回落 20 步）",
             "把 akhaliq v3 放到 models/akhaliq/akk_4step_lora.safetensors")

    for env, default, label in (("MOSS_MFLUX_PY", ".venv-mflux", "主引擎 venv"),
                                ("MOSS_MSMLX_PY", ".venv-msmlx", "第二引擎 venv")):
        py = python_of(env, default)
        check(py is not None, label, str(py or "未找到"),
              "跑 install.sh%s" % (" --with-msmlx" if "MSMLX" in env else ""))

    # 依赖能在子进程里 import 才算数（主进程不 import mlx / torch）
    probes = [("mlx8", ["mlx", "mflux"], "MOSS_MFLUX_PY", ".venv-mflux"),
              ("msmlx", ["mlx", "moss_image_mlx"], "MOSS_MSMLX_PY", ".venv-msmlx")]
    for name, mods, env, default in probes:
        py = python_of(env, default)
        if py is None:
            continue
        code = "import %s" % ", ".join(mods).replace(", ", "; import ")
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
        check(r.returncode == 0, "引擎依赖 " + name,
              "ok" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:120],
              "pip install -r requirements%s.txt" % ("-msmlx" if name == "msmlx" else ""))

    try:
        OUT.mkdir(parents=True, exist_ok=True)
        probe = OUT / ".doctor-write-test"
        probe.write_text("ok")
        probe.unlink()
        check(True, "输出目录可写", str(OUT))
    except Exception as e:
        check(False, "输出目录可写", str(e), "设置 MOSS_OUT 指向可写目录")

    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    check(free_gb >= 12, "磁盘可用", "%.1f GB" % free_gb, "至少留 12 GB（权重 + 出图）")

    import socket
    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", PORT)) == 0
    if busy:
        warn("端口 %d" % PORT, "已被占用", "换端口：MOSS_PORT=7861 ./run.sh")
    else:
        check(True, "端口 %d" % PORT, "空闲")

    dots = list(MODELS.rglob("._*"))
    if dots:
        warn("AppleDouble 文件", "%d 个 ._* 文件，moss-image-mlx 会把它当权重读" % len(dots),
             "dot_clean -m models/")
    else:
        check(True, "AppleDouble 检查", "干净")

    fails = [r for r in RESULTS if r["ok"] is False]
    print()
    if fails:
        print("有 %d 项没通过，按上面的 fix 处理。" % len(fails))
        return 1
    print("全部通过。启动：./run.sh")
    return 0


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    sys.exit(main())
