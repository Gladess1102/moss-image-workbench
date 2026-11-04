#!/usr/bin/env python3
"""MOSS-Image local workbench.

Engines
  mlx8   MLX 8-bit DiT via mlx_runner.py
         LoRA 档：4 步 akhaliq v3（Euler 原生单文件）
         two_stage on by default: encoder and DiT live in separate processes,
         so the 16 GB machine never swaps during generation.
"""
import hashlib, json, math, os, random, re, select, shutil, signal, subprocess, sys, tempfile, threading, time, uuid
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs

from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, BASE)          # 同目录模块（pngtools / upscale_registry）

# ── 2026-10-24 审查修复 ────────────────────────────────────────────────
# 1) /api/shutdown 原来是无鉴权 GET：浏览器里任意网页写一行
#    <img src="http://127.0.0.1:7860/api/shutdown"> 就能杀掉服务和正在跑的任务。
#    改成要求本地 token（写在 /tmp，desktop.py / .command 读它）。
# 2026-11-04：token 文件名带 pid —— 原来固定 /tmp/moss-shutdown.token，两个实例会互相
# 覆盖；并且改成只在 __main__ 里落盘（compress_history.py 会 import 本模块，不该有副作用）。
SHUTDOWN_TOKEN = os.environ.get("MOSS_SHUTDOWN_TOKEN") or uuid.uuid4().hex


def token_path(pid=None):
    """token 文件路径。desktop.py 按 /api/health 返回的 pid 读同名文件。"""
    return "/tmp/moss-shutdown-%d.token" % (os.getpid() if pid is None else int(pid))


def _write_token_file():
    """一次建好并带 0600：先 open 后 chmod 有窗口；O_NOFOLLOW 防符号链接截断目标文件。"""
    path = token_path()
    for attempt in (0, 1):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            if attempt:
                return
            try:
                os.remove(path)      # 同 pid 的上次残留（pid 复用）
            except OSError:
                return
            continue
        except OSError:
            return
        try:
            os.write(fd, SHUTDOWN_TOKEN.encode())
        finally:
            os.close(fd)
        return


# 2) pysafe 内部的 python 取自 $PYSAFE_PYTHON 或 PATH。
#    从 Finder/launchd 启动时 PATH 可能是 Apple 自带的 python3（无 torch/spandrel/mlx），
#    四个超分引擎会全部失效。这里钉死成当前解释器，并把它的 bin 前置进 PATH。
os.environ.setdefault("PYSAFE_PYTHON", sys.executable)
_PY_BIN = os.path.dirname(sys.executable)
if _PY_BIN and _PY_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _PY_BIN + ":" + os.environ.get("PATH", "")

MODELS = os.environ.get("MOSS_MODELS") or os.path.join(ROOT, "models")


def _pick_python(env_name, default_dir):
    """选引擎解释器：环境变量 > 仓库内默认 venv > 当前解释器。

    两个引擎的 mlx 依赖互斥（mflux 0.24 / moss-image-mlx 0.2.0），所以默认各用一个
    独立 venv；只装了一套时，用 MOSS_MFLUX_PY / MOSS_MSMLX_PY 指过去。
    """
    forced = os.environ.get(env_name)
    if forced:
        return forced, True
    cand = os.path.join(ROOT, default_dir, "bin", "python")
    return (cand, True) if os.path.exists(cand) else (sys.executable, False)
# 2026-10-28：mlx8 档从 mflux 0.23 fork（vanch007 分支 + 5 处手改）切到官方 0.24 + 本地适配层。
# 旧 fork 的补丁全部作废——两阶段 / 自适应跳步 / LoRA 运行时加载 / 时间嵌入覆盖
# 现在都由 web/mflux024_adapter.py 实现，CLI 是 web/mflux024_generate.py。
# 实测：25 步跳步 312.7s（fork 340s，少 4 次前向）、两阶段与单阶段逐像素一致、
# TE 余弦 0.9998（修掉 RMSNorm 语义相反那个致命坑之后）。
MFLUX024_PY, _MFLUX_OK = _pick_python("MOSS_MFLUX_PY", ".venv-mflux")
RUNNER = os.path.join(BASE, "mflux024_generate.py")
PY = sys.executable
# moss-image-mlx 跑在独立 venv 里（mlx 0.32.4 / mlx-lm 0.32.2），系统环境一行不动。
# 它内部自己做 TE→DiT→VAE 的分阶段释放，所以不需要 mlx_runner 那套外部两阶段编排。
VENV_PY, _MSMLX_OK = _pick_python("MOSS_MSMLX_PY", ".venv-msmlx")
MSMLX_RUNNER = os.path.join(BASE, "msmlx_runner.py")
MSMLX_BASE = os.path.join(MODELS, "mlx_moss_sft")
# 输出目录与 PNG 重压/快照的实现在 pngtools.py（compress_history.py 也要用，
# 但它不该 import 本模块去拿函数）。
import pngtools  # noqa: E402
from pngtools import (OUT, UPSCALE_DIR, _optimize_async, _optimize_png,  # noqa: E402
                      _read_meta, _write_meta)
os.makedirs(OUT, exist_ok=True)
os.makedirs(UPSCALE_DIR, exist_ok=True)

MLX8 = os.path.join(MODELS, "mlx_vi_8bit")
# 8 步 DMD2 档（lora_dmd2）已于 2026-10-30 撤下：权重文件删除、档位一并移除。
# 当时的取舍证据仍在 bench/dmd2-vs-old8step-20261022/；要恢复需重新取权重并加回 LORA_MODES。
# 旧 darask0 4 步档（lora_4step/ + dataspace_4step）已于 2026-10-22 淘汰：
# akhaliq v3 同尺寸下更快、文字更稳、且免 TE/dataspace 两件配套。
# 旧 SFT 三件套（lora_8step + time_embedding_8step + dataspace_8step）已于 2026-10-22 停用，
# 权重文件随之删除；现在 8 步档走 dmd2 单文件 + 标准 FlowMatch Euler。
# akhaliq v3（2026-10-20）：Euler 原生，单个 LoRA 即可，**不需要** time_embedding / dataspace。
# 键名格式与本地 lodestones/darask0 件完全一致（756 个 diffusion_model.* 张量），可直接加载。
AKK4 = os.path.join(MODELS, "akhaliq", "akk_4step_lora.safetensors")

# 角色 LoRA：用户自己练的人像/风格 LoRA 放这里，生成时可选叠加（与蒸馏档并行生效）。
# mlx_runner / msmlx_runner 都支持 --lora 逗号分隔多个、--lora-scale 单值广播。
LORA_DIR = os.path.join(MODELS, "loras")
os.makedirs(LORA_DIR, exist_ok=True)

# 2026-10-28：Coffmeteor 16 步蒸馏件已入库并转换键名（756/756）。
# 放在 models/coffmeteor/ 而不是 models/loras/——后者会被 _char_loras() 当成"角色 LoRA"
# 列给用户叠加，而蒸馏件叠到 akk4 上会把画面打崩。
# 暂不单独开档位：同前向次数下 Turbo（msmlx，15 次）的 stampscore dup 是 0.1194，
# Coffmeteor（16 次）是 0.1585，Turbo 明显更干净。要试的话加一条 LORA_MODES 即可。
COFF16 = os.path.join(MODELS, "coffmeteor", "coffmeteor_16step_kohya.safetensors")


def _char_loras():
    """列出 models/loras/ 下可叠加的角色 LoRA。"""
    out = []
    try:
        for n in sorted(os.listdir(LORA_DIR)):
            if not n.lower().endswith((".safetensors", ".ckpt")):
                continue
            fp = os.path.join(LORA_DIR, n)
            if os.path.isfile(fp):
                out.append({"id": n, "label": n.rsplit(".", 1)[0],
                            "size_mb": round(os.path.getsize(fp) / 1e6, 1)})
    except OSError:
        pass
    return out


GGUF = {
    "dit": os.path.join(MODELS, "MOSS-Image-SFT-Q8_0.gguf"),
    "llm": os.path.join(MODELS, "Qwen3.5-9B-Q4_K_M.gguf"),
    "vae": os.path.join(MODELS, "flux2-vae_small.safetensors"),
    "tok": os.path.join(MODELS, "tokenizer.json"),
}

ENGINES = {
    # 2026-10-28 换实现：不再是 fork，而是官方 mflux 0.24 + web/mflux024_adapter.py（529 行）。
    # 档位/参数/两阶段语义全部保持，所以前端与 LORA_MODES 不用改。
    "mlx8": {"label": "MLX 0.24", "note": "mflux 0.24 + 适配层 · 两阶段分进程 · 25步跳步 312.7s（旧 fork 340s）",
             "available": _MFLUX_OK},
    # 2026-10-23 接入：moss-image-mlx 0.2.0，纯 MLX 独立实现。
    # 单前向约快 7–8%（640×480 4步 51.9→47.6s、8步 96.3→89.7s，同机同 prompt/seed 实测），
    # 真正的价值是 Turbo 采样（见 LORA_MODES["turbo"]）与"任意步数都能走这条快路径"。
    # 引擎选择独立于档位：勾了它，自由步数也用它。
    "msmlx": {"label": "MOSS-MLX", "note": "moss-image-mlx 0.2.0 · 单前向快约 7–8% · Turbo 档必需 · 任意步数可用",
              "available": _MSMLX_OK},
}
LORA_MODES = {
    # akhaliq v3（2026-10-20）：Euler 原生，单 LoRA 走标准 FlowMatch（shift 5 / CFG 1）。
    # 实测同 prompt/seed、512²：4 步 44.4s、3 步 33.7s，招牌文字干净；
    # 同条件 darask0 档（lora4）40.5s，但瓦檐出现一排重复的"深夜食堂"。
    "akk4": {"label": "4步 · akhaliq v3", "steps": 4, "cfg": 1.0, "step_cache": 0.0,
             "eta640": "~44s（512²）", "desc": "Euler 原生单 LoRA · 免 dataspace/TE，文字最稳",
             "lora": True, "lora_scale": 1.0},
    # 8 步 DMD2（lora_dmd2）档已于 2026-10-30 撤下，权重同时删除；蒸馏档现只剩 akk4 与 turbo。
    # 2026-10-23：原「12步 · 原生」档撤下，由下面的 turbo 档接替。
    # 理由：12 步 CFG 3.0 = 24 次前向、约 21 分钟/1024²；turbo 15–16 次前向、约 10 分钟，
    # 实测同 prompt/seed 在 1184×896 上三行中文一字不差、材质与光影方向全对，质量明显更高。
    # turbo 走 moss-image-mlx 的固定调度（5 student + 2 guided + 8 DPM++ finish），
    # 忽略步数与 CFG，所以引擎必须是 MOSS-MLX。
    # cfg 必须 >1：turbo 里唯一那次 teacher 前向是 "guided step"（σ .957→.919 决定文字），
    # guidance=1.0 会退化成 uncond + 1*(cond-uncond) = cond，等于白跑 CFG 分支。
    # felix 的 CLI 默认就是 3.0（Request.guidance 默认值）。
    "turbo": {"label": "Turbo · 高质量", "steps": 15, "cfg": 3.0, "step_cache": 0.0,
              "eta640": "~10min（1184×896）", "desc": "15 次前向追 50 步质量 · 需 MOSS-MLX 引擎 · 支持负面词",
              "lora": False, "engine": "msmlx", "turbo": True},
    # StepCache 实测在 12 步上会把画面跳糊（有效步数只剩 7），20 步跳完也只剩 12 步，
    # 风险同源；在逐档实测通过之前一律不默认开启（要试就把 step_cache 改成 0.4）。
    # 2026-10-25 跳步档（自适应版）：CFG 3.0 下相邻 pred cos 只有 0.93~0.99（前期低、中段高、
    # 尾部再降），固定比例会在陡峭区硬跳。自适应按 cos 阈值决定，跳完必算一步，尾部自动收手。
    # 实测 25 步：跳 7/25 步，PSNR 33.5 dB（固定 0.33 同样跳 7 步只有 26.0 dB）。
    # 2026-10-28 实测（0.24 适配层，640×480，CFG 3.0，同 seed）：
    #   25 步不跳  499.7s / 50 次前向
    #   25 步自适应跳 7/25  351.3s / 36 次前向  → PSNR **26.33 dB**
    # 旧 fork 同样跳 7 步是 33.50 dB。0.24 的跳步质量只有"固定比例跳步"的水平（26.0 dB），
    # 说明适配层的自适应判据在 0.24 的轨迹上没落到平坦区（scheduler 实现与 fork 不同）。
    # 在重新标定 skip_cos 之前先关掉，避免拿到 26dB 的糊图。
    # 2026-10-28 在 0.24 适配层上重新标定（640×480，CFG 3.0，同 seed，基线=25 步不跳 499.7s）：
    #   warmup3  cos0.972  跳7/25  351s  PSNR 26.33  ← 前段被跳，构图跑偏
    #   官方 StepCache 0.3  跳8/25  366s  PSNR 10.47  ← flow-match 的 sigma 前密后疏，
    #                                                    "信号变化最小"全落在前 8 步，直接废掉
    #   warmup6  cos0.975  跳7/25  381s  PSNR 27.89  ← 跳了第 7 步（构图关键）
    #   warmup8  cos0.98   跳6/25  398s  PSNR 32.37  ← 当前最优（-20% 时间，接近旧 fork 33.5）
    #   warmup9  cos0.972  跳7/25  383s  PSNR 30.82
    # 结论：前 8 步是构图区，必须保护；cos 阈值 0.98 比 0.972 更稳。
    # 2026-10-29 最终标定（0.24 适配层，640×480，CFG 3.0，seed 42，基线 25 步不跳）：
    #   旧 fork 参数 warmup3/cos0.972  跳[7,10,12,14,16,18,20]  353.7s  26.33 dB  ← 0.24 上第 7 步是命门
    #   warmup8/cos0.98/tail3          跳[9,12,14,16,18,20]     398.0s  32.37 dB
    #   warmup8/cos0.975/tail2         跳[9,12,14,16,18,20,22]  335.3s  32.17 dB
    #   + 外推（--skip-extrap）        同上                     334.7s  **33.68 dB** ← 超过旧 fork 的 33.50
    # 跳步位置与旧 fork 一致却差 7 dB 的原因：0.24 的 DiT 实现不同，第 7 步的 pred 复用误差被放大；
    # 外推补上了轨迹曲率，才追平。旧 fork 340s/33.50dB，现在 334.7s/33.68dB。
    "native25": {"label": "25步 · 跳步甜点", "steps": 25, "cfg": 3.0, "step_cache": 0.0,
                 "skip": 0.1, "skip_mode": "adaptive", "skip_cos": 0.975,
                 "skip_warmup": 8, "skip_tail": 2, "skip_extrap": True,
                 "eta640": "~5.6min（640×480，实测 334.7s，跳 7/25）",
                 "desc": "25 步 · 保护前 8 步构图 · 中段跳 7/25 · 外推补轨迹 · PSNR 33.7dB（旧 fork 33.5）",
                 "lora": False},
    # 2026-10-25 实测：20 步轨迹太陡（中段 cos 只有 0.96~0.98，跳一步后掉到 0.92），
    # 自适应跳 5 步 PSNR 只有 24.3 dB，而 25 步同样参数是 33.5 dB。20 步档不开跳步。
    "none": {"label": "20步 · 精细", "steps": 20, "cfg": 3.0, "step_cache": 0.0,
             "skip": 0.0,
             "eta640": "~450s（512²）", "desc": "无蒸馏完整采样 · CFG 3.0 · 20 步轨迹太陡，不跳步",
             "lora": False},
    # 2026-10-29 标定（640×480，CFG 3.0，seed 42）：
    #   50 步不跳        1067.5 s / 100 次前向
    #   50 步跳 15/50     760.3 s /  70 次前向  → PSNR **38.01 dB**（省 28.8%）
    #   跳步位置 [16,18,20,...,44]，前 16 步构图区 + 尾部 4 步全算
    # 38 dB 已属肉眼无损区间（旧 fork 50 步跳 19 步只有 33.43 dB）。
    "full": {"label": "50步 · 官方完整", "steps": 50, "cfg": 3.0, "step_cache": 0.0,
             "skip": 0.1, "skip_mode": "adaptive", "skip_cos": 0.975,
             "skip_warmup": 16, "skip_tail": 4, "skip_extrap": True,
             "eta640": "~12.7min（640×480，实测 760.3s，跳 15/50）",
             "desc": "50 步 · 保护前 16 步 · 中段跳 15/50 · 外推 · PSNR 38.0dB（省 29%）",
             "lora": False},
}
DEFAULT_ENGINE = "mlx8"
DEFAULT_LORA = "akk4" if os.path.exists(AKK4) else "none"   # 没装 4 步蒸馏件时自动回落到 20 步档

OFFICIAL_NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"

# ── 超分 ──
# 两套引擎：
#   spandrel  ESRGAN 系，走 upscale_runner.py + pysafe（独立进程，主进程不碰 torch）
#   seedvr2   mflux 的 1-step 扩散超分，CLI 直调（重绘补结构，吃内存 ~11G）
# 模型表只有一份，在 upscale_registry.py（纯数据，本进程可安全 import）。
UPSCALE_RUNNER = os.path.join(BASE, "upscale_runner.py")
PYSAFE = os.environ.get("MOSS_PYSAFE") or os.path.join(ROOT, "tools", "pysafe")
sys.path.insert(0, BASE)
import upscale_registry as SR  # noqa: E402

UPSCALE_MODELS = SR.MODELS
UPSCALE_GROUPS = SR.GROUPS
DEFAULT_UPSCALE = SR.DEFAULT
SEEDVR2_CLI = shutil.which("mflux-upscale-seedvr2")
# MewZoom 的独立执行器：TrunkNet 架构不在 spandrel 里，必须用官方 mewzoom 库。
# 该库 import torch，所以只能在这个子进程里加载；主进程永不 import 它。
MEWZOOM_RUNNER = os.path.join(BASE, "mewzoom_runner.py")
# Thera：MLX 神经热场，支持任意尺度；代码在 web/thera/，同样独立进程执行
THERA_RUNNER = os.path.join(BASE, "thera_runner.py")


def _upscale_runner(mdef):
    """该超分引擎对应的执行器脚本（spandrel 系走 upscale_runner.py）。"""
    eng = mdef.get("engine", "spandrel")
    if eng == "mewzoom":
        return MEWZOOM_RUNNER
    if eng == "thera":
        return THERA_RUNNER
    return UPSCALE_RUNNER


def _upscale_ok(model_id, mdef):
    """可用性 = 引擎真能跑起来，而不只是权重文件在。

    原来只查 SR.weight_ok()（权重），pysafe 或执行器缺失时 /api/meta 仍报可用，
    用户在 UI 里点了才拿到 500。seedvr2 走 CLI、不经 pysafe，单独判。
    """
    if mdef.get("engine") == "seedvr2":
        return bool(SEEDVR2_CLI)
    if not (os.path.isfile(PYSAFE) and os.path.isfile(_upscale_runner(mdef))):
        return False
    return SR.weight_ok(model_id)


JOBS = {}
JOBS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()
# 排队上限：单任务峰值 13 GB，队列不限的话一口气点十几次就把机器压进 swap。
MAX_QUEUE = int(os.environ.get("MOSS_MAX_QUEUE", "6") or 0)
SYS = {"prev": None, "ring": deque(maxlen=90)}

# ── 生命周期（仅桌面壳拉起时启用）──────────────────────────────────────
# 背景：PyWebView 的 Cmd+Q 会绕过 Python 清理直接杀进程，服务可能残留在后台白占内存。
# 旧的兜底是"连续 MOSS_IDLE_EXIT 秒无请求且无任务就自退"——这个判据在 macOS 上
# 会误杀：窗口切到后台/合盖后 WKWebView 被 App Nap 冻结，前端 2.5s 的 /api/sys
# 轮询随之停摆，10 分钟后服务自己退出，而壳窗口还开着——用户看到的就是
# "UI 卡死、点刷新也没用"（刷新时服务已经没了）。
# 现在改成盯父进程：壳活着服务就活着，壳一退出（含被强杀）服务立刻跟着退。
_PARENT_PID = int(os.environ.get("MOSS_PARENT_PID", "0") or 0)
_IDLE_EXIT = int(os.environ.get("MOSS_IDLE_EXIT", "0") or 0)
_LAST_ACT = [time.time()]


def _touch():
    _LAST_ACT[0] = time.time()


def _idle_watchdog():
    """空闲自退：**只有显式设置 MOSS_IDLE_EXIT>0 才会启动**（main 里的 if 条件）。

    默认 0 = 永不启动。别把它当兜底：macOS 上窗口被 App Nap 冻结时前端轮询会停，
    这个判据会误杀服务。真正负责"壳退了服务跟着退"的是 _parent_watchdog。
    """
    while True:
        time.sleep(20)
        with JOBS_LOCK:
            busy = any(j.get("status") in ("running", "queued") for j in JOBS.values())
        if not busy and time.time() - _LAST_ACT[0] > _IDLE_EXIT:
            print("空闲 %ds 无请求，服务退出" % _IDLE_EXIT, flush=True)
            os._exit(0)


def _parent_watchdog():
    """父进程（桌面壳）消失就退出：等价于"关窗即清理"，但不受 App Nap 影响。"""
    while True:
        time.sleep(5)
        if os.getppid() != _PARENT_PID:
            print("父进程 %d 已退出，服务随之退出" % _PARENT_PID, flush=True)
            os._exit(0)

THUMB_DIR = os.path.join(BASE, ".thumbs")


def _thumb_path(src, width=148):
    """返回缩略图缓存路径（不存在则生成）。失败返回 None，调用方回退原图。"""
    try:
        os.makedirs(THUMB_DIR, exist_ok=True)
        key = hashlib.md5(("%s|%d|%d" % (src, os.path.getmtime(src), width)).encode()).hexdigest()
        dst = os.path.join(THUMB_DIR, key + ".jpg")
        if os.path.isfile(dst):
            return dst
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width * 2), Image.LANCZOS)
            im.save(dst, "JPEG", quality=82, optimize=True)
        return dst
    except Exception:
        return None


CJK = re.compile(r"[\u4e00-\u9fff]+")
FILLER = {"真实照片", "纪实摄影", "照片", "一个", "一位", "画面", "场景", "生成", "摄影"}
TEMPORAL = "年月日时分春夏秋冬个位些这那又再"


def make_topic(prompt):
    """Pick the most descriptive Chinese clause; fall back to a few English words."""
    runs = [r for r in CJK.findall(prompt or "") if len(r) >= 2 and r not in FILLER]
    if runs:
        ranked = sorted(runs, key=len, reverse=True)
        best = ranked[0]
        # A clause that opens on a date/measure word reads badly on its own
        # ("年夏天傍晚..." from "1999年夏天傍晚"): prefer a comparable alternative.
        if best[0] in TEMPORAL:
            for alt in ranked[1:]:
                if len(alt) >= len(best) - 3:
                    best = alt
                    break
        return best[:12]
    words = re.findall(r"[A-Za-z0-9]+", prompt or "")
    return "-".join(words[:3]) if words else "image"


_PATH_LOCK = threading.Lock()
_CLAIMED = set()


def unique_path(topic):
    """在锁内占位，返回 (文件名, 绝对路径)。

    原来的检查是"锁外 + os.path.exists + 文件名精度到秒"：同一秒里提交两个同 topic
    的任务会拿到同一个文件名，互相覆盖。这里用进程内占位集合（而不是先建空文件——
    空 .png 会被历史列表当成品图扫描）。
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    with _PATH_LOCK:
        n = 1
        while True:
            name = topic + "_" + ts + (".png" if n == 1 else "-" + str(n) + ".png")
            path = os.path.join(OUT, name)
            if path not in _CLAIMED and not os.path.exists(path):
                _CLAIMED.add(path)
                return name, path
            n += 1


def _under(root, path):
    """path 是否真在 root 里（先解析符号链接再比较）。

    原来的 `fp.startswith(OUT + os.sep)` 有两个坑：MOSS_OUT 带尾斜杠时前缀永远
    不匹配（所有图片 404），OUT 下的符号链接还能把任意文件读出去。
    """
    root_real = os.path.realpath(root)
    prefix = root_real if root_real == os.sep else root_real + os.sep
    return os.path.realpath(path).startswith(prefix)


def _busy_count():
    """在跑 + 排队的任务数（排队上限用）。"""
    with JOBS_LOCK:
        return sum(1 for j in JOBS.values() if j.get("status") in ("running", "queued"))


def _int_field(p, key, default, lo, hi):
    """取整数字段，非法返回 (None, 中文原因)。

    原来是裸 int(p["width"])：传 "abc" 直接抛 ValueError，连接被掐断，
    前端只看到 SyntaxError。现在统一 400 + 原因。
    """
    v = p.get(key)
    if v in (None, ""):
        return default, None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None, key + " 必须是整数"
    if n < lo or n > hi:
        return None, "%s 必须在 %d~%d 之间" % (key, lo, hi)
    return n, None


def _float_field(p, key, default, lo, hi):
    v = p.get(key)
    if v in (None, ""):
        return default, None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, key + " 必须是数字"
    if f < lo or f > hi:
        return None, "%s 必须在 %g~%g 之间" % (key, lo, hi)
    return f, None


def _read_swap():
    try:
        t = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=4).stdout
        u = re.search(r"used = ([\d.]+)M", t)
        tot = re.search(r"total = ([\d.]+)M", t)
        return (float(u.group(1)) / 1024 if u else 0.0,
                float(tot.group(1)) / 1024 if tot else 0.0)
    except Exception:
        return 0.0, 0.0


def _read_vm():
    """返回 (可用 GB, 解压次数)。

    可用 = free + inactive + speculative + purgeable。只算 Pages free 会低得吓人
    （inactive/purgeable 页随时可回收，本机实测 Pages free 0.2 GB 而实际可用 9 GB）。
    """
    try:
        t = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=4).stdout
        page = 16384
        pm = re.search(r"page size of (\d+) bytes", t)
        if pm:
            page = int(pm.group(1))

        def grab(key):
            mm = re.search(key + r":\s+(\d+)", t)
            return int(mm.group(1)) if mm else 0

        avail = (grab("Pages free") + grab("Pages inactive")
                 + grab("Pages speculative") + grab("Pages purgeable"))
        return avail * page / 1024 ** 3, grab("Decompressions")
    except Exception:
        return 0.0, 0


def _read_total_gb():
    """物理内存总量（hw.memsize），启动时读一次即可。"""
    try:
        t = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, timeout=4).stdout
        return round(int(t.strip()) / 1024 ** 3, 1)
    except Exception:
        return 0.0


TOTAL_MEM_GB = _read_total_gb()


_MEMPRES = {"t": 0.0, "v": None}


def _read_mem_pct():
    """真实可用内存百分比。vm_stat 的 "Pages free" 不含 inactive，会低得吓人
    （本机实测 0.07 GB，实际可用 6.9 GB）；memory_pressure 是 macOS 官方口径。

    这项要起一个子进程、单次 0.3–1s，是整个 /api/sys 里最贵的。前端每 2.5s 轮询
    一次，长期跑下来 fork 开销不小；30 秒内复用上次结果足够用。"""
    now = time.time()
    if _MEMPRES["v"] is not None and now - _MEMPRES["t"] < 30:
        return _MEMPRES["v"]
    try:
        t = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"free percentage:\s*(\d+)", t)
        v = int(m.group(1)) if m else None
    except Exception:
        v = None
    _MEMPRES["t"], _MEMPRES["v"] = now, v
    return v


_SYS_CACHE = {"t": 0.0, "v": None}


def sys_sample():
    now = time.time()
    # 多个前端/重复请求在 1.5s 内共用同一次采样，避免重复 fork vm_stat/sysctl
    cached = _SYS_CACHE["v"]
    if cached is not None and now - _SYS_CACHE["t"] < 1.5:
        return cached
    free_gb, decomp = _read_vm()
    swap_used, swap_total = _read_swap()
    prev = SYS["prev"]
    rate = 0
    if prev:
        dt = max(0.2, now - prev[0])
        rate = int(max(0, decomp - prev[1]) / dt)
    SYS["prev"] = (now, decomp)
    try:
        du = shutil.disk_usage("/")
        disk_free = du.free / 1024 ** 3
    except Exception:
        disk_free = 0.0
    job = None
    pid = None
    with JOBS_LOCK:
        for j in JOBS.values():
            if j.get("status") == "running":
                job, pid = j, j.get("pid")
        queued = sum(1 for j in JOBS.values() if j.get("status") == "queued")
    rss = None
    if pid:
        try:
            o = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=4).stdout.strip()
            rss = int(o) // 1024 if o else None
        except Exception:
            rss = None
    rec = {"t": round(now, 1), "free_gb": round(free_gb, 2),
           "swap_used_gb": round(swap_used, 2), "decomp_per_s": rate}
    SYS["ring"].append(rec)
    out = {
        "free_gb": round(free_gb, 2),
        "mem_avail_gb": round(free_gb, 1),
        "mem_total_gb": TOTAL_MEM_GB,
        "mem_free_pct": _read_mem_pct(),
        "swap_used_gb": round(swap_used, 2),
        "swap_total_gb": round(swap_total, 2),
        "disk_free_gb": round(disk_free, 1),
        "decomp_per_s": rate,
        "job_rss_mb": rss,
        "job_id": job.get("id") if job else None,
        "queue": queued,
        "ring": list(SYS["ring"])[-45:],
    }
    _SYS_CACHE["t"], _SYS_CACHE["v"] = now, out
    return out


# 兼容两种输出：旧 mlx_runner "第 3/25 步完成 (31.1s) [跳步复用]"，
# mflux 0.24 "第 3/25 步 (31.1s)" / "第 5/25 步 [跳] (0.0s)"。
# 0.24 少了"完成"二字、跳步时插了 [跳]，旧正则一条都匹配不到，
# 于是进度条恒为 0、阶段永远显示"文本编码中"（实际早在去噪了）。
RE_MLX = re.compile(r"第\s*(\d+)/(\d+)\s*步(?:完成)?\s*(?:\[[^\]]*\])?\s*\(([\d.]+)s\)")
RE_UPSCALE = re.compile(r"^UPSCALE (\d+)/(\d+) ([\d.]+)$")


def _prune_jobs():
    """已结束的任务只保留最近 50 条。JOBS 原本只增不减，长期运行会一直涨。"""
    with JOBS_LOCK:
        done = [k for k, v in JOBS.items() if v.get("status") in ("done", "error")]
        for k in done[:-50]:
            JOBS.pop(k, None)


# 参数快照（_write_meta/_read_meta）与 PNG 无损重压（_optimize_png/_optimize_async）
# 都在 pngtools.py 里：compress_history.py 要复用它们，但不该 import 本模块。


def _patch(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


ENH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_enhance.py")

# 每个任务的临时目录：提示词 / embeds 这些中间文件以前直接落在输出目录里
# （.prompt-<id>.txt、.embeds-<id>.safetensors 只在 finally 删了 embeds），
# 现在放系统临时目录，任务结束整目录删除。
_TMPDIRS = {}


def _jobdir(p):
    return _TMPDIRS[p["id"]]


def _enhance_prompt(p, job_id, log):
    """用本地 Qwen3.5-9B（与 text encoder 同一份权重）强化提示词。

    成功返回新 prompt；失败返回 None，调用方回退原文。增强在 DiT 加载前完成，
    两者不同时驻留，所以不额外占内存。
    """
    # 中间文件放任务临时目录（run_job 建的），不再往 ~/Pictures/MOSS-Image 里丢
    src = os.path.join(_jobdir(p), "prompt.txt")
    dst = os.path.join(_jobdir(p), "prompt.enhanced.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write(p["prompt"])
    cmd = [PY, ENH, "--prompt-file", src,
           "--width", str(p["width"]), "--height", str(p["height"]), "--out", dst]
    rc = _stream(cmd, dict(os.environ), job_id, log, quiet=True)
    if rc != 0 or not os.path.exists(dst):
        return None
    try:
        with open(dst, encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return None
    return text or None


def build_mlx(p, outfile):
    """mflux 0.24 适配层（web/mflux024_generate.py）：两阶段 encode/run。

    与旧 mlx_runner.py 的参数差异：
      * 没有 --step-cache（0.24 无 StepCache；各档 step_cache 本来就是 0）
      * 长 prompt 走 --prompt-file（Layout JSON 可能几 KB，argv 会超长）
      * 新增 --cache-limit-gb（MLX buffer 池上限；16G 机器不限会把系统挤进 swap）
    """
    env = dict(os.environ)
    base = [MFLUX024_PY, RUNNER,
            # 显式传模型目录：runner 侧的默认值只看 MOSS_MODEL（单数）+ 仓库内路径，
            # MOSS_MODELS 指到别处时两个引擎会读到不同权重。
            "--model", MLX8,
            "--negative", p.get("negative") or "",
            "--steps", str(p["steps"]), "--guidance", str(p["cfg"]),
            "--width", str(p["width"]), "--height", str(p["height"]),
            "--seed", str(p["seed"] if p.get("seed") is not None else 42),
            "--cache-limit-gb", "1", "--out", outfile]
    # 流式加载 DiT（按层 mmap）：常驻 5.5G -> 0.5G，单步约 +1.7%，输出逐像素一致。
    # 前端默认勾选；p 里没有这个键（旧预设/API 调用）时也默认开。
    if p.get("stream", True):
        base += ["--stream"]
    _mode = LORA_MODES.get(p.get("lora_mode"), {})
    _skip = p.get("skip")
    if _skip is None:
        _skip = _mode.get("skip") or 0.0
    if _skip and int(p.get("steps") or 0) >= 10:
        # 跳步只在 >=10 步的档位生效；4/8 步蒸馏档即使前端开了也不跳（轨迹太陡）
        base += ["--skip", str(_skip),
                 "--skip-warmup", str(_mode.get("skip_warmup", 5)),
                 "--skip-tail", str(_mode.get("skip_tail", 3))]
        # 档位里的 skip_cos 必须能生效：前端没传时回退到档位值，再回退到 0.972。
        # （之前只在 p["skip_mode"]=="adaptive" 时才传，导致档位配的 0.975 被 CLI 默认值盖掉）
        _smode = p.get("skip_mode") or _mode.get("skip_mode") or "adaptive"
        if _smode == "uniform":
            # 界面上的「均匀跳步」= 旧 mlx_runner 的 uniform = 0.24 CLI 的 fixed。
            # 这里不翻译的话，uniform 落到 argparse 默认值 adaptive：
            # 用户选「均匀」实际跑的是自适应（静默换语义）。
            base += ["--skip-mode", "fixed"]
        elif _smode == "adaptive":
            _scos = p.get("skip_cos")
            if _scos is None:
                _scos = _mode.get("skip_cos", 0.972)
            base += ["--skip-mode", "adaptive", "--skip-cos", str(_scos)]
        if _mode.get("skip_extrap"):
            # 2026-10-29：跳步时用最近两个真实 pred 半额外推，替代直接复制。
            # 0.24 上实测 32.17 -> 33.68 dB（25 步，跳 7/25，334.7s）。
            base += ["--skip-extrap"]
    loras, scales = [], []
    if p.get("lora_path"):
        loras.append(p["lora_path"])
        scales.append(str(p["lora_scale"]))
    if p.get("char_lora_path"):
        loras.append(p["char_lora_path"])
        scales.append(str(p.get("char_lora_scale")
                          if p.get("char_lora_scale") is not None else 1.0))
    if loras:
        # runner 侧：--lora 逗号分隔；--lora-scale 单值广播或等长列表
        base += ["--lora", ",".join(loras), "--lora-scale", ",".join(scales)]
    if p.get("two_stage"):
        emb = os.path.join(_jobdir(p), "embeds.safetensors")
        enc = [MFLUX024_PY, RUNNER, "--model", MLX8, "--stage", "encode",
               "--prompt-file", _write_prompt_file(p),
               "--negative", p.get("negative") or "",
               "--guidance", str(p["cfg"]), "--embeds", emb,
               "--cache-limit-gb", "1"]
        run = base + ["--stage", "run", "--embeds", emb]
        return [(enc, env, "编码"), (run, env, "去噪")]
    return [(base + ["--stage", "all", "--prompt-file", _write_prompt_file(p)], env, "编码+去噪")]


def build_msmlx(p, outfile):
    """moss-image-mlx 单进程：它自己按 low_memory 释放 TE/DiT/VAE，不需要两阶段。

    turbo 档忽略 steps/cfg（固定 5 student + 2 guided + 8 DPM++），其余档逐字透传，
    所以自由步数也能吃到这条路径。
    """
    env = dict(os.environ)
    cmd = [VENV_PY, MSMLX_RUNNER,
           "--negative", p.get("negative") or "",
           "--steps", str(p["steps"]), "--guidance", str(p["cfg"]),
           "--width", str(p["width"]), "--height", str(p["height"]),
           "--seed", str(p["seed"] if p.get("seed") is not None else 42),
           "--cache-limit-gb", "2",
           "--out", outfile,
           "--prompt-file", _write_prompt_file(p)]
    if not p.get("turbo"):
        loras, scales = [], []
        if p.get("lora_path"):
            loras.append(p["lora_path"])
            scales.append(str(p["lora_scale"]))
        if p.get("char_lora_path"):
            loras.append(p["char_lora_path"])
            scales.append(str(p.get("char_lora_scale")
                              if p.get("char_lora_scale") is not None else 1.0))
        if loras:
            cmd += ["--lora", ",".join(loras), "--lora-scale", ",".join(scales)]
    if p.get("turbo"):
        cmd += ["--turbo"]
    # 跳步：msmlx 引擎默认不启用。2026-10-26 实测 msmlx + adaptive 跳 8/25 步
    # 只有 PSNR 27.98 dB（同条件 mlx8 是 33.50 dB）——msmlx 的采样路径对跳步更敏感，
    # 且它从第 3 步就开始跳。msmlx_runner.py 仍保留 --skip 作为手动实验入口。
    return [(cmd, env, "生成")]


def _write_prompt_file(p):
    """提示词可能很长（Layout JSON），走文件而不是命令行参数，避开 argv 长度与引号坑。"""
    path = os.path.join(_jobdir(p), "prompt.run.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(p["prompt"])
    return path


def _stream(cmd, env, job_id, log, quiet=False):
    """Run one stage, streaming its pty output so the bar and steps are visible.

    quiet=True 给 SeedVR2：它的 CLI 中途不打进度行（只在结束时打一行内存统计），
    所以只把输出留进 log 供排错，不解析、也不覆盖已设好的 phase。

    读循环带空闲超时（MOSS_STREAM_IDLE，默认 1800s）：子进程卡死时不再永久占着
    GPU_LOCK 与任务（原来 os.read 会一直阻塞）。Popen 失败时两个 pty fd 都关。
    """
    import pty
    idle = float(os.environ.get("MOSS_STREAM_IDLE", "1800") or 0)
    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=slave,
                                stderr=slave, env=env)
    except Exception:
        # Popen 抛异常时 slave 还没关（原来只关 master，泄漏两个 fd 之一）
        os.close(slave)
        os.close(master)
        raise
    os.close(slave)
    _patch(job_id, pid=proc.pid)
    buf = b""
    last = time.time()
    timed_out = False
    try:
        while True:
            if idle > 0:
                ready, _, _ = select.select([master], [], [], 5.0)
                if not ready:
                    if time.time() - last > idle:
                        timed_out = True
                        log.append("超时：连续 %g 秒无输出，已终止子进程（MOSS_STREAM_IDLE 可调）"
                                   % idle)
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        break
                    continue
            try:
                chunk = os.read(master, 8192)
            except OSError:
                break
            if not chunk:
                break
            last = time.time()
            buf += chunk
            while True:
                idx = [i for i in (buf.find(b"\r"), buf.find(b"\n")) if i >= 0]
                if not idx:
                    break
                i = min(idx)
                raw, buf = buf[:i], buf[i + 1:]
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                log.append(line[:300])
                if len(log) > 80:
                    log.pop(0)
                if quiet:
                    continue
                mu = RE_UPSCALE.match(line)
                if mu:
                    cur, total, sec = int(mu.group(1)), int(mu.group(2)), float(mu.group(3))
                    _patch(job_id, progress=round(cur / total * 100, 1),
                           step=str(cur) + "/" + str(total), phase="超分",
                           sec_per_step=round(sec / max(cur, 1), 2),
                           eta=round((total - cur) * sec / max(cur, 1)))
                    continue
                m = RE_MLX.search(line)
                if m:
                    cur, total, per = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    # 回灌自开发版（2026-11-02）：实测某次 2000×1120/50 步的大任务，
                    # 全程解析出的 per 都是 0.0（同一份代码、同一套参数的小图却正常），
                    # 前端整场只显示「已用 Ns」，既没有 s/步 也没有剩余时间。
                    # per 不可信时不覆盖上一次有效值，用已知值继续估 ETA；
                    # 一次都没拿到过就不写，让前端走"已用"分支。
                    with JOBS_LOCK:
                        known = (JOBS.get(job_id) or {}).get("sec_per_step") or 0.0
                    use_per = per if per > 0 else known
                    upd = dict(progress=round(cur / total * 100, 1),
                               step=str(cur) + "/" + str(total), phase="去噪")
                    if use_per > 0:
                        upd["sec_per_step"] = use_per
                        upd["eta"] = round((total - cur) * use_per)
                    _patch(job_id, **upd)
                elif "文本编码完成" in line or "embeds" in line:
                    _patch(job_id, phase="编码")
                elif "去噪完成" in line:
                    _patch(job_id, phase="解码")
    finally:
        try:
            os.close(master)
        except Exception:
            pass
    # wait 也加超时：被 kill 的子进程理论上马上退，但不能再无限等
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    return -1 if timed_out else proc.returncode


def run_job(job_id, p):
    outfile = os.path.join(OUT, p["filename"])
    log = []
    # 本任务的中间文件目录（提示词 / embeds），finally 整目录删除
    _TMPDIRS[job_id] = tempfile.mkdtemp(prefix="moss-" + p["id"] + "-")
    with GPU_LOCK:
        # 排队期间被取消的任务不该再起进程
        if JOBS[job_id].get("cancelled"):
            _TMPDIRS.pop(job_id, None)
            with _PATH_LOCK:
                _CLAIMED.discard(outfile)
            return
        _patch(job_id, status="running", started=time.time(), phase="加载")
        try:
            if p.get("enhance"):
                _patch(job_id, phase="提示词强化")
                before = p["prompt"]
                after = _enhance_prompt(p, job_id, log)
                if after:
                    p["prompt"] = after
                    p["prompt_original"] = before
                    _patch(job_id, enhanced=True)
                else:
                    # 增强失败不阻断出图：回退原文，前端会看到 enhance_failed
                    _patch(job_id, enhance_failed=True)
            stages = build_msmlx(p, outfile) if p.get("engine") == "msmlx" else build_mlx(p, outfile)
            for cmd, env, label in stages:
                _patch(job_id, phase=label)
                rc = _stream(cmd, env, job_id, log)
                if rc != 0:
                    if JOBS[job_id].get("cancelled"):
                        return          # 用户手动取消，error 文案已经在 _post_cancel 里写好
                    _patch(job_id, status="error", phase="error",
                           error="\n".join(log[-12:]) or ("stage " + label + " rc=" + str(rc)))
                    return
            if os.path.exists(outfile):
                elapsed = round(time.time() - JOBS[job_id]["started"], 1)
                try:
                    with Image.open(outfile) as im:
                        w, h = im.size
                except Exception:
                    w, h = p["width"], p["height"]
                _patch(job_id, status="done", progress=100.0, phase="done",
                       output=os.path.basename(outfile), width=w, height=h,
                       elapsed=elapsed, size=os.path.getsize(outfile))
                # 参数快照：把这一张的完整配方写进同名 .json
                _write_meta(outfile, {
                    "type": "generate",
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": os.path.basename(outfile),
                    "prompt": p["prompt"],
                    # 开启强化时 prompt 已经是强化后的版本，原文单列一份便于对比/回退
                    "prompt_original": p.get("prompt_original"),
                    "enhanced": bool(p.get("enhance")),
                    "negative": p.get("negative") or "",
                    "engine": p.get("engine"),
                    "lora_mode": p.get("lora_mode"),
                    "lora_scale": p.get("lora_scale"),
                    "char_lora": p.get("char_lora"),
                    "char_lora_scale": p.get("char_lora_scale"),
                    "steps": p.get("steps"),
                    "cfg": p.get("cfg"),
                    "skip": p.get("skip"),
                    "skip_mode": p.get("skip_mode"),
                    "skip_cos": p.get("skip_cos"),
                    # seed=random 时后端当场摇过随机值，这里记的是实际用的那个；
                    # seed_requested 保留用户原始意图（random / 具体数字）
                    "seed": p.get("seed"),
                    "seed_requested": p.get("seed_requested"),
                    "width": w, "height": h,
                    "two_stage": bool(p.get("two_stage")),
                    "stream": bool(p.get("stream")),
                    "elapsed": elapsed,
                    "size": os.path.getsize(outfile),
                })
                # 无损重压放后台线程：8192² 要 70–80s，不该拖慢"完成"状态
                _optimize_async(outfile)
            else:
                _patch(job_id, status="error", error="no output file")
        except Exception as e:
            _patch(job_id, status="error", error=repr(e))
        finally:
            _patch(job_id, pid=None)
            # 整个临时目录一起删（原来只删 embeds，.prompt-*.txt 永久残留在输出目录）
            _tmp = _TMPDIRS.pop(job_id, None)
            if _tmp:
                shutil.rmtree(_tmp, ignore_errors=True)
            with _PATH_LOCK:
                _CLAIMED.discard(outfile)


def run_upscale_job(job_id, p):
    """对已有出图做 x4 超分。两条路线都独立成进程 / 独立 CLI，主进程不碰 torch。"""
    src = os.path.join(UPSCALE_DIR if p.get("sub") == "upscaled" else OUT, p["src"])
    dst = os.path.join(UPSCALE_DIR, p["filename"])
    _weights, mdef = SR.resolve(p["model"])
    seedvr2 = mdef.get("engine") == "seedvr2"
    mewzoom = mdef.get("engine") == "mewzoom"
    thera = mdef.get("engine") == "thera"
    if thera:
        # Thera 走独立 runner（MLX + 神经热场）。默认 4x，registry 的 scale 可改；
        # 它的独有能力是任意实数尺度（3.7x 等），需要时改 --scale 即可。
        # 整图放大时内存随「输出像素」暴涨：640² 源 x4 = 4.9MP 没问题，
        # 2048² 源 x4 = 67MP 会报 Metal Insufficient Memory。
        # 这里按目标像素自动分块，保证每块输出 ≤ 6MP（实测单块 4.9MP 峰值安全）。
        scale = float(mdef.get("scale") or 4)
        tiles = 1
        try:
            with Image.open(src) as im:
                sw, sh = im.size
            target_mp = sw * sh * scale * scale / 1e6
            if target_mp > 6:
                tiles = max(2, min(6, int(math.ceil(math.sqrt(target_mp / 6.0)))))
                print("Thera: 目标 %.1f MP → %dx%d 分块" % (target_mp, tiles, tiles), flush=True)
        except Exception:
            pass
        cmd = [PYSAFE, THERA_RUNNER, "--in", src, "--out", dst,
               "--scale", str(scale), "--tiles", str(tiles)]
    elif mewzoom:
        # 按用户要求：只有选中这个模型时才起子进程；进程内加载 mewzoom 库 + 权重，
        # 跑完 sys.exit 立即退场，不留常驻内存、主进程也从不 import torch。
        cmd = [PYSAFE, MEWZOOM_RUNNER, "--in", src, "--out", dst]
    elif seedvr2:
        # 优先用本地落盘 q8 副本（4.72 GB）而不是 HF 的 fp16 源（7.28 GB）：
        # mflux 的 --model 接受本地路径，目录不在时回退内置名（会触发 HF 下载）。
        mdir = SR.model_dir(p["model"])
        model_arg = mdir if (mdir and os.path.isdir(mdir)) else "seedvr2-3b"
        # 2026-10-21 实测结论（4x 不变）：
        #   实测拆解：transformer 前向仅 70.6s，整单 252s 的**大头是 VAE 解码（约 180s）**。
        #   ✗ 去掉 --low-ram：峰值失控，实测 swap 3.2→6.9GB 后 GPU OOM（Insufficient Memory）。
        #   ✗ --vae-tile-size 768：768² 的块比 512² 大 2.25 倍，VAE 解码阶段直接 OOM。
        #   ✓ 保留 --low-ram（它同时限制 MLX 缓存并防峰值失控），只把缓存上限从默认 1GB 提到 2GB，
        #     给 16384-token 的注意力张量留余量、减少 buffer 反复分配释放的抖动。
        cmd = [SEEDVR2_CLI, "--model", model_arg, "--quantize", "8",
               "--low-ram", "--vae-tiling",
               "--mlx-cache-limit-gb", "2",
               "--image-path", src, "--output", dst,
               "--resolution", "%dx" % int(mdef.get("scale") or 4),
               "--seed", str(p.get("seed") if p.get("seed") is not None else 42)]
    else:
        cmd = [PYSAFE, UPSCALE_RUNNER,
               "--in", src, "--out", dst,
               "--model", p["model"], "--tile", "512", "--overlap", "32"]
    log = []
    with GPU_LOCK:
        if JOBS[job_id].get("cancelled"):     # 排队期间被取消
            return
        _patch(job_id, status="running", started=time.time(),
               phase="重绘" if seedvr2 else "加载")
        try:
            rc = _stream(cmd, dict(os.environ), job_id, log, quiet=seedvr2)
            if rc != 0:
                _patch(job_id, status="error", phase="error",
                       error="\n".join(log[-12:]) or ("rc=" + str(rc)))
                return
            if os.path.exists(dst):
                with Image.open(dst) as im:
                    w, h = im.size
                elapsed = round(time.time() - JOBS[job_id]["started"], 1)
                # 超分产物落在 upscaled/ 子目录；前端拿 output 去拼 /out/<output>，
                # 不带前缀就会 404（历史列表因为另带 sub 字段才没暴露这个问题）
                _patch(job_id, status="done", progress=100.0, phase="done",
                       output="upscaled/" + p["filename"], topic=p["topic"], width=w, height=h,
                       elapsed=elapsed, size=os.path.getsize(dst))
                # 参数快照：记下源图、模型、尺度，方便日后对比哪个模型放大得最好
                _write_meta(dst, {
                    "type": "upscale",
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": p["filename"],
                    "source": p["src"],
                    "source_sub": p.get("sub") or "",
                    "model": p["model"],
                    "model_label": UPSCALE_MODELS.get(p["model"], {}).get("label", p["model"]),
                    "engine": mdef.get("engine", "spandrel"),
                    "scale": int(mdef.get("scale") or 4),
                    "seed": p.get("seed") if p.get("seed") is not None else None,
                    "width": w, "height": h,
                    "elapsed": elapsed,
                    "size": os.path.getsize(dst),
                })
                # 超分图最大，收益也最大（8192² 实测 -15%），同样后台压
                _optimize_async(dst)
            else:
                _patch(job_id, status="error", phase="error", error="no output file")
        except Exception as e:
            _patch(job_id, status="error", phase="error", error=repr(e))
        finally:
            _patch(job_id, pid=None)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, path, ctype, nocache=False):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if nocache:
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        _touch()
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._raw(os.path.join(BASE, "index.html"), "text/html; charset=utf-8", True)
        if u.path == "/api/health":
            # 应用级身份握手：壳/启动器用它判断「7860 上跑的到底是不是工作台」。
            # 之前只判断端口能连，任何占用 7860 的程序都会被当成工作台复用。
            # 同时它也是看护循环「活着且 HTTP 没卡死」的判据。
            return self._json({"app": "moss-image", "pid": os.getpid(),
                               "version": "2026-10-24"})
        if u.path == "/api/meta":
            return self._json({
                "engines": [{"id": k, "label": v["label"], "note": v["note"],
                             "available": v.get("available", True)} for k, v in ENGINES.items()],
                "lora_modes": [{"id": k, "label": v["label"], "steps": v["steps"], "cfg": v["cfg"],
                                "eta640": v["eta640"], "desc": v["desc"],
                                "lora": bool(v.get("lora", True)),
                                "lora_scale": v.get("lora_scale"),
                                "engine": v.get("engine"),
                                "skip": v.get("skip"),
                                "skip_mode": v.get("skip_mode"),
                                "skip_cos": v.get("skip_cos"),
                                "turbo": bool(v.get("turbo"))}
                               for k, v in LORA_MODES.items()],
                "default_engine": DEFAULT_ENGINE,
                "default_lora": DEFAULT_LORA,
                "official_neg": OFFICIAL_NEG,
                "upscale_models": [{"id": k, "label": v["label"], "note": v["note"],
                                    "group": v.get("group", ""),
                                    "engine": v.get("engine", "spandrel"),
                                    # 可用性要连 pysafe/执行器一起查：只查权重时，
                                    # 干净 clone 里四路超分全挂、UI 却仍报可用。
                                    "ok": _upscale_ok(k, v)}
                                   for k, v in UPSCALE_MODELS.items()],
                "upscale_groups": UPSCALE_GROUPS,
                "default_upscale": DEFAULT_UPSCALE,
                "char_loras": _char_loras(),
            })
        if u.path == "/api/sys":
            return self._json(sys_sample())
        if u.path == "/api/job":
            # 原来的 replace("id=", "") 把 "?id=abc&x=1" 整串当 id → 永远 404
            jid = (parse_qs(u.query).get("id") or [""])[0]
            with JOBS_LOCK:
                j = JOBS.get(jid)
            return self._json(j or {"error": "not found"}, 200 if j else 404)
        if u.path == "/api/current":
            # 页面刷新后靠它恢复"正在跑的任务"，否则前端一刷新就丢掉进度
            with JOBS_LOCK:
                active = [dict(j) for j in JOBS.values()
                          if j.get("status") in ("running", "queued")]
            active.sort(key=lambda j: j.get("started") or j.get("created") or 0)
            return self._json({"jobs": active})
        if u.path == "/api/shutdown":
            # 桌面壳关窗时调用：先杀掉还在跑的推理子进程（MLX 进程不杀会残留占内存），
            # 再让主进程干净退出。先回执再退出，避免前端拿到连接中断。
            # 鉴权：必须带本地 token（原来无校验，任意网页可跨站触发）。
            # token 优先从 header 读（不会出现在 ps aux 的 argv 里）；query 形式保留兼容
            tok = (self.headers.get("X-Moss-Token")
                   or parse_qs(u.query).get("token", [""])[0])
            if tok != SHUTDOWN_TOKEN:
                return self._json({"error": "forbidden：shutdown 需要本地 token"}, 403)
            with JOBS_LOCK:
                pids = [j.get("pid") for j in JOBS.values() if j.get("pid")]
            killed = 0
            for p in pids:
                try:
                    os.kill(p, signal.SIGKILL)
                    killed += 1
                except Exception:
                    pass
            self._json({"ok": True, "killed": killed})
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return
        if u.path == "/api/history":
            files = []
            for sub, d in (("", OUT), ("upscaled", UPSCALE_DIR)):
                if not os.path.isdir(d):
                    continue
                for n in os.listdir(d):
                    fp = os.path.join(d, n)
                    if not (n.lower().endswith((".png", ".jpg", ".webp")) and os.path.isfile(fp)):
                        continue
                    st = os.stat(fp)
                    topic = n.rsplit("_", 1)[0] if "_" in n else os.path.splitext(n)[0]
                    files.append({"name": n, "topic": topic, "size": st.st_size,
                                  "mtime": int(st.st_mtime), "sub": sub,
                                  "meta": _read_meta(fp)})
            files.sort(key=lambda f: f["mtime"], reverse=True)
            return self._json({"files": files[:500]})
        if u.path.startswith("/out/"):
            rest = unquote(u.path[5:])
            fp = os.path.normpath(os.path.join(OUT, rest))
            if _under(OUT, fp) and os.path.isfile(fp):
                return self._raw(fp, "image/png")
        if u.path.startswith("/thumb/"):
            # 图库缩略图：原图动辄 0.7 MB（最大 51 MB），直接塞进 148px 的格子
            # 既慢又费内存。这里按需生成 148px JPEG（约 10 KB）并落盘缓存。
            rest = unquote(u.path[7:])
            fp = os.path.normpath(os.path.join(OUT, rest))
            if _under(OUT, fp) and os.path.isfile(fp):
                tp = _thumb_path(fp, 148)
                if tp and os.path.isfile(tp):
                    return self._raw(tp, "image/jpeg")
                return self._raw(fp, "image/png")   # 生成失败就退回原图
        self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/out/"):
            rest = unquote(u.path[len("/api/out/"):])
            fp = os.path.normpath(os.path.join(OUT, rest))
            if _under(OUT, fp) and os.path.isfile(fp):
                # 先清缩略图缓存（它按 路径+mtime 命名，文件删了就再也命中不到）
                try:
                    key = hashlib.md5(("%s|%d|%d" % (fp, os.path.getmtime(fp), 148)).encode()).hexdigest()
                    tp = os.path.join(THUMB_DIR, key + ".jpg")
                    if os.path.isfile(tp):
                        os.remove(tp)
                except Exception:
                    pass
                os.remove(fp)
                return self._json({"deleted": rest})
            return self._json({"error": "not found"}, 404)
        self._json({"error": "not found"}, 404)

    def _write_guard(self):
        """写入类接口（generate / upscale / cancel）的跨站防护。

        /api/generate 与 /api/upscale 能起 13 GB 级任务，原来既不校验 Origin 也不看
        Content-Type：任意网页一个简单 POST（表单 / text-plain）就能触发。
        浏览器对 application/json 的跨站请求必须先发预检，而本服务不实现 OPTIONS，
        所以「Origin 同源 + Content-Type 是 json」合起来就挡住了跨站简单请求。
        返回 None 表示通过，否则返回原因。
        """
        origin = self.headers.get("Origin")
        if origin:
            port = self.server.server_port
            allowed = set()
            for host in ("127.0.0.1", "localhost", "[::1]"):
                for scheme in ("http", "https"):
                    allowed.add("%s://%s:%d" % (scheme, host, port))
            if origin not in allowed:
                return "forbidden：Origin 不是本机工作台"
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return "forbidden：Content-Type 必须是 application/json"
        return None

    def _read_json(self):
        """读请求体，返回 (dict, 错误文案)。"""
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            n = 0
        try:
            return json.loads(self.rfile.read(n) or b"{}"), None
        except Exception:
            return None, "bad json"

    def _post_cancel(self):
        """取消一个任务：kill 记录的子进程并把任务置成 error。

        原来没有任何取消入口，_stream 也没有超时，卡住的任务只能把整个服务杀掉。
        """
        err = self._write_guard()
        if err:
            return self._json({"error": err}, 403)
        p, bad = self._read_json()
        if bad:
            return self._json({"error": bad}, 400)
        jid = (p or {}).get("id") or ""
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if not j:
                return self._json({"error": "not found"}, 404)
            pid = j.get("pid")
            j.update(status="error", phase="error", pid=None, cancelled=True,
                     error="已取消（用户手动停止）")
        killed = False
        if pid:
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed = True
            except Exception:
                pass
        return self._json({"ok": True, "id": jid, "killed": killed})

    def _post_upscale(self):
        err = self._write_guard()
        if err:
            return self._json({"error": err}, 403)
        if MAX_QUEUE > 0 and _busy_count() >= MAX_QUEUE:
            return self._json({"error": "队列已满（%d 个任务在跑或排队），等一会儿再提交"
                                        % MAX_QUEUE}, 429)
        p, bad = self._read_json()
        if bad:
            return self._json({"error": bad}, 400)
        sub = p.get("sub") or ""
        srcdir = UPSCALE_DIR if sub == "upscaled" else OUT
        src = os.path.basename(p.get("name") or "")
        if not src or not os.path.isfile(os.path.join(srcdir, src)):
            return self._json({"error": "找不到源图"}, 400)
        model = p.get("model") or DEFAULT_UPSCALE
        if model not in UPSCALE_MODELS:
            return self._json({"error": "unknown model"}, 400)
        weights, mdef = SR.resolve(model)
        if mdef.get("engine") == "seedvr2":
            if not SEEDVR2_CLI:
                return self._json({"error": "SeedVR2 未就绪（PATH 里找不到 mflux-upscale-seedvr2）"}, 500)
        else:
            # 三个走 pysafe 的引擎各有自己的 runner；原来一律查 upscale_runner.py，
            # mewzoom/thera 缺执行器时会以一句看不懂的报错失败。
            runner = _upscale_runner(mdef)
            if not (os.path.isfile(runner) and os.path.isfile(PYSAFE)):
                return self._json({"error": "超分执行器未就绪（缺 %s 或 pysafe）"
                                            % os.path.basename(runner)}, 500)
            if not (weights and os.path.isfile(weights)):
                return self._json({"error": "权重缺失：" + str(weights)}, 500)

        base = os.path.splitext(src)[0]
        topic = base.rsplit("_", 1)[0] if "_" in base else base
        tag = UPSCALE_MODELS[model]["label"].replace(" ", "")
        filename = "%s_超分%s_%s.png" % (topic[:28], tag, time.strftime("%H%M%S"))
        jid = uuid.uuid4().hex[:12]
        _prune_jobs()
        with JOBS_LOCK:
            JOBS[jid] = {"id": jid, "status": "queued", "progress": 0.0, "phase": "排队",
                         "topic": topic, "filename": filename, "width": 0, "height": 0,
                         "pid": None}
        threading.Thread(target=run_upscale_job,
                         args=(jid, {"src": src, "filename": filename, "sub": sub,
                                     "topic": topic, "model": model,
                                     "seed": p.get("seed")}),
                         daemon=True).start()
        return self._json({"id": jid, "filename": filename})

    def do_POST(self):
        _touch()
        u = urlparse(self.path)
        if u.path == "/api/cancel":
            return self._post_cancel()
        if u.path == "/api/upscale":
            return self._post_upscale()
        if u.path != "/api/generate":
            return self._json({"error": "not found"}, 404)
        err = self._write_guard()
        if err:
            return self._json({"error": err}, 403)
        if MAX_QUEUE > 0 and _busy_count() >= MAX_QUEUE:
            return self._json({"error": "队列已满（%d 个任务在跑或排队），等一会儿再提交"
                                        % MAX_QUEUE}, 429)
        p, bad = self._read_json()
        if bad:
            return self._json({"error": bad}, 400)
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            return self._json({"error": "prompt empty"}, 400)
        # 裸 int() 遇到 "abc" 会抛 ValueError、连接被掐断，前端只看到 SyntaxError
        w, err = _int_field(p, "width", 640, 64, 4096)
        if err:
            return self._json({"error": err}, 400)
        h, err = _int_field(p, "height", 480, 64, 4096)
        if err:
            return self._json({"error": err}, 400)
        if w % 16 or h % 16:
            return self._json({"error": "width/height 必须是 16 的倍数"}, 400)
        engine = p.get("engine") or DEFAULT_ENGINE
        lora_mode = p.get("lora_mode") or DEFAULT_LORA
        if lora_mode not in LORA_MODES:
            return self._json({"error": "unknown lora_mode"}, 400)
        # 档位可以绑定引擎：turbo 的调度只存在于 moss-image-mlx 里，必须强制走它
        bound = LORA_MODES[lora_mode].get("engine")
        if bound:
            engine = bound
        if engine not in ENGINES:
            return self._json({"error": "unknown engine"}, 400)
        if not ENGINES[engine].get("available", True):
            return self._json({"error": ENGINES[engine]["note"]}, 400)
        # 权重预检：tier 目录不在时与其等 runner 崩在加载阶段，不如当场 400 并给出修复提示
        # （MOSS_MODELS 指错是最常见的原因）
        if engine == "mlx8" and not os.path.isdir(os.path.join(MLX8, "transformer")):
            return self._json({"error": "mlx8 引擎缺权重：%s 不存在。跑 scripts/download_models.py，"
                                        "或用 MOSS_MODELS 指向模型目录"
                                        % os.path.join(MLX8, "transformer")}, 400)
        # steps/cfg：用户显式传了就用用户的，否则用档位默认。不再静默覆盖。
        mode = LORA_MODES[lora_mode]
        # 步数可被前端显式覆盖（工作台支持自由调节）
        steps, err = _int_field(p, "steps", mode["steps"], 1, 200)
        if err:
            return self._json({"error": err}, 400)
        cfg, err = _float_field(p, "cfg", mode["cfg"], 0.0, 20.0)
        if err:
            return self._json({"error": err}, 400)
        # 跳步：auto/缺省 = 跟随档位；显式数值 = 用户覆盖。夹到 0~0.6，
        # 且只有 steps>=10 才在 runner 侧生效（4/8 步蒸馏档传了也不跳）。
        _skip_in = p.get("skip")
        skip_mode, skip_cos = "uniform", 0.972
        if _skip_in in (None, "", "auto"):
            skip = float(mode.get("skip") or 0.0)
            skip_mode = mode.get("skip_mode", "uniform")
            skip_cos = float(mode.get("skip_cos") or 0.972)
        elif _skip_in == "adaptive":
            skip, skip_mode = 0.1, "adaptive"
            skip_cos = float(mode.get("skip_cos") or 0.972)
        else:
            try:
                skip = max(0.0, min(0.6, float(_skip_in)))
            except (TypeError, ValueError):
                skip = float(mode.get("skip") or 0.0)
            # 显式比例 = 均匀/固定跳步。这里必须写死 uniform：build_mlx 会把它翻译成
            # CLI 的 --skip-mode fixed，不写就会落到 argparse 默认的 adaptive。
            skip_mode = "uniform"
        # seed=random 时当场摇一个真随机值：以前是 None 一路兜底成 42，
        # 导致同 prompt 连拍全是同一张、快照里也全是 42。
        # 摇完写进 job，快照记录实际用的种子（拿它就能复现），
        # seed_requested 保留 random 标记用于区分"用户指定"和"系统摇的"。
        seed_in = p.get("seed")
        seed_random = seed_in in (None, "", "random")
        if seed_random:
            seed = random.randint(1, 2 ** 31 - 1)
        else:
            seed, err = _int_field(p, "seed", 42, 0, 2 ** 63 - 1)
            if err:
                return self._json({"error": err}, 400)
        topic = make_topic(prompt)
        filename, outpath = unique_path(topic)
        jid = uuid.uuid4().hex[:12]

        lora_path = {"akk4": AKK4}.get(lora_mode)
        # 原生档不挂蒸馏件，强度记为 None：快照和灯箱才不会把界面上残留的 0.7 当真
        lora_scale = None
        if lora_path:
            # 0 是合法值（等于关掉 LoRA），不能用 or 兜底，否则 0 会被覆盖成默认强度
            _lv = p.get("lora_scale")
            lora_scale = float(_lv) if _lv is not None and _lv != "" else 1.0
        # 角色 LoRA（叠加在蒸馏档之上）：只认 models/loras/ 里的文件，防路径穿越
        char_lora = (p.get("char_lora") or "").strip()
        char_lora_path = None
        char_lora_scale = None
        if char_lora:
            cand = os.path.normpath(os.path.join(LORA_DIR, os.path.basename(char_lora)))
            if not (os.path.isfile(cand) and cand.startswith(LORA_DIR + os.sep)):
                return self._json({"error": "未知的角色 LoRA：" + char_lora}, 400)
            # 两条引擎都支持多 LoRA；turbo 档的调度来自 student 适配器，不叠角色
            if engine not in ("mlx8", "msmlx") or LORA_MODES[lora_mode].get("turbo"):
                return self._json(
                    {"error": "当前引擎/档位暂不支持叠加角色 LoRA"}, 400)
            char_lora_path = cand
            try:
                _cs = p.get("char_lora_scale")
                char_lora_scale = float(_cs) if _cs not in (None, "") else 1.0
            except (TypeError, ValueError):
                char_lora_scale = 1.0
        job = {
            "id": jid, "status": "queued", "progress": 0.0, "phase": "排队",
            "prompt": prompt, "topic": topic, "width": w, "height": h,
            "steps": steps, "cfg": cfg, "seed": seed,
            "skip": skip, "skip_mode": skip_mode, "skip_cos": skip_cos,
            "seed_requested": "random" if seed_random else seed,
            "filename": filename,
            "engine": engine, "lora_mode": lora_mode, "lora_path": lora_path,
            "lora_scale": lora_scale,
            "char_lora": char_lora or None,
            "char_lora_path": char_lora_path,
            "char_lora_scale": char_lora_scale,
            "negative": (p.get("negative") or "").strip(),
            # 两阶段是引擎固有行为，不接受外部开关：
            #   mlx8  → 永远两段（编码/去噪分进程，避免 13G 同时驻留）
            #   msmlx → 永远单进程（它内部自己做 TE/DiT/VAE 分阶段释放）
            # 2026-10-24 事故：复用 msmlx 任务的历史参数把 mlx8 也切成单进程，
            # 峰值 13G、swap 14.7G、所有任务慢 2-3 倍。所以这里不再读前端值。
            "two_stage": engine == "mlx8",
            # DiT 按层流式加载：只有 mlx8（build_mlx → --stream）有这条路径，
            # msmlx 自己分阶段释放 TE/DiT/VAE。记进 job 和 .json 快照，
            # 否则事后翻快照看不出这张图当时到底是不是流式跑的。
            "stream": bool(p.get("stream", True)) if engine == "mlx8" else False,
            "turbo": bool(LORA_MODES[lora_mode].get("turbo")),
            # 提示词强化（本地 Qwen3.5-9B，在锁内、DiT 加载前跑）
            "enhance": bool(p.get("enhance")),
            "created": time.time(),
        }
        _prune_jobs()
        with JOBS_LOCK:
            JOBS[jid] = job
        threading.Thread(target=run_job, args=(jid, job), daemon=True).start()
        return self._json({"id": jid, "filename": filename, "topic": topic,
                           "steps": steps, "cfg": cfg})


if __name__ == "__main__":
    # token 只在真正跑服务时落盘：import app 的脚本（compress_history.py）不该产生副作用
    _write_token_file()
    # 端口统一：MOSS_PORT 优先（.command / 壳都读它），其次兼容旧的 PORT
    port = int(os.environ.get("MOSS_PORT") or os.environ.get("PORT") or "7860")
    print("MOSS-Image workbench -> http://127.0.0.1:" + str(port), flush=True)
    if _PARENT_PID > 0:
        # 桌面壳模式：壳活着服务就活着；空闲计时会误杀被 App Nap 冻结的前端
        print("父进程看护已启用（parent pid %d）" % _PARENT_PID, flush=True)
        threading.Thread(target=_parent_watchdog, daemon=True).start()
    elif _IDLE_EXIT > 0:
        print("空闲自退已启用：连续 %d 秒无请求且无任务即退出" % _IDLE_EXIT, flush=True)
        threading.Thread(target=_idle_watchdog, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
