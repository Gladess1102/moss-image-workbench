#!/usr/bin/env python3
"""超分模型注册表 —— 单一数据源（2026-10-21 终版，5 模型）。

本模块必须是**纯数据**：app.py 主进程会 import 它，而主进程绝不能沾
torch / mlx（macOS 下 torch 退出期会崩在 MPS 的 MetalShaderLibrary 析构，
所以真正推理一律走独立进程 + pysafe）。

===================== 2026-10-21 终选（同图 8+1 模型实测后定稿）=====================
赛道        模型          引擎       512→2048   锐度   色偏     定位
写实·保真   nomos2_hq     spandrel   131.0s     836    1.76°   保真冠军，不脑补
通用·日常   mewzoom       mewzoom     7.4s      1594   9.23°   最快，锐度第二，零内存
通用·日常   thera_air     thera      10.5s      975    1.00°   最忠实 + **任意尺度**
动漫·线条   anime6b       spandrel    4.4s      1672   10.14°  动漫专用，最快
生成式·重绘 seedvr2       seedvr2    251.8s     1090   8.94°   重绘补结构（会改内容）

淘汰记录（同批实测数据）：
  hat_s        196.7s / 864 / 0.87°  —— 保真最强但比 nomos2 慢 66s，用户选性价比
  realwebphoto 118.7s / 818 / 3.81°  —— 三项全输给 nomos2_hq
  remacri       9.0s / 1085 / 4.33°  —— 被 mewzoom（更快更锐）与 thera（更忠实）夹杀
  x4plus        9.0s / 1984 / 11.24° —— 最锐但锐化白边严重、色偏最大
  HAT-L / Real-HAT-GAN —— 512 尺寸 OOM，已删（省 320MB）

权重一律放在 web/models/（工作台自包含）；upscale/models/legacy 只放淘汰件。
"""

# 权重根目录 = moss-image/（web/ 的上一级）
import os

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WEB_DIR)

# 前端分组顺序（下拉框按此顺序出 optgroup）
GROUPS = ["写实 · 保真", "通用 · 日常", "动漫 · 线条", "生成式 · 重绘"]

MODELS = {
    "nomos2_hq": {
        "label": "Nomos2 HQ",
        "group": "写实 · 保真",
        "engine": "spandrel",
        "rel": "web/models/4xNomos2_hq_atd.pth",
        "scale": 4,
        "note": "保真冠军 · 不脑补 · ~131s · ATD 吃内存",
    },
    "mewzoom": {
        "label": "MewZoom 4X",
        "group": "通用 · 日常",
        "engine": "mewzoom",
        "rel": "web/models/mewzoom_v1_4x.safetensors",
        "scale": 4,
        "note": "21M 参数 · 最快 7.4s · 内存零压力",
    },
    "thera_air": {
        "label": "Thera Air",
        "group": "通用 · 日常",
        "engine": "thera",
        "rel": "web/models/weights-air.npz",
        "scale": 4,
        "note": "任意尺度(3.7x等) · 最忠实 1.00° · 10.5s",
    },
    "anime6b": {
        "label": "anime 6B",
        "group": "动漫 · 线条",
        "engine": "spandrel",
        "rel": "web/models/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "note": "动漫专用 · 17.9MB · 最快 4.4s",
    },
    "seedvr2": {
        "label": "SeedVR2 3B",
        "group": "生成式 · 重绘",
        "engine": "seedvr2",
        "rel": None,          # 不经 spandrel 加载，故无单文件权重路径
        # 本地落盘 q8 副本（4.72 GB）比 HF fp16 源（7.28 GB）小，实测只慢 10s。
        # mflux 的 --model 接受本地路径；目录不在时 app.py 回退内置名 seedvr2-3b。
        "model_dir": "upscale/models/seedvr2-3b-q8",
        "scale": 4,
        "note": "重绘补结构 · 不改文字 · ~252s · 峰值 11G",
    },
}

DEFAULT = "mewzoom"   # remacri 已淘汰，默认改为最快且均衡的 mewzoom


def resolve(model_id):
    """返回 (权重绝对路径 or None, 模型定义)。未知 id 抛 KeyError。"""
    m = MODELS[model_id]
    rel = m.get("rel")
    return (os.path.join(ROOT_DIR, rel) if rel else None), m


def model_dir(model_id):
    """该模型的本地落盘目录（无则 None）。"""
    rel = MODELS[model_id].get("model_dir")
    return os.path.join(ROOT_DIR, rel) if rel else None


def weight_ok(model_id):
    p, _ = resolve(model_id)
    return p is None or os.path.isfile(p)
