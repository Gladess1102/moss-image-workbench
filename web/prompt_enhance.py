#!/usr/bin/env python3
"""MOSS-Image 提示词强化 CLI（纯本地，复用现有 Qwen3.5-9B MLX 权重）。

用法：
  python3 prompt_enhance.py --prompt-file in.json --out enhanced.txt
  python3 prompt_enhance.py --prompt "手写提示词" --mode caption

输入支持两种：
  * 纯文本 caption        -> 输出增强后的 caption 文本
  * 已编译的 Layout JSON  -> 输出同结构 JSON（label/bbox 原样保留，只增强 caption 文本）

设计原则（照抄官方 skill）：
  1. 保留用户已明确的一切：主体、数量、关系、媒介、风格、排除项、可见文字原文
  2. 只补充"可见的具体选择"（镜头/材质/光线/色调），不新增未提的主体
  3. 不输出解释、不加质量套话
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("MOSS_MODELS") or os.path.join(ROOT, "models")
# 提示词强化用 mlx_lm 直接加载一份 Qwen3.5-9B（mflux 布局的 4bit 权重）。
# 量化 tier（moss-image-mlx 的 affine6g64）mlx_lm 不认，所以这里单独指：
#   MOSS_TE=<mflux 格式 TE 目录>，默认 models/mlx_te_qwen35。
# 没有这份权重时强化会失败并自动回退原文，不影响出图。
TE = os.environ.get("MOSS_TE") or os.path.join(MODELS, "mlx_te_qwen35")

SYSTEM = """你是 MOSS-Image 的提示词工程师。MOSS-Image 是支持中英文字渲染与 JSON 版式控制的文生图模型。

你只做一件事：把用户给的提示词改写得更具体、更有画面感，但**绝不改变用户的意图**。

硬性要求：
- 保留所有已明确的内容：主体、数量、动作、空间关系、媒介（照片/插画/3D…）、风格、色调、排除项、可见文字
- 引号内的中文/英文文字必须逐字保留，包括标点与换行
- 只补充"看得见的选择"：镜头焦段、构图、材质、光线方向、色彩、质感、情绪
- 不新增用户没提的主体、品牌、人物、文字
- 不加"8k/best quality/masterpiece"这类空话
- 不写解释、不写分析、不写"以下是增强后的提示词"
"""

CAPTION_TASK = """把下面的提示词改写为一段更具体的 MOSS-Image caption（中文，连贯散文，可见文字用引号原样保留）。
只输出 caption 正文，不要 JSON，不要任何说明。

原提示词：
{prompt}
"""

LAYOUT_TASK = """下面是一个已经编译好的 MOSS-Image Layout 提示词，画布 {w}x{h}，坐标是 abs_centralized/yxyx（相对画布中心）。

任务：增强每个 item 的 caption 与 summary.caption 的文字描写，**其余一切保持不变**：
- 每个 item 的 label 和 bbox 数组：一个数字都不能改
- items 数量、顺序不变
- bbox_coord_space 必须是 "abs_centralized"，bbox_format 必须是 "yxyx"
- 顶层不要加 width/height
- 保留 summary.canvas
- 引号内的可见文字逐字保留

只输出一个 JSON 对象，不要代码块标记，不要解释。

原始 JSON：
{payload}
"""


def load_model():
    from mlx_lm import load
    t0 = time.time()
    model, tok = load(TE)
    print(f"[enhance] 模型加载 {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return model, tok


def generate(model, tok, messages, max_tokens=1400):
    from mlx_lm import generate as mlx_generate
    # enable_thinking 必须显式关掉：开着时 3.77 tok/s 且 token 全烧在思考链（实测）
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)


def strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def detect(prompt: str, width: int, height: int):
    """返回 (kind, payload)。kind in {caption, layout}"""
    s = prompt.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except Exception:
            return "caption", prompt
        # 工作台的 .json 快照是整条生成记录，真正的提示词在 obj["prompt"]
        if isinstance(obj, dict) and "prompt" in obj and "items" not in obj:
            return detect(obj["prompt"], width, height)
        if isinstance(obj, dict) and "items" in obj:
            return "layout", obj
        if isinstance(obj, dict) and "caption" in obj:
            return "caption", obj["caption"]
    return "caption", prompt


def enhance_caption(model, tok, prompt):
    out = generate(model, tok, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": CAPTION_TASK.format(prompt=prompt)},
    ], max_tokens=700)
    return strip_fence(out)


def enhance_layout(model, tok, obj, width, height):
    payload = json.dumps(obj, ensure_ascii=False, indent=1)
    raw = generate(model, tok, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": LAYOUT_TASK.format(w=width, h=height, payload=payload)},
    ], max_tokens=2000)
    txt = strip_fence(raw)
    try:
        new = json.loads(txt)
    except Exception as e:
        print(f"[enhance] Layout 输出不是合法 JSON: {e}", file=sys.stderr)
        print(txt[:600], file=sys.stderr)
        return None
    # 结构守护：label/bbox 锁死，坐标不许动
    old_items = obj.get("items", [])
    new_items = new.get("items", [])
    if len(old_items) != len(new_items):
        print(f"[enhance] items 数量变了 {len(old_items)} -> {len(new_items)}，拒绝", file=sys.stderr)
        return None
    for a, b in zip(old_items, new_items):
        if a.get("label") != b.get("label") or a.get("bbox") != b.get("bbox"):
            print(f"[enhance] label/bbox 被改动（{a.get('label')}），拒绝", file=sys.stderr)
            return None
    new["bbox_coord_space"] = "abs_centralized"
    new["bbox_format"] = "yxyx"
    new.pop("width", None)
    new.pop("height", None)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--mode", choices=["auto", "caption", "layout"], default="auto")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    prompt = a.prompt
    if a.prompt_file:
        with open(a.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    if not prompt.strip():
        ap.error("需要 --prompt 或 --prompt-file")

    kind, payload = detect(prompt, a.width, a.height)
    if a.mode != "auto":
        kind = a.mode
        if kind == "layout" and not isinstance(payload, dict):
            ap.error("mode=layout 需要 JSON 输入")

    model, tok = load_model()
    t0 = time.time()
    if kind == "layout":
        res = enhance_layout(model, tok, payload, a.width, a.height)
        if res is None:
            sys.exit(2)
        text = json.dumps(res, ensure_ascii=False, separators=(",", ":"))
    else:
        text = enhance_caption(model, tok, payload if isinstance(payload, str) else str(payload))
    print(f"[enhance] {kind} 用时 {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
