#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MOSS-Image MCP server —— stdio 传输、零第三方依赖（只用标准库）。

== 传输约定（别搞错）==
MCP 的 stdio 传输是「换行分隔的 JSON-RPC 2.0」：一行一条完整 JSON 消息。
不是 LSP 的 Content-Length: N + 空行 + body。所以本进程：
  · stdin  逐行读，每行一个 JSON-RPC 请求 / 通知；
  · stdout 只写 JSON-RPC 响应（一行一条，写完立刻 flush）；
  · 日志   一律写 MOSS_MCP_LOG（默认 ~/.moss-image/logs/mcp.log），绝不进 stdout；
  · 启动后 sys.stdout 被指向 stderr，任何漏网的 print 也污染不了协议流。

== 启动 ==
    python3 agent/mcp_server.py
MCP 客户端配置（Claude Desktop / Cursor / 自研 client 都一样）：
    {"mcpServers": {"moss-image": {
        "command": "python3",
        "args": ["/path/to/moss-image-workbench/agent/mcp_server.py"]}}}

也可以让 npm 包代劳（推荐，客户端配置里不用写 Python 路径）：
    {"mcpServers": {"moss-image": {
        "command": "npx", "args": ["-y", "moss-image-workbench", "mcp"]}}}

== 环境变量 ==
    MOSS_MCP_LOG         日志文件（默认 ~/.moss-image/logs/mcp.log）
    MOSS_AGENT_URL       强制指定后端地址（默认自动发现，必要时自动拉起）
    MOSS_AGENT_AUTOSTART 0 = 不自动拉起后端
    MOSS_OUT             产物目录（默认 ~/Pictures/MOSS-Image）

== 工具（11 个）==
    moss_capabilities     能力发现（先调这个）
    moss_backend_status   后端健康 + 当前任务
    moss_generate         同步生成一张（preset 选意图；同 seed 幂等命中）
    moss_generate_batch   同 prompt 多变体
    moss_upscale          x4 超分
    moss_list_outputs     列最近产物
    moss_inspect          读参数快照（无损复现的依据）
    moss_replay           按快照原样重跑并比对 sha256
    moss_validate_prompt  纯本地提示词校验（不烧 GPU）
    moss_build_layout     生成官方四键 Layout JSON
    moss_cancel           取消排队/运行中的任务

协议层实现：initialize（三版协议协商）/ notifications/initialized / ping /
tools/list / tools/call；未知方法 -32601，未知工具 -32602，坏 JSON -32700。
业务错误（moss_client.MossError）统一变成 isError:true 的 content 文本，
文本是 JSON，含 code / message / hint（E_JOB_TIMEOUT 还带 job_id）。
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import pathlib
import signal
import sys
import threading
import time
import traceback
import urllib.error

SERVER_NAME = "moss-image"
SERVER_TITLE = "MOSS-Image 端侧生图"
SERVER_VERSION = "1.0.0"

# 支持的 MCP 协议版本（新 -> 旧）；客户端要哪个且支持就回显哪个
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

DEFAULT_LOG_PATH = os.path.expanduser("~/.moss-image/logs/mcp.log")
MAX_LOG_BYTES = 8 << 20
MAX_CONCURRENT_TOOLS = 8

# ── 客户端（唯一依赖，本目录的零依赖模块）──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import moss_client as mc
    _IMPORT_ERROR = None
except Exception as _e:                       # 客户端缺失/损坏时给出可读原因
    mc = None
    _IMPORT_ERROR = "%s: %s" % (type(_e).__name__, _e)

_LOG_PATH = os.environ.get("MOSS_MCP_LOG") or DEFAULT_LOG_PATH
_OUT = None                                   # 真 stdout（bytes 流），main() 里固定下来
_WRITE_LOCK = threading.Lock()
_WORKERS = set()
_TRACK_LOCK = threading.Lock()
_SEM = threading.BoundedSemaphore(MAX_CONCURRENT_TOOLS)


# ── 日志（永远不写 stdout）────────────────────────────────────────────────
def _log(event, **fields):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "event": event}
    for k, v in fields.items():
        if v is not None:
            rec[k] = v
    try:
        line = json.dumps(rec, ensure_ascii=False, default=str)
    except Exception:
        line = json.dumps({"ts": rec["ts"], "event": event, "error": "unserializable"})
    try:
        d = os.path.dirname(_LOG_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        with contextlib.suppress(Exception):
            sys.stderr.write("[moss-mcp] " + line + "\n")
            sys.stderr.flush()


def _rotate_log():
    """日志超过 8MB 就挪成 .1（只在启动时做，不打断服务）。"""
    try:
        if os.path.getsize(_LOG_PATH) > MAX_LOG_BYTES:
            os.replace(_LOG_PATH, _LOG_PATH + ".1")
    except OSError:
        pass


# ── 参数校验（在碰 moss_client 之前就地拦下，错误码统一 E_BAD_ARGS）──────
class ArgsError(Exception):
    """工具参数不合法（不是业务失败，是调用方式不对）。"""

    def __init__(self, message, code="E_BAD_ARGS", hint="", detail=None):
        self.code = code
        self.message = message
        self.hint = hint
        self.detail = detail
        super().__init__(message)

    def to_dict(self):
        d = {"ok": False, "code": self.code, "message": self.message}
        if self.hint:
            d["hint"] = self.hint
        if self.detail is not None:
            d["detail"] = self.detail
        return d


_SCHEMA_HINT = "参数名/类型见 tools/list 里该工具的 inputSchema"

# 这几个枚举在 moss_client 里各有专属错误码（见 agent/contract.json），
# 所以本地校验时就回契约里的码，agent 才能按 code 分支。
PRESET_IDS = ("draft", "standard", "quality", "precise")
LORA_MODE_IDS = ("akk4", "turbo", "native25", "none", "full")
ENGINE_IDS = ("mlx8", "msmlx")


def _as_bool(v, key, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
    raise ArgsError("%s 必须是布尔值（true/false），当前 %r" % (key, v), hint=_SCHEMA_HINT)


def _as_int(v, key, lo=None, hi=None, default=None, required=False):
    if v is None or v == "":
        if required:
            raise ArgsError("缺少必填参数 %s" % key, hint=_SCHEMA_HINT)
        return default
    if isinstance(v, bool):
        raise ArgsError("%s 必须是整数，当前 %r" % (key, v), hint=_SCHEMA_HINT)
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ArgsError("%s 必须是整数，当前 %r" % (key, v), hint=_SCHEMA_HINT)
    if lo is not None and n < lo:
        raise ArgsError("%s 不能小于 %s（当前 %s）" % (key, lo, n), hint=_SCHEMA_HINT)
    if hi is not None and n > hi:
        raise ArgsError("%s 不能大于 %s（当前 %s）" % (key, hi, n), hint=_SCHEMA_HINT)
    return n


def _as_num(v, key, lo=None, hi=None, default=None, required=False):
    if v is None or v == "":
        if required:
            raise ArgsError("缺少必填参数 %s" % key, hint=_SCHEMA_HINT)
        return default
    if isinstance(v, bool):
        raise ArgsError("%s 必须是数字，当前 %r" % (key, v), hint=_SCHEMA_HINT)
    try:
        n = float(v)
    except (TypeError, ValueError):
        raise ArgsError("%s 必须是数字，当前 %r" % (key, v), hint=_SCHEMA_HINT)
    if lo is not None and n < lo:
        raise ArgsError("%s 不能小于 %s（当前 %s）" % (key, lo, n), hint=_SCHEMA_HINT)
    if hi is not None and n > hi:
        raise ArgsError("%s 不能大于 %s（当前 %s）" % (key, hi, n), hint=_SCHEMA_HINT)
    return n


def _as_str(v, key, default=None, required=False, allow_empty=False,
            choices=None, maxlen=None, strict=False):
    if v is None:
        if required:
            raise ArgsError("缺少必填参数 %s" % key, hint=_SCHEMA_HINT)
        return default
    if not isinstance(v, str):
        if strict:
            # 提示词这类"内容"参数不做隐式转换：数字 123 变成提示词 "123" 会静默出一张废图
            raise ArgsError("%s 必须是字符串，当前是 %s（%r）"
                            % (key, type(v).__name__, v),
                            hint="把内容写成字符串；数字/对象不是合法提示词")
        v = str(v)
    if not allow_empty and not v.strip():
        if required:
            raise ArgsError("%s 不能为空" % key, hint=_SCHEMA_HINT)
        return default
    if choices is not None and v not in choices:
        raise ArgsError("%s 必须是 %s 之一，当前 %r" % (key, " / ".join(choices), v),
                        hint=_SCHEMA_HINT)
    if maxlen is not None and len(v) > maxlen:
        raise ArgsError("%s 太长（%d 字符，上限 %d）" % (key, len(v), maxlen),
                        hint="长提示词先落成文件再读，或拆短")
    return v


def _as_choice(args, key, choices, code, hint):
    """枚举参数：非法值回契约里的专属错误码（E_UNKNOWN_PRESET 等），不笼统报 E_BAD_ARGS。"""
    v = args.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if not isinstance(v, str) or v not in choices:
        raise ArgsError("%s 必须是 %s 之一，当前 %r" % (key, " / ".join(choices), v),
                        code=code, hint=hint)
    return v


# ── 预览图（include_preview）──────────────────────────────────────────────
def _preview_block(path, max_side=768):
    """把落盘 PNG 转成 JPEG 预览的 image content block。

    PIL 缺失 / 图读不了时返回 None（静默降级成纯文本），绝不因此让工具失败。
    只读原图，不重编码输出目录里的产物（缓存写在 ~/.moss-image/cache）。
    """
    if not path:
        return None
    if mc is None:
        return None
    try:
        data = mc.thumbnail(path, max_side=int(max_side))
    except Exception:
        _log("preview_error", path=path, error=traceback.format_exc(limit=3))
        return None
    if not data:
        _log("preview_unavailable", path=path,
             hint="Pillow 缺失或图像不可读，已降级为纯文本")
        return None
    return {"type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mimeType": "image/jpeg"}


def _resource_link_block(path):
    """给客户端一个可直接读取的 file:// 链接（MCP resource_link）。

    图像本体留在磁盘上（无损），客户端需要原图时自己去读，不必把整张图塞进上下文。
    """
    if not path:
        return None
    try:
        uri = pathlib.Path(path).resolve().as_uri()
    except Exception:
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}.get(ext.lstrip("."), "application/octet-stream")
    return {"type": "resource_link", "uri": uri, "name": os.path.basename(path),
            "mimeType": mime, "description": "原始产物（无损 PNG，勿重编码）"}


def _preview_blocks(path, include, max_side=768):
    blocks = []
    link = _resource_link_block(path)
    if link:
        blocks.append(link)
    if include:
        blk = _preview_block(path, max_side)
        if blk:
            blocks.append(blk)
    return blocks or None


# ── 工具实现：每个 handler 收 dict，返回 (payload, extra_content_blocks) ──
def _t_capabilities(args):
    caps = mc.capabilities(refresh=_as_bool(args.get("refresh"), "refresh", False))
    payload = dict(caps)                      # 浅拷贝，避免污染客户端缓存
    payload.setdefault("ok", True)
    return payload, None


def _t_backend_status(args):
    include_jobs = _as_bool(args.get("include_jobs"), "include_jobs", True)
    payload = dict(mc.backend_info())         # 不抛异常、不自动拉起
    payload.setdefault("ok", False)
    payload["mcp"] = {"server_version": SERVER_VERSION, "log": _LOG_PATH,
                      "pid": os.getpid()}
    if include_jobs:
        try:
            payload["jobs"] = mc.current_jobs()
        except Exception as e:
            payload["jobs"] = []
            payload.setdefault("warnings", [])
            if isinstance(payload["warnings"], list):
                payload["warnings"].append({"code": getattr(e, "code", "E_HTTP"),
                                            "message": "拿不到任务列表：%s" % e})
    return payload, None


def _generate_kwargs(args):
    """把工具参数翻成 moss_client.generate 的关键字参数（两者语义一一对应）。"""
    seed = args.get("seed")
    if seed is None or (isinstance(seed, str) and seed.strip().lower() in ("", "random")):
        seed = "random"
    else:
        seed = _as_int(seed, "seed", 0, 2 ** 31 - 1)
    return {
        "negative": _as_str(args.get("negative"), "negative", default=None),
        "preset": _as_choice(args, "preset", PRESET_IDS, "E_UNKNOWN_PRESET",
                             "可用：draft / standard / quality / precise"
                             "（见 capabilities().presets）"),
        "width": _as_int(args.get("width"), "width", 256, 4096, None),
        "height": _as_int(args.get("height"), "height", 256, 4096, None),
        "steps": _as_int(args.get("steps"), "steps", 1, 200, None),
        "cfg": _as_num(args.get("cfg"), "cfg", 0.0, 20.0, None),
        "seed": seed,
        "engine": _as_choice(args, "engine", ENGINE_IDS, "E_UNKNOWN_ENGINE",
                             "可用：mlx8 / msmlx（见 capabilities().engines）"),
        "lora_mode": _as_choice(args, "lora_mode", LORA_MODE_IDS, "E_UNKNOWN_LORA_MODE",
                                "可用：akk4 / turbo / native25 / none / full"
                                "（见 capabilities().lora_modes）"),
        "char_lora": _as_str(args.get("char_lora"), "char_lora", default=None),
        "char_lora_scale": _as_num(args.get("char_lora_scale"), "char_lora_scale",
                                   0.0, 2.0, None),
        "enhance": _as_bool(args.get("enhance"), "enhance", None),
        "timeout": _as_num(args.get("timeout_s"), "timeout_s", 1, 6 * 3600, 1800),
        "idempotency_key": _as_str(args.get("idempotency_key"), "idempotency_key",
                                   default=None),
        "dedupe": _as_bool(args.get("dedupe"), "dedupe", True),
    }


def _t_generate(args):
    include_preview = _as_bool(args.get("include_preview"), "include_preview", False)
    preview_side = _as_int(args.get("preview_max_side"), "preview_max_side", 128, 1600, 768)

    job_id = _as_str(args.get("job_id"), "job_id", default=None)
    if job_id:
        # 续等模式：上次 E_JOB_TIMEOUT 的 job_id 直接接着等，不重新提交
        res = mc.wait_job(job_id,
                          timeout=_as_num(args.get("timeout_s"), "timeout_s",
                                          1, 6 * 3600, 1800))
        payload = mc.result_summary(res)
        payload["resumed_job_id"] = job_id
        return payload, _preview_blocks(res.get("path"), include_preview, preview_side)

    prompt = _as_str(args.get("prompt"), "prompt", required=True, maxlen=60000,
                     strict=True)
    res = mc.generate(prompt, **_generate_kwargs(args))
    payload = mc.result_summary(res)
    return payload, _preview_blocks(res.get("path"), include_preview, preview_side)


def _t_generate_batch(args):
    prompt = _as_str(args.get("prompt"), "prompt", required=True, maxlen=60000,
                     strict=True)
    n = _as_int(args.get("n"), "n", 1, 16, 4)
    seeds = args.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list):
            raise ArgsError("seeds 必须是整数数组", hint=_SCHEMA_HINT)
        seeds = [_as_int(s, "seeds[%d]" % i, 0, 2 ** 31 - 1)
                 for i, s in enumerate(seeds)]
        if len(seeds) < n:
            raise ArgsError("seeds 只有 %d 个，少于 n=%d" % (len(seeds), n),
                            hint="seeds 数量 >= n，或者不传 seeds 用随机种子")
        seeds = seeds[:n]
    seed_base = _as_int(args.get("seed_base"), "seed_base", 0, 2 ** 31 - 1, None)

    kw = _generate_kwargs(args)
    kw.pop("seed", None)
    kw.pop("dedupe", None)                    # 变体本来就是不同 seed，无需去重语义
    include_preview = _as_bool(args.get("include_preview"), "include_preview", False)
    preview_side = _as_int(args.get("preview_max_side"), "preview_max_side", 128, 1600, 768)

    results = mc.generate_batch(prompt, n=n, seeds=seeds, seed_base=seed_base, **kw)
    payload = {
        "ok": True,
        "count": len(results),
        "preset": kw.get("preset"),
        "elapsed_total": round(sum(float(r.get("elapsed") or 0) for r in results), 1),
        "results": [mc.result_summary(r) for r in results],
        "note": "变体串行执行（后端 GPU 锁），总耗时约等于 n 张之和",
    }
    first = results[0].get("path") if results else None
    return payload, _preview_blocks(first, include_preview, preview_side)


def _t_upscale(args):
    source = _as_str(args.get("source"), "source", required=True)
    include_preview = _as_bool(args.get("include_preview"), "include_preview", False)
    preview_side = _as_int(args.get("preview_max_side"), "preview_max_side", 128, 1600, 768)
    res = mc.upscale(
        source,
        model=_as_str(args.get("model"), "model", default=None),
        seed=_as_int(args.get("seed"), "seed", 0, 2 ** 31 - 1, None),
        sub=_as_str(args.get("sub"), "sub", default="") or "",
        timeout=_as_num(args.get("timeout_s"), "timeout_s", 1, 6 * 3600, 1800),
    )
    payload = mc.result_summary(res)
    return payload, _preview_blocks(res.get("path"), include_preview, preview_side)


def _t_list_outputs(args):
    items = mc.list_outputs(
        limit=_as_int(args.get("limit"), "limit", 1, 200, 20),
        since=_as_int(args.get("since"), "since", 0, None, None),
        prompt_contains=_as_str(args.get("prompt_contains"), "prompt_contains", default=None),
        sub=_as_str(args.get("sub"), "sub", default="") or "",
        with_meta=_as_bool(args.get("with_meta"), "with_meta", False),
    )
    return {"ok": True, "count": len(items), "outputs": items}, None


def _t_inspect(args):
    return mc.inspect(_as_str(args.get("path_or_name"), "path_or_name", required=True)), None


def _t_replay(args):
    source = _as_str(args.get("source"), "source", required=True)
    overrides = args.get("overrides")
    if overrides is not None and not isinstance(overrides, dict):
        raise ArgsError("overrides 必须是对象（只覆盖你想改的那几个参数）",
                        hint='{"seed": 123}')
    include_preview = _as_bool(args.get("include_preview"), "include_preview", False)
    preview_side = _as_int(args.get("preview_max_side"), "preview_max_side", 128, 1600, 768)
    res = mc.replay(
        source,
        overrides=overrides,
        timeout=_as_num(args.get("timeout_s"), "timeout_s", 1, 6 * 3600, 1800),
        compare=_as_bool(args.get("compare"), "compare", True),
    )
    payload = mc.result_summary(res)
    payload["replayed_from"] = res.get("replayed_from")
    return payload, _preview_blocks(res.get("path"), include_preview, preview_side)


def _t_validate_prompt(args):
    prompt = _as_str(args.get("prompt"), "prompt", required=True, maxlen=60000)
    r = mc.validate_prompt(
        prompt,
        width=_as_int(args.get("width"), "width", 256, 4096, None),
        height=_as_int(args.get("height"), "height", 256, 4096, None),
        negative=_as_str(args.get("negative"), "negative", default=None),
    )
    # 注意：ok=false 表示「提示词有问题」，工具本身执行成功（isError 仍是 false）
    r["verdict"] = "ok" if r.get("ok") else "invalid"
    if not r.get("ok"):
        r["hint"] = "先按 errors 修好再提交，否则会白跑一次生成"
    return r, None


def _t_build_layout(args):
    summary = _as_str(args.get("summary_caption"), "summary_caption", required=True)
    raw_items = args.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ArgsError("items 必须是非空数组",
                        hint='[{"label":"character","caption":"...","bbox":[y1,x1,y2,x2]}]，'
                             "bbox 可省略（按顺序竖直均分）")
    items = []
    for i, it in enumerate(raw_items):
        if not isinstance(it, dict):
            raise ArgsError("items[%d] 必须是对象" % i, hint=_SCHEMA_HINT)
        bbox = it.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ArgsError("items[%d].bbox 必须是 4 个数字 [y1,x1,y2,x2]" % i,
                                hint=_SCHEMA_HINT)
            bbox = [_as_num(v, "items[%d].bbox[%d]" % (i, j)) for j, v in enumerate(bbox)]
        items.append({
            "label": _as_str(it.get("label"), "items[%d].label" % i,
                             default="item%d" % i) or "item%d" % i,
            "caption": _as_str(it.get("caption"), "items[%d].caption" % i, required=True),
            "bbox": bbox,
        })
    width = _as_int(args.get("width"), "width", 256, 4096, 1184)
    height = _as_int(args.get("height"), "height", 256, 4096, 896)
    canvas = _as_str(args.get("canvas"), "canvas", default="single") or "single"
    text = mc.build_layout_json(summary, items, width, height, canvas=canvas, validate=True)
    check = mc.validate_prompt(text, width=width, height=height)
    return {"ok": True, "layout_json": text, "chars": len(text), "items": len(items),
            "width": width, "height": height,
            "warnings": check.get("warnings") or [],
            "note": "layout_json 直接原样作为 moss_generate 的 prompt 传（别再包一层）"}, None


def _t_cancel(args):
    job_id = _as_str(args.get("job_id"), "job_id", required=True)
    resp = mc.cancel(job_id)
    return {"ok": True, "job_id": job_id, "backend": resp}, None


# ── 工具清单（description 就是给 agent 的规范，写清楚边界）───────────────
_P_PRESET = {
    "type": "string",
    "enum": ["draft", "standard", "quality", "precise"],
    "default": "draft",
    "description": (
        "意图档位，决定引擎/步数/尺寸（除非另外显式传 width/height）：\n"
        "· draft    — 4 步 akk4，512×512，约 56 秒。验构图/措辞，定稿后再升档重跑。\n"
        "· standard — 4 步 akk4，1184×896（≈1MP，模型训练比例），实测 189 秒。日常出片，文字最稳。\n"
        "· quality  — Turbo 固定调度（15 次前向追 50 步质量），1184×896，实测 676 秒。负面词生效。\n"
        "· precise  — 25 步 CFG 3.0 + 自适应跳步，1184×896，约 15–20 分钟。复杂 Layout JSON 用。\n"
        "耗时是本机实测/折算的量级，不是承诺。"
    ),
}
_P_NEGATIVE = {
    "type": "string",
    "description": (
        "负面词。关键：akk4 档（draft/standard）CFG=1.0，负面词完全不参与计算，传了等于没传；"
        "要负面词生效请用 quality/precise。不传时，CFG≠1 的档位自动用官方默认负面词"
        "（nsfw, worst quality, bad anatomy, watermark, jpeg artifact）。"
    ),
}
_P_WIDTH = {"type": "integer", "minimum": 256, "maximum": 4096,
            "description": "宽（像素，必须是 16 的倍数；不传则用档位默认）。1MP 左右最贴训练分布。"}
_P_HEIGHT = {"type": "integer", "minimum": 256, "maximum": 4096,
             "description": "高（像素，必须是 16 的倍数；不传则用档位默认）。"}
_P_TIMEOUT = {"type": "number", "minimum": 1, "maximum": 21600, "default": 1800,
              "description": ("本次调用最多等多少秒（默认 1800）。到点不会丢任务："
                              "返回文本里有 job_id，用它再调一次本工具（job_id 续等模式）"
                              "或 moss_cancel 取消。")}
_P_INCLUDE_PREVIEW = {"type": "boolean", "default": False,
                      "description": ("true = 额外返回一个 image content block（base64 JPEG 预览，"
                                      "最长边由 preview_max_side 控制）。只用于快速看图，"
                                      "落盘产物仍是无损 PNG 绝对路径；Pillow 缺失时静默降级为纯文本。")}
_P_PREVIEW_SIDE = {"type": "integer", "minimum": 128, "maximum": 1600, "default": 768,
                   "description": "预览图最长边像素（仅 include_preview=true 时生效）。"}

TOOLS = [
    {
        "name": "moss_capabilities",
        "title": "能力发现（先调这个）",
        "description": (
            "本机 MOSS-Image 的真实能力清单，**任何生成/超分之前先调它**，不要凭记忆猜参数。\n"
            "返回：可用引擎（mlx8/msmlx）与档位（akk4/turbo/native25/none/full）、角色 LoRA、"
            "超分模型、意图档位预设（draft/standard/quality/precise，含各档耗时量级）、"
            "尺寸规则（16 的倍数、256..4096）、steps/cfg 范围、负面词生效规则、"
            "提示词长度建议（推荐 1300–2600 字符，4096 token 上下文）、后端 url/pid/产物目录、"
            "以及全部结构化错误码。\n"
            "后端串行执行（GPU 锁），排队上限默认 6。"
        ),
        "properties": {
            "refresh": {"type": "boolean", "default": False,
                        "description": "true = 绕过 60 秒本地缓存，重新问后端。"},
        },
        "required": [],
        "annotations": {"title": "能力发现", "readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_capabilities,
    },
    {
        "name": "moss_backend_status",
        "title": "后端状态",
        "description": (
            "后端健康与任务队列：url / pid / 产物目录 / health（running、queued）/ 当前任务列表。\n"
            "不会自动拉起后端（纯只读探测）。ok=false 表示后端没在跑——这是**状态报告**，"
            "不是工具失败（isError 仍是 false）；此时调 moss_generate 会按 MOSS_AGENT_AUTOSTART 决定"
            "是否自动拉起。\n"
            "用途：① 出图前确认后端活着；② 长任务超时后查 job_id 是否还在跑；③ 拿 pid 排查。"
        ),
        "properties": {
            "include_jobs": {"type": "boolean", "default": True,
                             "description": "是否附带当前任务列表（/api/current）。"},
        },
        "required": [],
        "annotations": {"title": "后端状态", "readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_backend_status,
    },
    {
        "name": "moss_generate",
        "title": "生成一张图（同步）",
        "description": (
            "同步生成一张图并等它落盘。**先调 moss_capabilities**，再用 moss_validate_prompt 校验提示词。\n"
            "\n"
            "推荐节奏：preset=draft（512²，约 56 秒）先验构图/措辞 → 满意后同 prompt 换 "
            "standard（实测 189s）或 quality（turbo，实测 676s）出成品；复杂四键 Layout JSON 用 precise。\n"
            "\n"
            "硬规则（照做能省掉大部分坑）：\n"
            "1) prompt 是自然语言 caption 或官方四键 Layout JSON 字符串；推荐 1300–2600 字符，"
            "超过 3000 字符会被编码器截断（关键内容别放最后）。\n"
            "2) 负面词：akk4 档（draft/standard）CFG=1.0 → negative **完全不生效**；"
            "要负面词请用 quality/precise。\n"
            "3) 幂等：同 prompt + 同 seed + 同参数 = 同一张图，默认命中台账直接返回（cached=true，"
            "不重复烧 GPU）。seed 不传或 seed=\"random\" 时没有幂等语义；要强制重跑传 dedupe=false。\n"
            "4) 返回的 path 是磁盘绝对路径（无损 PNG），**不要重新编码/压缩**；"
            "另有同名 .json 参数快照，moss_replay 靠它复现。要看图用 include_preview=true 拿 JPEG 预览。\n"
            "5) 长任务：timeout_s 默认 1800 秒；超时抛 E_JOB_TIMEOUT，返回文本里带 job_id —— "
            "用 job_id 再调一次本工具即可续等（此时 prompt 可省略），或 moss_cancel 取消。\n"
            "6) 后端一次只跑一个任务，并发提交会排队；同参数任务正在跑时会挂到同一 job 一起等。"
        ),
        "properties": {
            "prompt": {"type": "string",
                       "description": ("提示词：自然语言 caption，或 moss_build_layout 产出的四键 Layout JSON "
                                       "字符串。推荐 1300–2600 字符；写清风格/场景/主体/服装道具/动作表情/"
                                       "材质/光线。job_id 续等模式下可省略。")},
            "preset": _P_PRESET,
            "negative": _P_NEGATIVE,
            "width": _P_WIDTH,
            "height": _P_HEIGHT,
            "steps": {"type": "integer", "minimum": 1, "maximum": 200,
                      "description": ("采样步数（不传用档位默认：akk4=4、turbo=15、native25=25、none=20）。"
                                      "turbo 档忽略 steps/cfg。")},
            "cfg": {"type": "number", "minimum": 0, "maximum": 20,
                    "description": ("CFG（不传用档位默认：akk4=1.0，其余 3.0）。CFG=1.0 时负面词不参与计算；"
                                    "turbo 档忽略 cfg。")},
            "seed": {"type": ["integer", "string"], "minimum": 0, "maximum": 2147483647,
                     "description": ("采样种子：整数 0..2^31-1，或 \"random\"（默认）。"
                                     "给整数才有幂等语义（同 seed 同参数 → 同一张图）；"
                                     "要出变体就换 seed 或用 moss_generate_batch。")},
            "engine": {"type": "string", "enum": ["mlx8", "msmlx"],
                       "description": "引擎（一般不用传：preset 已绑定；turbo 强制 msmlx）。"},
            "lora_mode": {"type": "string",
                          "enum": ["akk4", "turbo", "native25", "none", "full"],
                          "description": "底层档位（一般不用传，用 preset 表达意图）。"},
            "char_lora": {"type": "string",
                          "description": ("角色 LoRA 文件名（取值见 capabilities().char_loras）；"
                                          "turbo 档不支持叠加。注意：触发词必须自己写进 caption，"
                                          "挂权重不会自动补触发词。")},
            "char_lora_scale": {"type": "number", "minimum": 0, "maximum": 2, "default": 1.0,
                                "description": "角色 LoRA 强度（0..2，默认 1.0）。"},
            "enhance": {"type": "boolean", "default": False,
                        "description": ("true = 本地 Qwen 强化提示词（+10~90 秒，失败自动回退）。"
                                        "提示词已经很细时不必开。")},
            "timeout_s": _P_TIMEOUT,
            "include_preview": _P_INCLUDE_PREVIEW,
            "preview_max_side": _P_PREVIEW_SIDE,
            "dedupe": {"type": "boolean", "default": True,
                       "description": ("false = 即使同 prompt+seed 也重新生成（默认 true 命中台账直接返回）。"
                                       "seed=random 时本来就是每次新图。")},
            "idempotency_key": {"type": "string",
                                "description": "自定义幂等键（高级用法；一般不用传）。"},
            "job_id": {"type": "string",
                       "description": ("续等模式：把上次 E_JOB_TIMEOUT 返回的 job_id 传进来，"
                                       "接着等同一任务（不会重新提交、不会再烧一次 GPU）。"
                                       "给了 job_id 时 prompt 等生成参数全部忽略。")},
        },
        "required": [],
        # 两种合法调用形态：新生成（要 prompt）或续等（要 job_id）
        "anyOf": [{"required": ["prompt"]}, {"required": ["job_id"]}],
        "annotations": {"title": "生成一张图", "readOnlyHint": False, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_generate,
    },
    {
        "name": "moss_generate_batch",
        "title": "同 prompt 多变体",
        "description": (
            "同一个 prompt 生成 n 个变体（不同 seed），串行执行后一次性返回全部结果。\n"
            "总耗时 ≈ n 张之和（后端串行），所以**先用 draft 档试变体**，选定后再单张升档出成品。\n"
            "种子：传 seeds（整数数组，长度 >= n）或 seed_base（自动 seed_base+i），都不传则每张随机。\n"
            "返回 results 数组，每项含 path（绝对路径）/sha256/params/warnings；"
            "include_preview=true 时只回第一张的预览图（省 token）。"
        ),
        "properties": {
            "prompt": {"type": "string", "description": "提示词（caption 或四键 Layout JSON 字符串）。"},
            "n": {"type": "integer", "minimum": 1, "maximum": 16, "default": 4,
                  "description": "变体数量（1..16，默认 4）。n 张串行，耗时按 n 倍算。"},
            "seeds": {"type": "array", "items": {"type": "integer"},
                      "description": "显式种子数组（长度必须 >= n；给了就不用 seed_base）。"},
            "seed_base": {"type": "integer", "minimum": 0, "maximum": 2147483647,
                          "description": "种子基准：第 i 张用 seed_base+i。"},
            "preset": _P_PRESET,
            "negative": _P_NEGATIVE,
            "width": _P_WIDTH,
            "height": _P_HEIGHT,
            "steps": {"type": "integer", "minimum": 1, "maximum": 200, "description": "采样步数（不传用档位默认）。"},
            "cfg": {"type": "number", "minimum": 0, "maximum": 20, "description": "CFG（不传用档位默认）。"},
            "engine": {"type": "string", "enum": ["mlx8", "msmlx"], "description": "引擎（一般不用传）。"},
            "lora_mode": {"type": "string", "enum": ["akk4", "turbo", "native25", "none", "full"],
                          "description": "底层档位（一般不用传）。"},
            "char_lora": {"type": "string", "description": "角色 LoRA 文件名（turbo 不支持）。"},
            "char_lora_scale": {"type": "number", "minimum": 0, "maximum": 2, "default": 1.0,
                                "description": "角色 LoRA 强度。"},
            "enhance": {"type": "boolean", "default": False, "description": "提示词强化（每张都做，更慢）。"},
            "timeout_s": _P_TIMEOUT,
            "include_preview": _P_INCLUDE_PREVIEW,
            "preview_max_side": _P_PREVIEW_SIDE,
        },
        "required": ["prompt"],
        "annotations": {"title": "批量变体", "readOnlyHint": False, "destructiveHint": False,
                        "idempotentHint": False, "openWorldHint": False},
        "handler": _t_generate_batch,
    },
    {
        "name": "moss_upscale",
        "title": "x4 超分",
        "description": (
            "对输出目录里已有的图做 x4 超分（产物落 <out_dir>/upscaled/，返回绝对路径）。\n"
            "source 传文件名（先用 moss_list_outputs 拿）或绝对路径；源图必须在本机输出目录内，"
            "否则返回 E_UPSCALE_SRC。\n"
            "模型：mewzoom（默认，最快约 7 秒）/ thera_air（最忠实）/ nomos2_hq（保真冠军，吃内存）/ "
            "anime6b（动漫线条）。取值以 capabilities().upscale_models 为准。\n"
            "超分也是后端串行任务，长边尺寸受内存限制；超时同样可以用 timeout_s + job_id 续等。"
        ),
        "properties": {
            "source": {"type": "string",
                       "description": "源图：文件名（如 xxx_20261108-145845.png）或绝对路径。"},
            "model": {"type": "string",
                      "enum": ["mewzoom", "anime6b", "thera_air", "nomos2_hq"],
                      "description": "超分模型 id（默认 mewzoom）。实际可用性以 "
                                     "capabilities().upscale_models[].ok 为准。"},
            "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647,
                     "description": "超分种子（可选，一般不用传）。"},
            "sub": {"type": "string", "default": "",
                    "description": "产物子目录（默认空 = upscaled/）。"},
            "timeout_s": _P_TIMEOUT,
            "include_preview": _P_INCLUDE_PREVIEW,
            "preview_max_side": _P_PREVIEW_SIDE,
        },
        "required": ["source"],
        "annotations": {"title": "x4 超分", "readOnlyHint": False, "destructiveHint": False,
                        "idempotentHint": False, "openWorldHint": False},
        "handler": _t_upscale,
    },
    {
        "name": "moss_list_outputs",
        "title": "列出最近产物",
        "description": (
            "列出输出目录里最近的图（默认 20 条，含原图与 upscaled/ 超分图），"
            "每条给 filename / 绝对路径 / 尺寸 / seed / 引擎 / 档位 / prompt 前 200 字。\n"
            "用途：拿 filename 去 moss_upscale / moss_inspect / moss_replay；"
            "或按 prompt_contains 找上次那张图，避免重复生成。"
        ),
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20,
                      "description": "最多返回多少条（默认 20）。"},
            "since": {"type": "integer",
                      "description": "只要 mtime >= 该 Unix 时间戳的产物（可选）。"},
            "prompt_contains": {"type": "string",
                                "description": "按 prompt 子串过滤（不区分大小写）。"},
            "sub": {"type": "string", "default": "",
                    "description": "只看某个子目录（空 = 全部，\"upscaled\" = 只看超分图）。"},
            "with_meta": {"type": "boolean", "default": False,
                          "description": "true = 附带每张的完整 .json 参数快照（返回体更大）。"},
        },
        "required": [],
        "annotations": {"title": "产物列表", "readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_list_outputs,
    },
    {
        "name": "moss_inspect",
        "title": "读参数快照",
        "description": (
            "读一张图的同名 .json 参数快照（无损复现的唯一依据），返回绝对路径 / 字节数 / sha256 / "
            "mtime / meta（prompt、seed、engine、lora_mode、steps、cfg 等）/ reproducible。\n"
            "reproducible=false 说明这张图没有可用快照（不是工作台生成的），moss_replay 会拒绝。\n"
            "纯只读，不触发任何生成。"
        ),
        "properties": {
            "path_or_name": {"type": "string",
                             "description": "图片绝对路径，或 moss_list_outputs 里的文件名。"},
        },
        "required": ["path_or_name"],
        "annotations": {"title": "读快照", "readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_inspect,
    },
    {
        "name": "moss_replay",
        "title": "原样复现并比对",
        "description": (
            "按参数快照原样重跑（同 prompt/seed/引擎/步数），并默认比对 sha256，返回 identical 字段。\n"
            "这是「无损」的验证路径：identical=true 表示逐字节一致；false 说明模型/引擎/跳步参数变过"
            "（warnings 里会说明）。overrides 可只改个别参数（如 {\"seed\": 7} 出变体）。\n"
            "注意 replay 一定会真跑一次生成（dedupe 关闭），耗时与原来那张相当；先确认后端空闲。"
        ),
        "properties": {
            "source": {"type": "string",
                       "description": "要复现的图：绝对路径或文件名（必须有同名 .json 快照）。"},
            "overrides": {"type": "object",
                          "description": ("只覆盖想改的参数，例如 {\"seed\": 7} 或 "
                                          "{\"width\": 1024, \"height\": 1024}。其余照快照。")},
            "compare": {"type": "boolean", "default": True,
                        "description": "true = 重跑后比对 sha256，给出 identical（默认 true）。"},
            "timeout_s": _P_TIMEOUT,
            "include_preview": _P_INCLUDE_PREVIEW,
            "preview_max_side": _P_PREVIEW_SIDE,
        },
        "required": ["source"],
        "annotations": {"title": "复现并比对", "readOnlyHint": False, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_replay,
    },
    {
        "name": "moss_validate_prompt",
        "title": "提示词校验（本地）",
        "description": (
            "纯本地校验，不连后端、不烧 GPU，几毫秒返回。**提交生成前先跑它**，能省掉一次 10 分钟的失败。\n"
            "检查：caption 长度（>3000 字符警告接近 4096 token 上限；<40 字符警告画面易失控）；"
            "Layout JSON 的顶层四键、bbox_coord_space=abs_centralized、bbox_format=yxyx、"
            "summary.caption、items 结构、bbox 是否为 4 个数字且顺序/范围合理、items 数量建议 5–6、"
            "negative 是否被误写进 JSON。\n"
            "返回 ok / kind(caption|layout_json) / errors[] / warnings[] / stats{chars,items} / verdict。\n"
            "注意：ok=false 表示**提示词有问题**（工具本身成功，isError=false），按 errors 修好再提交。"
        ),
        "properties": {
            "prompt": {"type": "string", "description": "待校验的提示词（caption 或 Layout JSON 字符串）。"},
            "width": _P_WIDTH,
            "height": _P_HEIGHT,
            "negative": {"type": "string", "description": "负面词（可选，用于上下文提示）。"},
        },
        "required": ["prompt"],
        "annotations": {"title": "提示词校验", "readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_validate_prompt,
    },
    {
        "name": "moss_build_layout",
        "title": "生成 Layout JSON",
        "description": (
            "把结构化输入编成官方四键 Layout JSON（summary/items/bbox_coord_space/bbox_format），"
            "避免手写 JSON 时键名/坐标写错——手写错键名不会报错，只会变成模型忽略的废话。\n"
            "items[].bbox 可省略：省略时按 items 顺序把画布竖直均分；给了就按 [y1,x1,y2,x2]"
            "（相对画布中心、单位像素）。换尺寸要按新画布等比缩放 bbox。\n"
            "返回 layout_json 字符串，**原样**作为 moss_generate 的 prompt 传（别再包一层 JSON）。"
        ),
        "properties": {
            "summary_caption": {"type": "string", "description": "整张图的整体描述（summary.caption）。"},
            "items": {"type": "array",
                      "description": ("元素数组（建议 5–6 个）。每项 {label, caption, bbox?}；"
                                      "bbox 省略则自动竖直均分。"),
                      "items": {"type": "object",
                                "properties": {
                                    "label": {"type": "string", "description": "元素标签，如 character / object。"},
                                    "caption": {"type": "string", "description": "该元素的详细描述。"},
                                    "bbox": {"type": "array", "items": {"type": "number"},
                                             "minItems": 4, "maxItems": 4,
                                             "description": "[y1,x1,y2,x2]，相对画布中心、单位像素。"},
                                },
                                "required": ["caption"]}},
            "width": {"type": "integer", "minimum": 256, "maximum": 4096, "default": 1184,
                      "description": "画布宽（16 的倍数，默认 1184）。"},
            "height": {"type": "integer", "minimum": 256, "maximum": 4096, "default": 896,
                       "description": "画布高（16 的倍数，默认 896）。"},
            "canvas": {"type": "string", "default": "single",
                       "description": "summary.canvas（固定 single）。"},
        },
        "required": ["summary_caption", "items"],
        "annotations": {"title": "生成 Layout JSON", "readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_build_layout,
    },
    {
        "name": "moss_cancel",
        "title": "取消任务",
        "description": (
            "取消排队中或运行中的任务（后端会杀掉子进程）。job_id 来源：E_JOB_TIMEOUT 的返回文本、"
            "moss_backend_status 的 jobs 列表。\n"
            "取消后那张图不会落盘；台账里也不会记成功，所以下次同参数会重新生成。"
        ),
        "properties": {
            "job_id": {"type": "string", "description": "要取消的任务 id。"},
        },
        "required": ["job_id"],
        "annotations": {"title": "取消任务", "readOnlyHint": False, "destructiveHint": True,
                        "idempotentHint": True, "openWorldHint": False},
        "handler": _t_cancel,
    },
]

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def _tool_spec(t):
    schema = {"type": "object",
              "properties": t["properties"],
              "required": list(t["required"])}
    if t.get("anyOf"):
        schema["anyOf"] = t["anyOf"]          # 例如 generate：要么给 prompt，要么给 job_id
    return {"name": t["name"], "title": t["title"], "description": t["description"],
            "inputSchema": schema,
            "annotations": t["annotations"]}


TOOL_SPECS = [_tool_spec(t) for t in TOOLS]

INSTRUCTIONS = (
    "MOSS-Image 端侧出图（本机 MLX，GPU 串行）。按这个顺序用工具：\n"
    "1) moss_capabilities —— 先看可用引擎/档位/预设/尺寸规则/错误码，不要凭记忆猜参数。\n"
    "2) moss_validate_prompt（复杂多主体用 moss_build_layout 生成四键 Layout JSON）—— 提交前本地校验。\n"
    "3) moss_generate(preset=draft) 验构图 → 满意后同 prompt 换 standard/quality 出成品。\n"
    "关键事实：\n"
    "· 耗时量级（本机实测）：draft 512² 约 56 秒；standard 1184×896 约 189 秒；"
    "quality（Turbo）约 676 秒；precise 25 步约 19 分钟。并发排队时等待不计入 elapsed。\n"
    "· akk4 档（draft/standard）CFG=1.0 → negative 完全不生效；要负面词用 quality/precise。\n"
    "· 同 prompt + 同 seed + 同参数 = 同一张图，默认幂等命中（cached=true）不重复生成；"
    "seed=\"random\"（默认）没有幂等语义。\n"
    "· 返回 path 是磁盘绝对路径（无损 PNG）+ 同名 .json 快照；不要重新编码原图。"
    "看图用 include_preview=true 拿 base64 JPEG 预览。\n"
    "· 长任务用 timeout_s（默认 1800）；超时返回 E_JOB_TIMEOUT 且文本里带 job_id，"
    "用它再调 moss_generate 续等，或 moss_cancel 取消。\n"
    "· 后端一次只跑一个任务（排队上限 6），并发提交会排队而不是并行。\n"
    "· 业务错误统一是 isError=true 的 JSON 文本，含 code / message / hint；按 code 分支处理。"
)


# ── 结果封装 ──────────────────────────────────────────────────────────────
def _tool_result(payload, extra=None):
    blocks = [{"type": "text",
               "text": json.dumps(payload, ensure_ascii=False, indent=1)}]
    if extra:
        blocks.extend([b for b in extra if b])
    out = {"content": blocks, "isError": False}
    if isinstance(payload, dict):
        out["structuredContent"] = payload
    return out


def _tool_error(d):
    return {"content": [{"type": "text", "text": json.dumps(d, ensure_ascii=False, indent=1)}],
            "isError": True}


def _moss_error_dict(e):
    if hasattr(e, "to_dict"):
        d = e.to_dict()
    else:
        d = {"ok": False, "code": "E_INTERNAL", "message": str(e)}
    jid = getattr(e, "job_id", None)
    if jid:
        d["job_id"] = jid
        d["resume"] = ("再调一次 moss_generate 并带上 job_id=%s 即可续等" % jid)
    return d


def _http_error_dict(e, tool_name=None):
    """把 urllib 的裸异常翻成和 MossError 同形状的结构化错误。

    moss_client 的 _json_get 没有包 HTTPError（例如 job_id 不存在时后端回 404），
    不兜住就会变成 E_INTERNAL，agent 会误以为是 MCP server 坏了。
    """
    status = getattr(e, "code", None)
    body = ""
    with contextlib.suppress(Exception):
        body = (e.read() or b"").decode("utf-8", "replace")[:400]
    message = "后端返回 HTTP %s" % status
    detail = None
    if body:
        with contextlib.suppress(Exception):
            obj = json.loads(body)
            if isinstance(obj, dict):
                message = str(obj.get("error") or obj.get("detail") or message)
                detail = obj
        if detail is None:
            detail = {"body": body}
    hint = "后端可能已重启或该资源已失效；用 moss_backend_status 确认后端状态"
    if tool_name == "moss_generate" and status == 404:
        hint = ("job_id 不存在或后端重启导致任务丢失：用 moss_backend_status 看当前任务，"
                "或去掉 job_id 重新生成")
    return {"ok": False, "code": "E_HTTP", "message": message, "hint": hint,
            "detail": detail}


def _url_error_dict(e):
    return {"ok": False, "code": "E_HTTP",
            "message": "连不上后端：%s" % (getattr(e, "reason", None) or e),
            "hint": "确认 web/app.py 在跑（或设置 MOSS_AGENT_URL）；"
                    "moss_backend_status 可查状态"}


def _attach_unknown_warnings(tool, args, payload):
    """把拼错的参数名变成一条可见警告（而不是静默忽略）。"""
    props = tool.get("properties") or {}
    unknown = sorted(k for k in args if not k.startswith("_") and k not in props)
    if unknown and isinstance(payload, dict):
        w = payload.get("warnings")
        if not isinstance(w, list):
            w = []
            payload["warnings"] = w
        w.append({"code": "W_UNKNOWN_ARG",
                  "message": "不认识的参数已忽略：%s" % ", ".join(unknown),
                  "hint": "参数名见 tools/list 里该工具的 inputSchema"})
    return payload


def _trim_args(args):
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 120:
            out[k] = v[:120] + "...(%d chars)" % len(v)
        elif isinstance(v, (list, dict)):
            s = json.dumps(v, ensure_ascii=False, default=str)
            out[k] = s if len(s) <= 240 else s[:240] + "..."
        else:
            out[k] = v
    return out


# ── JSON-RPC 收发 ─────────────────────────────────────────────────────────
def _send(obj):
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with _WRITE_LOCK:
        out = _OUT if _OUT is not None else sys.stdout.buffer
        out.write(data + b"\n")
        out.flush()


def _ok_response(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err_response(mid, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": mid, "error": err}


def _initialize(params):
    wanted = params.get("protocolVersion")
    version = wanted if wanted in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
    client = params.get("clientInfo") or {}
    _log("initialize", requested=wanted, negotiated=version,
         client=client if isinstance(client, dict) else str(client))
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "title": SERVER_TITLE,
                       "version": SERVER_VERSION},
        "instructions": INSTRUCTIONS,
    }


def _do_tools_call(mid, params):
    name = params.get("name")
    if not isinstance(name, str) or not name:
        _send(_err_response(mid, JSONRPC_INVALID_PARAMS, "tools/call 缺少 name"))
        return
    args = params.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        _send(_err_response(mid, JSONRPC_INVALID_PARAMS, "arguments 必须是对象"))
        return
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        _log("unknown_tool", tool=name)
        _send(_err_response(mid, JSONRPC_INVALID_PARAMS,
                            "Unknown tool: %s" % name,
                            {"known": sorted(TOOLS_BY_NAME)}))
        return

    t0 = time.time()
    _log("tool_call", tool=name, args=_trim_args(args))
    try:
        payload, extra = tool["handler"](args)
        payload = _attach_unknown_warnings(tool, args, payload)
        result = _tool_result(payload, extra)
        _log("tool_ok", tool=name, ms=int((time.time() - t0) * 1000),
             ok=payload.get("ok") if isinstance(payload, dict) else None)
    except ArgsError as e:
        d = e.to_dict()
        _log("tool_bad_args", tool=name, code=d.get("code"), message=d.get("message"))
        result = _tool_error(d)
    except mc.MossError as e:
        d = _moss_error_dict(e)
        _log("tool_moss_error", tool=name, code=d.get("code"),
             message=d.get("message"), ms=int((time.time() - t0) * 1000))
        result = _tool_error(d)
    except urllib.error.HTTPError as e:
        d = _http_error_dict(e, name)
        _log("tool_http_error", tool=name, status=getattr(e, "code", None),
             message=d.get("message"), ms=int((time.time() - t0) * 1000))
        result = _tool_error(d)
    except urllib.error.URLError as e:
        d = _url_error_dict(e)
        _log("tool_url_error", tool=name, message=d.get("message"))
        result = _tool_error(d)
    except Exception as e:
        _log("tool_crash", tool=name, error=traceback.format_exc())
        result = _tool_error({
            "ok": False, "code": "E_INTERNAL",
            "message": "%s: %s" % (type(e).__name__, e),
            "hint": "这是 MCP server 内部错误（不是后端报错），日志见 %s" % _LOG_PATH,
        })
    _send(_ok_response(mid, result))


def _spawn(fn, *a):
    def run():
        try:
            with _SEM:
                fn(*a)
        except Exception:
            _log("worker_crash", error=traceback.format_exc())
        finally:
            with _TRACK_LOCK:
                _WORKERS.discard(threading.current_thread())

    t = threading.Thread(target=run, daemon=True)
    with _TRACK_LOCK:
        _WORKERS.add(t)
    t.start()


def _drain(timeout=2.0):
    """stdin 关闭后给在飞的任务一点时间把响应写出去。"""
    deadline = time.time() + timeout
    while True:
        with _TRACK_LOCK:
            alive = [t for t in _WORKERS if t.is_alive()]
        if not alive or time.time() > deadline:
            return
        alive[0].join(max(0.05, deadline - time.time()))


def _dispatch(msg):
    if not isinstance(msg, dict):
        _send(_err_response(None, JSONRPC_INVALID_REQUEST,
                            "Invalid Request: 每条消息必须是 JSON 对象"))
        return
    method = msg.get("method")
    mid = msg.get("id")
    is_notification = "id" not in msg
    if not isinstance(method, str):
        if not is_notification:
            _send(_err_response(mid, JSONRPC_INVALID_REQUEST,
                                "Invalid Request: 缺少 method"))
        return

    params = msg.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        if not is_notification:
            _send(_err_response(mid, JSONRPC_INVALID_PARAMS, "params 必须是对象"))
        return

    # 通知（无 id）一律不回包，只记日志
    if method.startswith("notifications/"):
        _log("notification", method=method)
        return

    if method == "initialize":
        if not is_notification:
            _send(_ok_response(mid, _initialize(params)))
        return
    if method == "ping":
        if not is_notification:
            _send(_ok_response(mid, {}))
        return
    if method == "tools/list":
        if not is_notification:
            _send(_ok_response(mid, {"tools": TOOL_SPECS}))
        return
    if method == "tools/call":
        if is_notification:
            _log("tools_call_as_notification_ignored")
            return
        _spawn(_do_tools_call, mid, params)
        return
    if method in ("shutdown", "exit"):
        # 非 MCP 标准方法，但有些客户端会发；礼貌应答并退出
        if not is_notification:
            _send(_ok_response(mid, {}))
        _log("shutdown_requested", method=method)
        raise SystemExit(0)

    _log("method_not_found", method=method)
    if not is_notification:
        _send(_err_response(mid, JSONRPC_METHOD_NOT_FOUND,
                            "Method not found: %s" % method))


def _handle_line(line):
    try:
        msg = json.loads(line)
    except Exception as e:
        _log("parse_error", error=str(e), sample=line[:200])
        _send(_err_response(None, JSONRPC_PARSE_ERROR, "Parse error: %s" % e))
        return
    if isinstance(msg, list):                 # JSON-RPC 批量（MCP 不用，但别炸）
        for item in msg:
            _dispatch(item)
        return
    _dispatch(msg)


def _serve():
    stdin = sys.stdin.buffer
    while True:
        try:
            raw = stdin.readline()
        except Exception as e:
            _log("stdin_error", error=str(e))
            break
        if not raw:
            _log("eof")
            break
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        _handle_line(line)
    _drain(2.0)


def _install_signals():
    def _bye(signum, _frame):
        _log("signal", signum=signum)
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(Exception):
            signal.signal(sig, _bye)


def main(argv=None):
    global _OUT
    _OUT = sys.stdout.buffer
    # 从此以后任何 print 都进 stderr，协议流只由 _send() 写
    with contextlib.suppress(Exception):
        sys.stdout = sys.stderr

    if _IMPORT_ERROR:
        sys.stderr.write("[moss-mcp] 无法 import moss_client（%s），先确认 %s 存在\n"
                         % (_IMPORT_ERROR, os.path.join(_HERE, "moss_client.py")))
        return 2

    _rotate_log()
    _install_signals()
    _log("start", server=SERVER_NAME, version=SERVER_VERSION,
         python=sys.version.split()[0], argv=(argv if argv is not None else sys.argv[1:]),
         log=_LOG_PATH, out_dir=os.environ.get("MOSS_OUT"))
    try:
        _serve()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        _log("interrupt")
    except Exception:
        _log("fatal", error=traceback.format_exc())
        return 1
    finally:
        _log("stop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
