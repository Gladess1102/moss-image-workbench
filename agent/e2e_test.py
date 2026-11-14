#!/usr/bin/env python3
"""agent 接口层验收套件 —— 可重复运行，任何改动后跑一遍就知道有没有回归。

用法：
  python3 agent/e2e_test.py              # 静态 + 无 GPU（约 5 秒）
  python3 agent/e2e_test.py --gpu        # 加真实出图（draft 512² + 复现 + 超分，约 2 分钟）
  python3 agent/e2e_test.py --tiers      # 加三档实测（standard/quality/precise，30 分钟+）
  python3 agent/e2e_test.py --json       # 机器可读结果

设计原则：
  * 不修改任何产物、不改后端代码；GPU 项每次都先查内存与 swap。
  * 失败不中断，最后汇总 PASS/FAIL 与失败详情。
  * 每个断言都带"证据"，失败时打印命令/期望/实际。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moss_client as mc  # noqa: E402

ROOT = mc.ROOT
RESULTS = []
FAILED = []
SKIPPED = []
CREATED = []          # 套件自己生成的产物（结束时清理，避免污染图片库）


def _remember(res):
    """记下本次真生成的产物，便于收尾清理（幂等命中的不算）。"""
    try:
        if res and not res.get("cached") and res.get("path"):
            CREATED.append(res["path"])
    except Exception:
        pass
    return res


def check(name, fn, *args, **kw):
    t0 = time.time()
    try:
        detail = fn(*args, **kw)
        RESULTS.append({"name": name, "ok": True, "detail": detail,
                        "sec": round(time.time() - t0, 2)})
        print("  PASS  %-52s %s" % (name, _short(detail)))
        return True
    except SkipTest as e:
        SKIPPED.append({"name": name, "reason": str(e)})
        print("  SKIP  %-52s %s" % (name, str(e)[:70]))
        return False
    except Exception as e:                       # noqa: BLE001
        info = "%s: %s" % (type(e).__name__, e)
        FAILED.append({"name": name, "error": info})
        RESULTS.append({"name": name, "ok": False, "detail": info,
                        "sec": round(time.time() - t0, 2)})
        print("  FAIL  %-52s %s" % (name, info[:90]))
        return False


class SkipTest(Exception):
    pass


def _short(d):
    s = d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
    return s if len(s) <= 90 else s[:87] + "..."


def eq(actual, expected, what=""):
    if actual != expected:
        raise AssertionError("%s 期望 %r，实际 %r" % (what, expected, actual))
    return "%s=%r" % (what, actual) if what else "ok"


def truthy(v, what=""):
    if not v:
        raise AssertionError("%s 期望真值，实际 %r" % (what, v))
    return "%s=%r" % (what, v) if what else "ok"


def expect_error(fn, code, what=""):
    try:
        fn()
    except mc.MossError as e:
        if e.code != code:
            raise AssertionError("%s 期望 %s，实际 %s（%s）" % (what, code, e.code, e.message))
        return "%s -> %s" % (what or "err", e.code)
    raise AssertionError("%s 期望抛 %s，实际没抛" % (what, code))


# ── 内存纪律 ──────────────────────────────────────────────────────────────
def mem_state():
    free_pct, swap_used = None, None
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True,
                             timeout=15).stdout
        m = re.search(r"free percentage:\s*(\d+)", out)
        if m:
            free_pct = int(m.group(1))
    except Exception:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r"used = ([\d.]+)M", out)
        if m:
            swap_used = round(float(m.group(1)) / 1024.0, 2)
    except Exception:
        pass
    return {"free_pct": free_pct, "swap_used_gb": swap_used}


def require_memory(min_free=20, max_swap=10.0):
    st = mem_state()
    if st["free_pct"] is not None and st["free_pct"] < min_free:
        raise SkipTest("内存不足：free %d%% < %d%%（swap %.2fGB）"
                       % (st["free_pct"], min_free, st["swap_used_gb"] or 0))
    if st["swap_used_gb"] is not None and st["swap_used_gb"] > max_swap:
        raise SkipTest("swap 已用 %.2fGB > %.1fGB" % (st["swap_used_gb"], max_swap))
    return st


def backend_idle(wait_s=900):
    """等后端空闲，避免和别的任务/验收撞车。"""
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            h = mc._json_get(mc.backend(autostart=True)["url"] + "/api/health", timeout=5)
        except Exception:
            time.sleep(2)
            continue
        if not h.get("running") and not h.get("queued"):
            return h
        time.sleep(3)
    raise SkipTest("后端一直忙（等待 %.0fs 超时）" % wait_s)


# ── 1. 契约与实现一致性 ──────────────────────────────────────────────────
def t_contract_load():
    with open(os.path.join(HERE, "contract.json"), encoding="utf-8") as f:
        c = json.load(f)
    truthy(c.get("version"), "contract.version")
    truthy(c.get("presets"), "contract.presets")
    return "v%s, %d presets, %d errors" % (c["version"], len(c["presets"]),
                                           len(c.get("errors") or {}))


def t_contract_presets():
    with open(os.path.join(HERE, "contract.json"), encoding="utf-8") as f:
        c = json.load(f)
    eq(set(c["presets"]), set(mc.PRESETS), "preset ids")
    for pid, p in c["presets"].items():
        eq(p["lora_mode"], mc.PRESETS[pid]["lora_mode"], "%s.lora_mode" % pid)
        eq(p["width"], mc.PRESETS[pid]["width"], "%s.width" % pid)
        eq(p["height"], mc.PRESETS[pid]["height"], "%s.height" % pid)
    return "%d presets 对齐" % len(c["presets"])


def t_contract_errors():
    with open(os.path.join(HERE, "contract.json"), encoding="utf-8") as f:
        c = json.load(f)
    eq(set(c["errors"]), set(mc.ERROR_CODES), "error codes")
    return "%d 个错误码对齐" % len(mc.ERROR_CODES)


def t_contract_warnings():
    with open(os.path.join(HERE, "contract.json"), encoding="utf-8") as f:
        c = json.load(f)
    known = set(c.get("warnings") or {})
    emitted = {"W_NEG_IGNORED", "W_NEG_DEFAULT", "W_TURBO_IGNORES_STEPS", "W_PROMPT_LONG",
               "W_PROMPT_SHORT", "W_LAYOUT_EXTRA_KEYS", "W_LAYOUT_SUMMARY_KEYS",
               "W_LAYOUT_ITEM_KEYS", "W_LAYOUT_LABEL", "W_LAYOUT_CANVAS", "W_LAYOUT_ITEMS",
               "W_LAYOUT_BBOX_ORDER", "W_LAYOUT_BBOX_RANGE", "W_LAYOUT_NEGATIVE",
               "W_NOT_IDENTICAL", "W_STALE_LEDGER", "W_DRAFT_DEFAULT"}
    missing = emitted - known
    if missing:
        raise AssertionError("契约 warnings 缺：%s" % sorted(missing))
    return "%d 个警告码对齐" % len(known)


def t_mcp_tools():
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "mcp_server.py")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)

    def send(o):
        p.stdin.write(json.dumps(o, ensure_ascii=False) + "\n")
        p.stdin.flush()

    def recv():
        return json.loads(p.stdout.readline())

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                     "clientInfo": {"name": "e2e", "version": "0"}}})
    init = recv()
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = recv()["result"]["tools"]
    p.stdin.close()
    p.wait(timeout=10)
    eq(len(tools), 11, "tools 数量")
    names = {t["name"] for t in tools}
    expected = {"moss_capabilities", "moss_backend_status", "moss_generate",
                "moss_generate_batch", "moss_upscale", "moss_list_outputs",
                "moss_inspect", "moss_replay", "moss_validate_prompt",
                "moss_build_layout", "moss_cancel"}
    eq(names, expected, "tools 名单")
    for t in tools:
        truthy(t.get("inputSchema"), "%s.inputSchema" % t["name"])
        truthy(t.get("description"), "%s.description" % t["name"])
    eq(init["result"]["protocolVersion"], "2025-06-18", "协议版本协商")
    return "11 工具 + 协议协商"



def t_mcp_schema_coverage():
    """工具 schema 与实现互相对得上：schema 里的参数确实被读，读的参数 schema 里都有。"""
    src = open(os.path.join(HERE, "mcp_server.py"), encoding="utf-8").read()
    read_keys = set(re.findall(r'args\.get\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', src))
    read_keys |= set(re.findall(r'_as_choice\(\s*args\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"', src))
    import importlib.util
    spec = importlib.util.spec_from_file_location("mcp_mod", os.path.join(HERE, "mcp_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    schema_keys = set()
    for t in mod.TOOLS:
        schema_keys |= set(t["properties"])
    ghost = sorted(schema_keys - read_keys)          # schema 声明了但没人读
    orphan = sorted(read_keys - schema_keys)          # 读了但 schema 没声明
    if ghost:
        raise AssertionError("schema 声明但实现没读：%s" % ghost)
    if orphan:
        raise AssertionError("实现读了但 schema 没声明：%s" % orphan)
    return "%d 个参数名双向对齐" % len(schema_keys)


def t_python_api_errors():
    """Python 路径也必须只抛 MossError（不能漏 ValueError/AttributeError）。"""
    # validate_prompt 按设计"返回结构"而不是抛异常；其余入口必须抛 MossError
    vr = mc.validate_prompt(123)
    truthy(not vr["ok"], "validate_prompt(123) 返回 ok=False")
    eq(vr["errors"][0]["code"], "E_VALIDATION", "validate_prompt 错误码")
    cases = [
        ("steps 非数字", lambda: mc.normalize_params("x", steps="abc")),
        ("cfg 非数字", lambda: mc.normalize_params("x", cfg="x")),
        ("prompt 非字符串(generate)", lambda: mc.generate(123)),
    ]
    for name, fn in cases:
        try:
            fn()
        except mc.MossError as e:
            if not e.code:
                raise AssertionError("%s 抛了 MossError 但没有 code" % name)
        except Exception as e:                    # noqa: BLE001
            raise AssertionError("%s 漏出裸异常 %s: %s" % (name, type(e).__name__, e))
        else:
            raise AssertionError("%s 没有报错" % name)
    return "%d 例全部是 MossError" % len(cases)


def t_cli_prompt_file_missing():
    r = run_cli("generate", "-f", "/tmp/definitely-not-here-12345.txt")
    eq(r.returncode, 2, "exit code")
    err = json.loads(r.stderr)
    eq(err["code"], "E_VALIDATION", "错误码")
    return "缺失 -f 文件 -> E_VALIDATION/exit2"


def t_mcp_schema_defaults():
    src = open(os.path.join(HERE, "mcp_server.py"), encoding="utf-8").read()
    if '"default": "draft"' not in src:
        raise AssertionError("preset schema 默认值不是 draft")
    if '"default": "standard"' in src:
        raise AssertionError("仍有 schema 默认值是 standard")
    return "preset 默认 draft"

# ── 2. 能力清单 ──────────────────────────────────────────────────────────
def t_capabilities():
    caps = mc.capabilities(refresh=True)
    for k in ("backend", "engines", "lora_modes", "presets", "size_rule", "steps_rule",
              "negative_rule", "prompt_rule", "error_codes", "version", "unsupported"):
        truthy(k in caps, "capabilities.%s" % k)
    eq(len(caps["engines"]), 2, "engines 数量")
    ids = {m["id"] for m in caps["lora_modes"]}
    truthy({"akk4", "turbo"} <= ids, "lora_modes 含 akk4/turbo")
    truthy(caps["version"].get("mflux"), "version.mflux")
    truthy(caps["version"].get("mlx"), "version.mlx")
    return "engine=%d modes=%d ver=mflux %s" % (len(caps["engines"]), len(ids),
                                                caps["version"].get("mflux"))


# ── 3. 参数校验矩阵 ──────────────────────────────────────────────────────
VALID_MATRIX = [
    ("空 prompt", dict(prompt=""), "E_PROMPT_EMPTY"),
    ("纯空格 prompt", dict(prompt="   "), "E_PROMPT_EMPTY"),
    ("宽非 16 倍数", dict(prompt="x", width=1000), "E_SIZE"),
    ("高非 16 倍数", dict(prompt="x", height=900), "E_SIZE"),
    ("宽太小", dict(prompt="x", width=16), "E_SIZE"),
    ("宽太大", dict(prompt="x", width=8192), "E_SIZE"),
    ("未知 preset", dict(prompt="x", preset="nope"), "E_UNKNOWN_PRESET"),
    ("未知档位", dict(prompt="x", lora_mode="lora8"), "E_UNKNOWN_LORA_MODE"),
    ("未知引擎", dict(prompt="x", engine="cuda"), "E_UNKNOWN_ENGINE"),
    ("未知 LoRA", dict(prompt="x", char_lora="nope.safetensors"), "E_UNKNOWN_LORA"),
    ("turbo+LoRA", dict(prompt="x", preset="quality",
                        char_lora="mengmeng_mflux.safetensors"), "E_LORA_UNSUPPORTED"),
    ("steps=0", dict(prompt="x", steps=0), "E_VALIDATION"),
    ("steps=201", dict(prompt="x", steps=201), "E_VALIDATION"),
    ("cfg=-1", dict(prompt="x", cfg=-1), "E_VALIDATION"),
    ("cfg=21", dict(prompt="x", cfg=21), "E_VALIDATION"),
    ("seed 负数", dict(prompt="x", seed=-1), "E_VALIDATION"),
    ("seed 超范围", dict(prompt="x", seed=2 ** 31), "E_VALIDATION"),
    ("seed 非数字", dict(prompt="x", seed="abc"), "E_VALIDATION"),
    ("超长 prompt", dict(prompt="夜色老街，暖黄灯光。" * 900), "E_PROMPT_TOO_LONG"),
    ("JSON 数组", dict(prompt="[1,2,3]"), "E_BAD_LAYOUT_JSON"),
    ("Layout 缺键", dict(prompt='{"summary":{"caption":"x"}}'), "E_BAD_LAYOUT_JSON"),
    ("Layout 错 coord", dict(prompt=json.dumps({
        "summary": {"caption": "x"}, "items": [{"label": "object", "caption": "a",
                                                "bbox": [0, 0, 1, 1]}],
        "bbox_coord_space": "abs", "bbox_format": "yxyx"})), "E_BAD_LAYOUT_JSON"),
    ("Layout bbox 三项", dict(prompt=json.dumps({
        "summary": {"caption": "x"}, "items": [{"label": "object", "caption": "a",
                                                "bbox": [0, 0, 1]}],
        "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"})),
     "E_BAD_LAYOUT_JSON"),
    ("Layout text 项缺 text", dict(prompt=json.dumps({
        "summary": {"caption": "x"}, "items": [{"label": "text", "caption": "a",
                                                "bbox": [0, 0, 1, 1]}],
        "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"})),
     "E_BAD_LAYOUT_JSON"),
    ("弯引号 JSON", dict(prompt='{\u201csummary\u201d:{\u201ccaption\u201d:\u201cx\u201d}}'),
     "E_BAD_LAYOUT_JSON"),
]


def t_validation_matrix():
    ok = 0
    for name, kw, code in VALID_MATRIX:
        prompt = kw.pop("prompt")
        # 校验在本地就该拦下，不会提交到后端
        expect_error(lambda p=prompt, k=kw: mc.generate(p, _validate=True, **k), code, name)
        ok += 1
    return "%d 例全部命中预期错误码" % ok


def t_validation_ok_paths():
    cases = [
        ("普通 caption", "夜景街道，暖黄路灯，一位摊主正在翻烤串"),
        ("中文方括号开头", "[镜头推近] 夜市街景，暖黄灯光"),
        ("英文 caption", "a rainy night market, warm street lamps, cinematic"),
    ]
    for name, p in cases:
        r = mc.validate_prompt(p, 512, 512)
        truthy(r["ok"], name)
    layout = mc.build_layout_json("夜景", [
        {"label": "character", "caption": "店主，深蓝围裙"},
        {"label": "text", "caption": "招牌", "text": "深夜食堂"},
        {"label": "object", "caption": "木凳与水渍"},
    ], 1184, 896)
    r = mc.validate_prompt(layout, 1184, 896)
    truthy(r["ok"], "自建 Layout")
    eq(r["stats"]["items"], 3, "items")
    return "3 caption + 1 layout"


def t_layout_math():
    txt = mc.build_layout_json("测试", [{"label": "object", "caption": "a"}], 1000 - 8, 896)
    obj = json.loads(txt)
    bb = obj["items"][0]["bbox"]
    eq(bb, [-448, -496, 448, 496], "单 item 铺满画布")
    txt2 = mc.build_layout_json("测试", [{"label": "object", "caption": "a"},
                                        {"label": "object", "caption": "b"}], 1184, 896)
    b0 = json.loads(txt2)["items"][0]["bbox"]
    b1 = json.loads(txt2)["items"][1]["bbox"]
    truthy(b0[0] == -448 and b0[2] == 0, "第一块上半")
    truthy(b1[0] == 0 and b1[2] == 448, "第二块下半")
    return "bbox 中心坐标系正确"


# ── 4. 幂等键与 ETA ──────────────────────────────────────────────────────
def t_idem_key():
    base, _ = mc.normalize_params("同一个提示词", preset="draft", seed=42)
    k1 = mc._idem_key(base)
    k2 = mc._idem_key(dict(base))
    eq(k1, k2, "同参数同键")
    for field, val in (("seed", 43), ("width", 640), ("steps", 5), ("negative", "blur"),
                       ("lora_mode", "native25")):
        other = dict(base, **{field: val})
        if mc._idem_key(other) == k1:
            raise AssertionError("改 %s 后幂等键没变" % field)
    return "键稳定且对 5 个参数敏感"


def t_eta():
    p1, _ = mc.normalize_params("x" * 100, preset="draft")
    p2, _ = mc.normalize_params("x" * 100, preset="standard")
    p3, _ = mc.normalize_params("x" * 100, preset="quality")
    e1, e2, e3 = (mc.estimate_eta(p1)["seconds"], mc.estimate_eta(p2)["seconds"],
                  mc.estimate_eta(p3)["seconds"])
    truthy(e1 and e2 and e3, "三档都有估计")
    truthy(e1 < e2 < e3, "draft < standard < quality（%s/%s/%s）" % (e1, e2, e3))
    return "draft %ss / standard %ss / quality %ss" % (e1, e2, e3)


# ── 5. CLI ───────────────────────────────────────────────────────────────
def run_cli(*args, timeout=120):
    r = subprocess.run([sys.executable, os.path.join(HERE, "mossimg.py")] + list(args),
                       capture_output=True, text=True, timeout=timeout)
    return r


def t_cli_basics():
    for args in (["capabilities"], ["status"], ["list", "-n", "3"], ["prompt-guide"]):
        r = run_cli(*args)
        eq(r.returncode, 0, "exit(%s)" % " ".join(args))
        truthy(r.stdout.strip(), "stdout(%s)" % args[0])
    out = json.loads(run_cli("capabilities").stdout)
    truthy(out.get("engines"), "CLI capabilities 含 engines")
    truthy(out.get("error_codes"), "CLI capabilities 含 error_codes")
    return "capabilities/status/list/prompt-guide"


def t_cli_json_errors():
    r = run_cli("generate", "x", "--preset", "nope")
    eq(r.returncode, 2, "exit code")
    err = json.loads(r.stderr)
    eq(err["code"], "E_UNKNOWN_PRESET", "错误码")
    r2 = run_cli("validate-prompt", "x", "--w", "1000")
    eq(r2.returncode, 2, "validate exit")
    truthy("E_SIZE" in r2.stdout, "validate 输出含 E_SIZE")
    return "CLI 错误走 stderr JSON + exit 2"


def t_cli_layout_roundtrip():
    r = run_cli("layout", "--summary", "夜景", "--w", "1024", "--h", "1024",
                "--item", json.dumps({"label": "character", "caption": "店主"}))
    eq(r.returncode, 0, "exit")
    obj = json.loads(r.stdout)
    eq(obj["bbox_coord_space"], "abs_centralized", "coord space")
    eq(obj["items"][0]["bbox"], [-512, -512, 512, 512], "bbox")
    return "layout 输出合规"


# ── 6. MCP 工具调用（只读，不出图）────────────────────────────────────────
def t_mcp_calls():
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "mcp_server.py")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)

    def call(i, name, args):
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                                  "params": {"name": name, "arguments": args}},
                                 ensure_ascii=False) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())["result"]

    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                              "params": {"protocolVersion": "2024-11-05",
                                         "capabilities": {},
                                         "clientInfo": {"name": "e2e", "version": "0"}}},
                             ensure_ascii=False) + "\n")
    p.stdin.flush()
    json.loads(p.stdout.readline())
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    p.stdin.flush()

    r = call(1, "moss_capabilities", {})
    eq(r["isError"], False, "capabilities")
    truthy(r.get("structuredContent"), "structuredContent")
    r = call(2, "moss_validate_prompt", {"prompt": "夜景", "width": 1000})
    eq(r["isError"], False, "validate isError")
    r = call(3, "moss_backend_status", {})
    eq(r["isError"], False, "status")
    r = call(4, "moss_build_layout", {"summary_caption": "夜景", "items": [
        {"label": "character", "caption": "店主"}]})
    eq(r["isError"], False, "build_layout")
    r = call(5, "moss_generate", {"prompt": "x", "preset": "nope"})
    eq(r["isError"], True, "非法 preset 应 isError")
    txt = json.loads(r["content"][0]["text"])
    eq(txt["code"], "E_UNKNOWN_PRESET", "错误码")
    r = call(6, "moss_cancel", {"job_id": "deadbeef"})
    truthy(r["isError"], "cancel 不存在的 job 应 isError")
    p.stdin.close()
    p.wait(timeout=10)
    return "6 次 tools/call 全部符合预期"


# ── 7. 台账 / 端点容错 ───────────────────────────────────────────────────
def t_corrupt_ledger():
    tmp = tempfile.mkdtemp(prefix="moss-e2e-home-")
    old_home = os.environ.get("MOSS_AGENT_HOME")
    old_ledger = mc.LEDGER_FILE
    old_ep = mc.ENDPOINT_FILE
    try:
        os.environ["MOSS_AGENT_HOME"] = tmp
        mc.LEDGER_FILE = os.path.join(tmp, "agent-ledger.jsonl")
        mc.ENDPOINT_FILE = os.path.join(tmp, "agent-endpoint.json")
        with open(mc.LEDGER_FILE, "w", encoding="utf-8") as f:
            f.write('{"kind":"generate","key":"abc"}\n')
            f.write("这不是 JSON\n")
            f.write('{"kind":"generate","key":"def","status":"done","path":"/nope"}\n')
        entries = mc._ledger_read()
        eq(len(entries), 2, "坏行被跳过")
        with open(mc.ENDPOINT_FILE, "w", encoding="utf-8") as f:
            f.write("{{ broken")
        info = mc.backend_info()
        truthy(info.get("ok"), "端点文件坏了仍能发现后端")
        return "坏行跳过 + 端点自愈"
    finally:
        if old_home is None:
            os.environ.pop("MOSS_AGENT_HOME", None)
        else:
            os.environ["MOSS_AGENT_HOME"] = old_home
        mc.LEDGER_FILE = old_ledger
        mc.ENDPOINT_FILE = old_ep


# ── 8. GPU：真实出图 ────────────────────────────────────────────────────
E2E_PROMPT = ("雨后的老城巷口，一盏暖黄路灯照亮积水，穿深灰风衣的人撑着黑伞走过，"
              "背景是模糊的招牌与远处车灯，纪实摄影，浅景深，胶片颗粒")


GPU_SEED = [0]        # 生成用例摇一次，后面的幂等/复现用例复用它


def t_gpu_generate():
    require_memory()
    backend_idle()
    seed = int(time.time()) % 100000 + 7
    GPU_SEED[0] = seed
    res = _remember(mc.generate(E2E_PROMPT, preset="draft", seed=seed, dedupe=False,
                                timeout=600))
    truthy(os.path.isfile(res["path"]), "产物存在")
    import hashlib
    with open(res["path"], "rb") as f:
        real = hashlib.sha256(f.read()).hexdigest()
    eq(res["sha256"], real, "sha256 与文件一致")
    eq((res["width"], res["height"]), (512, 512), "尺寸")
    truthy(res.get("meta"), "快照存在")
    return "%s %dB %.1fs" % (os.path.basename(res["path"]), res["bytes"],
                             res["elapsed"] or 0)


def t_gpu_dedupe():
    require_memory()
    backend_idle()
    seed = GPU_SEED[0] or (int(time.time()) % 100000 + 7)
    t0 = time.time()
    res = mc.generate(E2E_PROMPT, preset="draft", seed=seed, timeout=60)
    dt = time.time() - t0
    truthy(res["cached"], "命中幂等")
    truthy(dt < 3, "命中耗时 < 3s（实际 %.2fs；锁不该被别人的任务挡住）" % dt)
    eq(res["elapsed"], 0.0, "cached elapsed")
    truthy(res.get("elapsed_original"), "elapsed_original 保留")
    return "cached，%.1fs" % dt


def t_gpu_replay():
    require_memory()
    backend_idle()
    seed = GPU_SEED[0] or (int(time.time()) % 100000 + 7)
    src = _remember(mc.generate(E2E_PROMPT, preset="draft", seed=seed, timeout=600))
    rep = _remember(mc.replay(src["path"], timeout=600))
    truthy(rep["identical"], "identical")
    eq(rep["sha256"], src["sha256"], "哈希一致")
    return "identical=True"


def t_gpu_upscale():
    require_memory()
    backend_idle()
    seed = GPU_SEED[0] or (int(time.time()) % 100000 + 7)
    src = _remember(mc.generate(E2E_PROMPT, preset="draft", seed=seed, timeout=600))
    up = _remember(mc.upscale(src["path"], model="mewzoom", timeout=600))
    truthy(os.path.isfile(up["path"]), "超分产物存在")
    eq(up["width"], 2048, "输出宽")
    return "%sx%s %.1fs" % (up["width"], up["height"], up["elapsed"] or 0)



def t_concurrency():
    """两个进程同时提交同一参数：应只生成一次，两个都拿到同一张图。"""
    require_memory()
    backend_idle()
    seed = int(time.time()) % 100000 + 11
    prompt = "清晨的旧书店，木架上堆满旧书，一束光从高窗斜射进来，尘埃可见"
    child = (
        "import sys, json, time\n"
        "sys.path.insert(0, %r)\n"
        "import moss_client as mc\n"
        "t0 = time.time()\n"
        "r = mc.generate(%r, preset='draft', seed=%d, timeout=900)\n"
        "print(json.dumps({'path': r['path'], 'cached': r['cached'],"
        " 'elapsed': r['elapsed'], 'wall': round(time.time()-t0, 1)}))\n"
    ) % (HERE, prompt, seed)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, "-c", child],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(2)]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=1200)
        if p.returncode != 0:
            raise AssertionError("子进程失败：%s" % (err or out)[-300:])
        outs.append(json.loads(out.strip().splitlines()[-1]))
    wall = time.time() - t0
    for o in outs:
        if not o.get("cached"):
            CREATED.append(o["path"])
    eq(outs[0]["path"], outs[1]["path"], "两个进程拿到同一张图")
    fresh = [o for o in outs if not o["cached"]]
    truthy(len(fresh) <= 1, "最多一个进程真正生成（实际 %d）" % len(fresh))
    truthy(wall < 150, "总耗时接近单张（%.1fs）" % wall)
    key_params, _ = mc.normalize_params(prompt, preset="draft", seed=seed)
    key = mc._idem_key(key_params)
    submitted = [e for e in mc._ledger_read()
                 if e.get("key") == key and e.get("status") == "submitted"]
    truthy(len(submitted) == 1, "台账里该键只有 1 次提交（实际 %d）" % len(submitted))
    return "同键并发 → 1 次生成，%.1fs" % wall

TIERS = [
    ("standard", dict(preset="standard")),
    ("quality", dict(preset="quality")),
    ("precise", dict(preset="precise")),
]


def t_tier(name, kw):
    require_memory(min_free=25, max_swap=10.0)
    backend_idle(wait_s=1800)
    seed = 424242
    res = _remember(mc.generate(E2E_PROMPT, seed=seed, dedupe=False, timeout=3600, **kw))
    return "%s %sx%s %.1fs %dB" % (name, res["width"], res["height"],
                                   res["elapsed"] or 0, res["bytes"])


# ── main ─────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="MOSS-Image agent 接口验收套件")
    ap.add_argument("--gpu", action="store_true", help="加真实出图（draft + 复现 + 超分）")
    ap.add_argument("--tiers", action="store_true", help="加三档实测（很慢）")
    ap.add_argument("--no-concurrency", action="store_true", help="跳过并发用例")
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="保留套件生成的图片（默认跑完删掉，避免污染图片库）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("== 静态 / 契约 ==")
    check("contract.json 可读", t_contract_load)
    check("presets 与实现一致", t_contract_presets)
    check("错误码与实现一致", t_contract_errors)
    check("警告码与契约一致", t_contract_warnings)
    check("MCP 工具清单", t_mcp_tools)
    check("MCP schema 与实现对齐", t_mcp_schema_coverage)
    check("MCP schema 默认值", t_mcp_schema_defaults)

    print("== 能力与校验 ==")
    check("capabilities 结构", t_capabilities)
    check("参数校验矩阵（%d 例）" % len(VALID_MATRIX), t_validation_matrix)
    check("合法输入不误杀", t_validation_ok_paths)
    check("Layout bbox 坐标数学", t_layout_math)
    check("Python API 只抛 MossError", t_python_api_errors)
    check("幂等键稳定性", t_idem_key)
    check("ETA 估计单调", t_eta)

    print("== CLI ==")
    check("基础子命令", t_cli_basics)
    check("错误输出为 JSON", t_cli_json_errors)
    check("-f 文件缺失", t_cli_prompt_file_missing)
    check("layout 输出合规", t_cli_layout_roundtrip)

    print("== MCP ==")
    check("tools/call 六例", t_mcp_calls)

    print("== 容错 ==")
    check("台账/端点文件损坏", t_corrupt_ledger)

    if args.gpu or args.tiers:
        print("== GPU（内存纪律生效）==")
        check("真实出图 + 哈希一致", t_gpu_generate)
        check("幂等命中", t_gpu_dedupe)
        check("replay 逐字节一致", t_gpu_replay)
        check("超分 mewzoom", t_gpu_upscale)
        if not args.no_concurrency:
            check("同键并发只生成一次", t_concurrency)
    if args.tiers:
        print("== 三档实测（很慢）==")
        for name, kw in TIERS:
            check("档位 %s" % name, t_tier, name, kw)

    if not args.keep_artifacts and CREATED:
        removed = 0
        for p in set(CREATED):
            for f in (p, os.path.splitext(p)[0] + ".json"):
                with contextlib.suppress(Exception):
                    os.remove(f)
                    removed += 1
        print("  已清理 %d 个测试产物（--keep-artifacts 可保留）" % removed)

    dt = time.time() - t0
    summary = {"ok": not FAILED, "passed": len(RESULTS) - len(FAILED),
               "failed": len(FAILED), "skipped": len(SKIPPED),
               "sec": round(dt, 1), "failed_items": FAILED, "skipped_items": SKIPPED,
               "results": RESULTS}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    else:
        print("\n结果：%d 通过 / %d 失败 / %d 跳过（%.1fs）"
              % (summary["passed"], summary["failed"], summary["skipped"], dt))
        for f in FAILED:
            print("  FAIL %s -> %s" % (f["name"], f["error"]))
        for s in SKIPPED:
            print("  SKIP %s -> %s" % (s["name"], s["reason"]))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
