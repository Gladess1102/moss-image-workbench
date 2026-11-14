#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent/mcp_server.py 的协议自测 —— 不需要真实出图。

它做什么
  起一个子进程跑 mcp_server.py，用「换行分隔的 JSON-RPC」走完整链路：
  initialize → notifications/initialized → ping → tools/list → tools/call
  （只读工具 moss_capabilities / moss_backend_status / moss_validate_prompt /
   moss_build_layout / moss_inspect，外加非法参数、未知工具、未知方法、坏 JSON 的
   错误路径），并断言 stdout 上除了 JSON-RPC 之外没有任何杂质。

为什么不需要出图
  所有用例都不调 moss_generate 的真生成路径（唯一涉及它的是「缺 prompt」的
  参数错误用例，在碰后端之前就被拦下）。因此不烧 GPU、不产生新图。

用法
  python3 agent/mcp_selftest.py              # 默认禁止自动拉起后端（无副作用）
  python3 agent/mcp_selftest.py --autostart  # 允许 moss_capabilities 自动拉起后端
  python3 agent/mcp_selftest.py -v           # 打印每条原始消息（截断）

退出码
  0 = 全部通过（SKIP 不算失败）  1 = 有 FAIL  2 = 起不了子进程
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(HERE, "mcp_server.py")

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

EXPECTED_TOOLS = [
    "moss_capabilities", "moss_backend_status", "moss_generate", "moss_generate_batch",
    "moss_upscale", "moss_list_outputs", "moss_inspect", "moss_replay",
    "moss_validate_prompt", "moss_build_layout", "moss_cancel",
]
READ_ONLY_TOOLS = {"moss_capabilities", "moss_backend_status", "moss_list_outputs",
                   "moss_inspect", "moss_validate_prompt", "moss_build_layout"}
GEN_TOOLS = {"moss_generate", "moss_generate_batch", "moss_upscale", "moss_replay"}

TMP = tempfile.mkdtemp(prefix="moss-mcp-selftest-")
_SPAWN_COUNT = [0]


class Fail(AssertionError):
    pass


class Skip(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


# ── 子进程客户端 ──────────────────────────────────────────────────────────
class Server(object):
    def __init__(self, env_extra=None, verbose=False):
        _SPAWN_COUNT[0] += 1
        self.verbose = verbose
        self.log_path = os.path.join(TMP, "mcp-%d.log" % _SPAWN_COUNT[0])
        env = dict(os.environ)
        env["MOSS_MCP_LOG"] = self.log_path
        env.setdefault("PYTHONUNBUFFERED", "1")
        if env_extra:
            env.update(env_extra)
        self.proc = subprocess.Popen(
            [sys.executable, SERVER_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, bufsize=0)
        self.q = queue.Queue()
        self.out_lines = []
        self.err_lines = []
        self.next_id = 0
        self.stray = []
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    # -- IO
    def _pump_out(self):
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not text.strip():
                    continue
                self.out_lines.append(text)
                self.q.put(text)
        finally:
            self.q.put(None)

    def _pump_err(self):
        try:
            for raw in iter(self.proc.stderr.readline, b""):
                self.err_lines.append(raw.decode("utf-8", "replace").rstrip("\r\n"))
        except Exception:
            pass

    def send_raw(self, text):
        self.proc.stdin.write((text + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.send_raw(json.dumps(msg, ensure_ascii=False))

    def next_message(self, timeout=30.0):
        try:
            text = self.q.get(timeout=timeout)
        except queue.Empty:
            raise Fail("等消息超时（%.0fs，server 还活着=%s）"
                       % (timeout, self.proc.poll() is None))
        if text is None:
            raise Fail("server 提前关闭了 stdout（exit=%s）" % self.proc.poll())
        if self.verbose:
            print("      <<< " + text[:300])
        try:
            return json.loads(text)
        except Exception as e:
            raise Fail("stdout 出现非 JSON 内容：%r（%s）" % (text[:200], e))

    def request(self, method, params=None, timeout=120.0):
        self.next_id += 1
        mid = self.next_id
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        payload = json.dumps(msg, ensure_ascii=False)
        if self.verbose:
            print("      >>> " + payload[:300])
        self.send_raw(payload)
        deadline = time.time() + timeout
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise Fail("等 id=%d（%s）的响应超时（%.0fs）" % (mid, method, timeout))
            obj = self.next_message(timeout=remain)
            if obj.get("id") == mid:
                return obj
            self.stray.append(obj)          # 乱序/无关消息先存着

    def assert_quiet(self, seconds=0.6):
        try:
            text = self.q.get(timeout=seconds)
        except queue.Empty:
            return
        if text is None:
            raise Fail("server 意外退出（exit=%s）" % self.proc.poll())
        raise Fail("预期没有响应，却收到：%s" % text[:300])

    def close(self, timeout=8.0):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        return self.proc.returncode

    def stderr_tail(self, n=15):
        return "\n".join(self.err_lines[-n:])


# ── 断言小工具 ────────────────────────────────────────────────────────────
def expect_result(resp, what="请求"):
    need("error" not in resp, "%s 返回了 JSON-RPC 错误：%s"
         % (what, json.dumps(resp.get("error"), ensure_ascii=False)))
    need("result" in resp, "%s 没有 result 字段：%s" % (what, resp))
    return resp["result"]


def expect_error(resp, code, what="请求"):
    need("error" in resp, "%s 预期是 JSON-RPC 错误，实际拿到 result：%s"
         % (what, json.dumps(resp.get("result"), ensure_ascii=False)[:300]))
    err = resp["error"]
    need(err.get("code") == code, "%s 错误码应为 %s，实际 %r（%s）"
         % (what, code, err.get("code"), err.get("message")))
    return err


def call_tool(srv, name, args=None, timeout=120.0):
    """调工具，返回 (MCP 结果 dict, 解析出的 payload dict)。"""
    resp = srv.request("tools/call", {"name": name, "arguments": args if args is not None else {}},
                       timeout=timeout)
    res = expect_result(resp, "tools/call %s" % name)
    need(isinstance(res.get("content"), list) and res["content"], "content 必须是非空数组")
    first = res["content"][0]
    need(first.get("type") == "text", "content[0].type 应为 text，实际 %r" % first.get("type"))
    need(isinstance(res.get("isError"), bool), "结果里必须有布尔 isError")
    try:
        payload = json.loads(first["text"])
    except Exception as e:
        raise Fail("content[0].text 不是 JSON：%r（%s）" % (first.get("text")[:200], e))
    return res, payload


def newest_png():
    roots = [os.environ.get("MOSS_OUT"), os.path.expanduser("~/Pictures/MOSS-Image")]
    best = None
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pat in ("*.png", "upscaled/*.png"):
            for p in glob.glob(os.path.join(root, pat)):
                try:
                    m = os.path.getmtime(p)
                except OSError:
                    continue
                if best is None or m > best[0]:
                    best = (m, p)
    return best[1] if best else None


# ── 用例 ──────────────────────────────────────────────────────────────────
def build_tests(srv, env_extra):
    T = []

    def case(name):
        def deco(fn):
            T.append((name, fn))
            return fn
        return deco

    @case("initialize（协议协商 + 回显客户端版本）")
    def _():
        res = expect_result(srv.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "moss-mcp-selftest", "version": "1.0.0"},
        }), "initialize")
        need(res.get("protocolVersion") == "2025-06-18",
             "没有回显客户端请求的协议版本，实际 %r" % res.get("protocolVersion"))
        caps = res.get("capabilities") or {}
        need(isinstance(caps.get("tools"), dict), "capabilities.tools 缺失")
        need(caps["tools"].get("listChanged") is False, "capabilities.tools.listChanged 应为 false")
        info = res.get("serverInfo") or {}
        need(info.get("name") == "moss-image", "serverInfo.name 应为 moss-image，实际 %r" % info.get("name"))
        need(bool(info.get("version")), "serverInfo.version 缺失")
        instr = res.get("instructions") or ""
        need("moss_capabilities" in instr, "instructions 应引导先调 moss_capabilities")
        need("job_id" in instr, "instructions 应说明 job_id 续等")
        return "protocolVersion=2025-06-18, serverInfo=%s v%s, instructions=%d 字符" % (
            info.get("name"), info.get("version"), len(instr))

    @case("协议版本：旧版回显 + 未知版本兜底")
    def _():
        notes = []
        for want, expect_echo in (("2024-11-05", True), ("1999-01-01", False)):
            s = Server(env_extra)
            try:
                res = expect_result(s.request("initialize", {"protocolVersion": want,
                                                            "capabilities": {}}), "initialize")
                got = res.get("protocolVersion")
                if expect_echo:
                    need(got == want, "请求 %s 应回显 %s，实际 %r" % (want, want, got))
                    notes.append("%s→回显" % want)
                else:
                    need(got in SUPPORTED_PROTOCOLS,
                         "未知版本应兜底到支持的版本，实际 %r" % got)
                    notes.append("%s→兜底 %s" % (want, got))
            finally:
                s.close()
        return "，".join(notes)

    @case("notifications/initialized（通知不回包）")
    def _():
        srv.notify("notifications/initialized", {})
        srv.notify("notifications/cancelled", {"requestId": 999})
        srv.assert_quiet(0.7)
        return "两条通知都没有回包"

    @case("ping")
    def _():
        res = expect_result(srv.request("ping"), "ping")
        need(res == {}, "ping 应返回空对象，实际 %r" % res)
        return "result={}"

    @case("tools/list（11 个工具 / schema / annotations）")
    def _():
        res = expect_result(srv.request("tools/list"), "tools/list")
        tools = res.get("tools")
        need(isinstance(tools, list), "tools 必须是数组")
        names = [t.get("name") for t in tools]
        need(names == EXPECTED_TOOLS,
             "工具清单不符：\n  期望 %s\n  实际 %s" % (EXPECTED_TOOLS, names))
        by_name = {t["name"]: t for t in tools}
        for t in tools:
            n = t["name"]
            need(len(t.get("description") or "") > 80, "%s 的 description 太短（agent 看不懂）" % n)
            sch = t.get("inputSchema") or {}
            need(sch.get("type") == "object", "%s.inputSchema.type 应为 object" % n)
            need(isinstance(sch.get("properties"), dict) and sch["properties"],
                 "%s.inputSchema.properties 缺失" % n)
            need(isinstance(sch.get("required"), list), "%s.inputSchema.required 必须是数组" % n)
            for r in sch["required"]:
                need(r in sch["properties"], "%s.required 里的 %r 不在 properties 里" % (n, r))
            ann = t.get("annotations") or {}
            for k in ("readOnlyHint", "destructiveHint", "openWorldHint", "idempotentHint"):
                need(k in ann, "%s 缺 annotations.%s" % (n, k))
            if n in READ_ONLY_TOOLS:
                need(ann["readOnlyHint"] is True, "%s 是只读工具，readOnlyHint 必须 true" % n)
                need(ann["destructiveHint"] is False, "%s.destructiveHint 应为 false" % n)
            if n in GEN_TOOLS:
                need(ann["readOnlyHint"] is False, "%s 不是只读工具，readOnlyHint 应为 false" % n)
                need(ann["destructiveHint"] is False, "%s.destructiveHint 必须 false" % n)
                need(ann["openWorldHint"] is False, "%s.openWorldHint 必须 false" % n)

        gen = by_name["moss_generate"]
        d = gen["description"]
        for needle in ("moss_capabilities", "draft", "standard", "quality", "precise",
                       "akk4", "幂等", "绝对路径", "timeout_s", "job_id", "cached"):
            need(needle in d, "moss_generate 的 description 缺少关键规范：%r" % needle)
        props = gen["inputSchema"]["properties"]
        need(props["preset"]["enum"] == ["draft", "standard", "quality", "precise"],
             "moss_generate.preset 枚举不对")
        need(props["include_preview"]["default"] is False, "include_preview 默认应为 false")
        need(props["timeout_s"]["default"] == 1800, "timeout_s 默认应为 1800")
        need("job_id" in props, "moss_generate 应支持 job_id 续等")
        need("negative" in props and "akk4" in props["negative"]["description"],
             "negative 的描述里应写清 akk4 不生效")
        return "%d 个工具，required/默认值/annotations 齐全" % len(tools)

    @case("tools/call moss_validate_prompt（合法 caption）")
    def _():
        prompt = ("老城区一间深夜面馆，木质门框上挂着褪色招牌，暖黄灯光从屋内漏到湿漉漉的青石板上，"
                  "一位穿深蓝围裙的店主正弯腰掀开蒸笼盖，白汽升腾，背景是模糊的街巷与远处霓虹，"
                  "纪实摄影风格，浅景深，胶片颗粒，柯达 Portra 400 色彩")
        res, p = call_tool(srv, "moss_validate_prompt", {"prompt": prompt})
        need(res["isError"] is False, "合法提示词不该 isError")
        need(p.get("ok") is True, "合法 caption 应校验通过：%s" % json.dumps(p.get("errors"), ensure_ascii=False))
        need(p.get("kind") == "caption", "kind 应为 caption，实际 %r" % p.get("kind"))
        need(p.get("stats", {}).get("chars") == len(prompt), "stats.chars 应等于字符数")
        need(p.get("verdict") == "ok", "verdict 应为 ok")
        need("warnings" in p and "errors" in p, "缺 warnings/errors 字段")
        return "kind=caption chars=%d errors=0" % len(prompt)

    @case("tools/call moss_validate_prompt（坏 Layout JSON → ok=false）")
    def _():
        res, p = call_tool(srv, "moss_validate_prompt", {"prompt": '{"summary": '})
        need(res["isError"] is False, "提示词校验本身是成功的（ok=false 不是 isError）")
        need(p.get("ok") is False, "坏 JSON 应校验失败")
        codes = [e.get("code") for e in (p.get("errors") or [])]
        need("E_BAD_LAYOUT_JSON" in codes, "错误码应为 E_BAD_LAYOUT_JSON，实际 %s" % codes)
        return "ok=false, errors=%s" % codes

    @case("tools/call moss_build_layout（生成合规四键 JSON）")
    def _():
        res, p = call_tool(srv, "moss_build_layout", {
            "summary_caption": "深夜面馆柜台前的店主，暖光纪实摄影",
            "items": [
                {"label": "character", "caption": "穿深蓝围裙弯腰掀蒸笼盖的店主"},
                {"label": "object", "caption": "冒着白汽的竹制蒸笼"},
                {"label": "scene", "caption": "湿漉漉的青石板与褪色招牌"},
                {"label": "scene", "caption": "模糊街巷与远处霓虹"},
                {"label": "light", "caption": "屋内暖黄灯光漏到地面"},
            ],
            "width": 1184, "height": 896,
        })
        need(res["isError"] is False, "build_layout 不该失败")
        need(p.get("ok") is True, "ok 应为 true")
        text = p.get("layout_json") or ""
        obj = json.loads(text)
        need(set(obj) == {"summary", "items", "bbox_coord_space", "bbox_format"},
             "顶层键必须是四键，实际 %s" % sorted(obj))
        need(obj["bbox_coord_space"] == "abs_centralized", "bbox_coord_space 必须是 abs_centralized")
        need(obj["bbox_format"] == "yxyx", "bbox_format 必须是 yxyx")
        need(len(obj["items"]) == 5, "items 数量应为 5")
        for it in obj["items"]:
            bb = it.get("bbox")
            need(isinstance(bb, list) and len(bb) == 4, "自动 bbox 必须是 4 个数字")
            need(bb[0] <= bb[2] and bb[1] <= bb[3], "bbox 顺序应为 y1<=y2, x1<=x2")
        # 生成物必须能通过自己的校验器
        _, v = call_tool(srv, "moss_validate_prompt", {"prompt": text, "width": 1184, "height": 896})
        need(v.get("ok") is True, "build_layout 的产物没通过 validate_prompt：%s"
             % json.dumps(v.get("errors"), ensure_ascii=False))
        return "%d 字符 / %d items，自校验通过" % (len(text), len(obj["items"]))

    @case("tools/call moss_capabilities（只读）")
    def _():
        res, p = call_tool(srv, "moss_capabilities", {"refresh": True})
        if res["isError"]:
            code = p.get("code")
            if code in ("E_BACKEND_DOWN", "E_BACKEND_START_TIMEOUT", "E_HTTP"):
                raise Skip("后端未运行（%s），结构化错误路径正确" % code)
            raise Fail("capabilities 失败，code=%r message=%r" % (code, p.get("message")))
        need(p.get("ok") is True, "ok 应为 true")
        presets = p.get("presets") or {}
        for k in ("draft", "standard", "quality", "precise"):
            need(k in presets, "presets 缺 %r" % k)
        need(isinstance(p.get("engines"), list) and p["engines"], "engines 为空")
        need(isinstance(p.get("lora_modes"), list) and p["lora_modes"], "lora_modes 为空")
        need(isinstance(p.get("error_codes"), dict) and p["error_codes"], "error_codes 缺失")
        rule = p.get("size_rule") or {}
        need(rule.get("multiple_of") == 16, "size_rule.multiple_of 应为 16")
        need(isinstance((p.get("backend") or {}).get("url"), str), "backend.url 缺失")
        return "engines=%d modes=%d presets=%s out_dir=%s" % (
            len(p["engines"]), len(p["lora_modes"]),
            ",".join(sorted(presets)), (p.get("backend") or {}).get("out_dir"))

    @case("tools/call moss_backend_status（只读，不自动拉起）")
    def _():
        res, p = call_tool(srv, "moss_backend_status", {"include_jobs": True})
        need(res["isError"] is False, "状态查询不该 isError（后端没跑也应是状态报告）")
        need(isinstance(p.get("ok"), bool), "ok 必须是布尔")
        need(isinstance(p.get("jobs"), list), "jobs 必须是数组")
        need(isinstance(p.get("mcp"), dict) and p["mcp"].get("server_version"),
             "应带上 mcp.server_version")
        if p.get("ok"):
            need(str(p.get("url", "")).startswith("http"), "url 不合法：%r" % p.get("url"))
            need(os.path.isdir(str(p.get("out_dir"))), "out_dir 不存在：%r" % p.get("out_dir"))
            return "ok=true url=%s jobs=%d" % (p.get("url"), len(p["jobs"]))
        return "ok=false code=%s（后端未运行，状态报告正确）" % p.get("code")

    @case("tools/call moss_inspect（读已有产物的快照）")
    def _():
        png = newest_png()
        if not png:
            raise Skip("输出目录里没有现成 PNG，跳过（不需要真出图）")
        res, p = call_tool(srv, "moss_inspect", {"path_or_name": png})
        need(res["isError"] is False, "inspect 不该失败：%s" % p.get("message"))
        need(p.get("ok") is True, "ok 应为 true")
        need(os.path.isabs(p.get("path", "")), "path 必须是绝对路径：%r" % p.get("path"))
        need(os.path.realpath(p["path"]) == os.path.realpath(png), "path 与实际文件不一致")
        need(len(p.get("sha256") or "") == 64, "sha256 应为 64 位十六进制")
        need(isinstance(p.get("reproducible"), bool), "reproducible 必须是布尔")
        return "%s sha256=%s… reproducible=%s" % (
            os.path.basename(p["path"]), (p.get("sha256") or "")[:12], p.get("reproducible"))

    @case("非法参数：moss_generate 缺 prompt → isError + E_BAD_ARGS")
    def _():
        res, p = call_tool(srv, "moss_generate", {})
        need(res["isError"] is True, "缺必填参数必须 isError=true")
        need(p.get("ok") is False, "payload.ok 应为 false")
        need(p.get("code") == "E_BAD_ARGS", "code 应为 E_BAD_ARGS，实际 %r" % p.get("code"))
        need("prompt" in (p.get("message") or ""), "message 应指明缺的是 prompt：%r" % p.get("message"))
        need(bool(p.get("hint")), "错误里应有 hint")
        return "code=E_BAD_ARGS message=%s" % p.get("message")

    @case("非法参数：moss_build_layout 的 items 类型错 → isError")
    def _():
        res, p = call_tool(srv, "moss_build_layout",
                           {"summary_caption": "x", "items": "not-a-list"})
        need(res["isError"] is True, "类型错必须 isError=true")
        need(p.get("code") == "E_BAD_ARGS", "code 应为 E_BAD_ARGS，实际 %r" % p.get("code"))
        return "code=E_BAD_ARGS message=%s" % p.get("message")

    @case("非法参数：moss_validate_prompt 的 width 越界 → isError")
    def _():
        res, p = call_tool(srv, "moss_validate_prompt", {"prompt": "abc", "width": 99999})
        need(res["isError"] is True, "越界参数必须 isError=true")
        need(p.get("code") == "E_BAD_ARGS", "code 应为 E_BAD_ARGS，实际 %r" % p.get("code"))
        return "code=E_BAD_ARGS message=%s" % p.get("message")

    @case("非法参数：未知 preset → 契约错误码 E_UNKNOWN_PRESET")
    def _():
        res, p = call_tool(srv, "moss_generate",
                           {"prompt": "一个足够长的测试提示词，用来验证枚举参数的错误码是否符合契约", "preset": "ultra"})
        need(res["isError"] is True, "未知 preset 必须 isError=true")
        need(p.get("code") == "E_UNKNOWN_PRESET",
             "code 应为契约里的 E_UNKNOWN_PRESET，实际 %r" % p.get("code"))
        need("draft" in (p.get("hint") or ""), "hint 应列出可用档位：%r" % p.get("hint"))
        return "code=E_UNKNOWN_PRESET hint=%s" % p.get("hint")

    @case("续等不存在的 job_id → 结构化错误（不得漏 E_INTERNAL）")
    def _():
        res, p = call_tool(srv, "moss_generate",
                           {"job_id": "job-does-not-exist-selftest", "timeout_s": 5}, timeout=30)
        need(res["isError"] is True, "不存在的 job_id 必须 isError=true")
        allowed = ("E_HTTP", "E_JOB_FAILED", "E_BACKEND_DOWN", "E_BACKEND_START_TIMEOUT",
                   "E_JOB_TIMEOUT")
        need(p.get("code") in allowed,
             "code 应为 %s 之一，实际 %r（说明底层异常漏成了 E_INTERNAL）"
             % (list(allowed), p.get("code")))
        need(bool(p.get("hint")), "错误里应有 hint 告诉 agent 下一步")
        return "code=%s hint=%s" % (p.get("code"), (p.get("hint") or "")[:60])

    @case("未知工具 → JSON-RPC -32602")
    def _():
        resp = srv.request("tools/call", {"name": "moss_no_such_tool", "arguments": {}})
        err = expect_error(resp, -32602, "未知工具")
        need("Unknown tool" in (err.get("message") or ""), "错误信息应含 Unknown tool：%r" % err.get("message"))
        return "code=-32602 message=%s" % err.get("message")

    @case("未知方法 → JSON-RPC -32601")
    def _():
        resp = srv.request("resources/list")
        err = expect_error(resp, -32601, "未知方法")
        need("Method not found" in (err.get("message") or ""), "错误信息应含 Method not found")
        return "code=-32601 message=%s" % err.get("message")

    @case("坏 JSON / 非对象参数 → -32700 / -32602 且服务不死")
    def _():
        srv.send_raw('{"jsonrpc": "2.0", "id": 1, "method": ')   # 故意截断
        obj = srv.next_message(timeout=10)
        need("error" in obj, "坏 JSON 应回 JSON-RPC 错误，实际 %r" % obj)
        need(obj["error"].get("code") == -32700, "坏 JSON 错误码应为 -32700，实际 %r" % obj["error"].get("code"))
        need(obj.get("id") is None, "解析失败时 id 应为 null，实际 %r" % obj.get("id"))

        resp = srv.request("tools/call", {"name": "moss_generate", "arguments": "oops"})
        expect_error(resp, -32602, "arguments 不是对象")

        res = expect_result(srv.request("ping"), "坏输入之后的 ping")
        need(res == {}, "坏输入之后服务应仍可用")
        return "-32700 → -32602 → ping 仍通"

    @case("stdout 只含 JSON-RPC（无日志/无 LSP 头）")
    def _():
        need(srv.out_lines, "没抓到任何 stdout 行")
        for i, line in enumerate(srv.out_lines):
            need(not line.startswith("Content-Length"), "第 %d 行是 LSP 头，不是换行分隔 JSON" % (i + 1))
            need("Traceback" not in line, "第 %d 行是 Python traceback" % (i + 1))
            obj = json.loads(line)                      # 解析失败直接抛
            need(isinstance(obj, dict), "第 %d 行不是 JSON 对象" % (i + 1))
            need(obj.get("jsonrpc") == "2.0", "第 %d 行缺 jsonrpc=2.0" % (i + 1))
            need("id" in obj and ("result" in obj or "error" in obj),
                 "第 %d 行既不是响应也没有 id" % (i + 1))
        need(srv.proc.poll() is None, "此时 server 不应退出（exit=%s）" % srv.proc.poll())
        return "%d 行全是合法 JSON-RPC 响应" % len(srv.out_lines)

    @case("include_preview 的预览块（PIL 缺失则静默降级）")
    def _():
        os.environ["MOSS_MCP_LOG"] = os.path.join(TMP, "inproc.log")
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import mcp_server as srvmod
        png = newest_png()
        if not png:
            raise Skip("没有现成 PNG 可做预览")
        blk = srvmod._preview_block(png, 256)
        need(srvmod._preview_block("/nonexistent-moss-xyz.png", 256) is None,
             "读不到的图必须返回 None（静默降级），不能抛异常")
        if blk is None:
            return "Pillow 不可用 → 已按预期降级为纯文本（None）"
        need(blk.get("type") == "image", "预览块 type 应为 image")
        need(blk.get("mimeType") == "image/jpeg", "预览块 mimeType 应为 image/jpeg")
        raw = base64.b64decode(blk.get("data") or "")
        need(raw[:2] == b"\xff\xd8", "预览数据不是 JPEG")
        need(len(raw) > 1000, "预览数据太小（%d 字节）" % len(raw))
        return "JPEG %d 字节（base64 %d 字符）" % (len(raw), len(blk["data"]))

    @case("日志写入 MOSS_MCP_LOG（不污染 stdout）")
    def _():
        need(os.path.isfile(srv.log_path), "日志文件没生成：%s" % srv.log_path)
        with open(srv.log_path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        need(len(lines) >= 3, "日志行数太少（%d）" % len(lines))
        events = []
        for ln in lines:
            obj = json.loads(ln)
            need("event" in obj and "ts" in obj, "日志行不是 JSONL（缺 event/ts）：%r" % ln[:120])
            events.append(obj["event"])
        for want in ("start", "initialize", "tool_call"):
            need(want in events, "日志里缺事件 %r（有 %s）" % (want, sorted(set(events))))
        return "%d 行，事件 %s" % (len(lines), ", ".join(sorted(set(events))))

    @case("EOF 后干净退出（exit=0）")
    def _():
        code = srv.close()
        need(code == 0, "stdin 关闭后应 exit=0，实际 %r；stderr 尾部：\n%s"
             % (code, srv.stderr_tail()))
        need(not srv.err_lines, "stderr 不该有内容，实际 %d 行：%s"
             % (len(srv.err_lines), srv.stderr_tail(5)))
        return "exit=0，stderr 干净"

    return T


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MOSS-Image MCP server 协议自测")
    ap.add_argument("--autostart", action="store_true",
                    help="允许 moss_capabilities 自动拉起后端（默认禁止，保证零副作用）")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印原始 JSON-RPC 消息")
    ap.add_argument("--timeout", type=float, default=180.0, help="整体兜底超时（秒）")
    args = ap.parse_args()

    if not os.path.isfile(SERVER_PATH):
        print("找不到 %s" % SERVER_PATH)
        return 2

    env_extra = {"MOSS_AGENT_AUTOSTART": "1" if args.autostart else "0"}
    print("MOSS-Image MCP server 协议自测")
    print("  server : %s" % SERVER_PATH)
    print("  python : %s (%s)" % (sys.version.split()[0], sys.executable))
    print("  后端   : %s" % ("允许自动拉起（MOSS_AGENT_AUTOSTART=1）" if args.autostart
                             else "不自动拉起（MOSS_AGENT_AUTOSTART=0，无副作用）"))
    print("  临时目录: %s" % TMP)
    print("")

    started = time.time()
    try:
        srv = Server(env_extra, verbose=args.verbose)
    except Exception as e:
        print("起不了子进程：%s" % e)
        return 2

    tests = build_tests(srv, env_extra)
    passed = failed = skipped = 0
    failures = []
    for i, (name, fn) in enumerate(tests, 1):
        t0 = time.time()
        try:
            detail = fn()
            passed += 1
            print(" %2d. PASS  %s" % (i, name))
            if detail:
                print("          %s" % detail)
        except Skip as e:
            skipped += 1
            print(" %2d. SKIP  %s" % (i, name))
            print("          %s" % e)
        except Fail as e:
            failed += 1
            failures.append((name, str(e)))
            print(" %2d. FAIL  %s" % (i, name))
            print("          %s" % str(e).replace("\n", "\n          "))
        except Exception as e:
            failed += 1
            failures.append((name, "%s: %s" % (type(e).__name__, e)))
            print(" %2d. FAIL  %s" % (i, name))
            print("          %s: %s" % (type(e).__name__, e))
        finally:
            dt = time.time() - t0
            if dt > 1.0:
                print("          (%.1fs)" % dt)

    # 兜底清理
    try:
        if srv.proc.poll() is None:
            srv.close(3.0)
    except Exception:
        pass

    elapsed = time.time() - started
    print("")
    print("结果：%d 通过 / %d 失败 / %d 跳过（%.1fs）"
          % (passed, failed, skipped, elapsed))
    print("日志：%s" % srv.log_path)
    if failures:
        print("")
        print("失败详情：")
        for name, msg in failures:
            print("  · %s\n      %s" % (name, msg.replace("\n", "\n      ")))
        tail = srv.stderr_tail()
        if tail:
            print("")
            print("server stderr 尾部：")
            print(tail)
        print("")
        print("自测失败")
        return 1
    print("")
    print("自测通过（未触发任何真实出图）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
