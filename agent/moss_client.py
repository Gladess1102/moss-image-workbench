#!/usr/bin/env python3
"""MOSS-Image agent client —— 本机 agent 调用端侧出图的标准入口。

设计目标（对应"正确 / 无损 / 更好"三条）：

  正确  · 参数在本地先校验（16 倍数、档位/引擎绑定、CFG 与负面词的关系、Layout JSON 结构），
          把后端只回一句 "unknown lora_mode" 的失败提前变成可读的 ValidationError。
  无损  · 只返回磁盘上的绝对路径 + 完整参数快照，绝不重编码图像；
          replay() 用快照原样重跑，并比对 sha256 报告是否逐字节一致。
  更好  · 能力发现、意图档位、ETA 估计、幂等（同参数同 seed 不重复烧 10 分钟）、
          变体批量、进度回调、结构化错误码。

零第三方依赖，只用标准库；可被任意 agent 进程 import，也可经 mossimg CLI / MCP server 调用。

后端就是现有的 web/app.py（HTTP + JSON）。本模块只做三件事：
  1) 找到或拉起一个 app.py（7860 被占时自动换端口，端点写进 ~/.moss-image/agent-endpoint.json）
  2) 把 /api/generate、/api/job、/api/upscale、/api/history、/api/meta 包成同步语义
  3) 维护本地台账（幂等键 → job/产物），让重复请求直接命中已有图片

环境变量（都有默认值，见 DEFAULTS）：
  MOSS_AGENT_URL     显式指定后端地址，如 http://127.0.0.1:7861
  MOSS_OUT           输出目录（默认 ~/Pictures/MOSS-Image，与后端一致）
  MOSS_PYTHON        拉起后端用的解释器（默认探测有 PIL 的 python3）
  MOSS_AGENT_HOME    端点/台账/日志目录（默认 ~/.moss-image）
  MOSS_AGENT_AUTOSTART  0 = 只复用已有后端，不自动拉起
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

__version__ = "1.0.0"

# ── 路径与默认值 ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ~/moss-image
WEB = os.path.join(ROOT, "web")
APP_PY = os.path.join(WEB, "app.py")
DEFAULT_OUT = os.path.expanduser("~/Pictures/MOSS-Image")
AGENT_HOME = os.environ.get("MOSS_AGENT_HOME") or os.path.expanduser("~/.moss-image")
ENDPOINT_FILE = os.path.join(AGENT_HOME, "agent-endpoint.json")
LEDGER_FILE = os.path.join(AGENT_HOME, "agent-ledger.jsonl")
GEN_LOCK = os.path.join(AGENT_HOME, "agent-generate.lock")
LEDGER_MAX_BYTES = 4 * 1024 * 1024          # 超过就压缩到最近 3000 条
LEDGER_TTL = float(os.environ.get("MOSS_AGENT_CACHE_TTL") or 7 * 86400)
INFLIGHT_TTL = float(os.environ.get("MOSS_AGENT_INFLIGHT_TTL") or 6 * 3600)
LOG_DIR = os.path.join(AGENT_HOME, "logs")
START_LOCK = os.path.join(AGENT_HOME, "agent-start.lock")
CACHE_DIR = os.path.join(AGENT_HOME, "cache")

# 发现：7860..7869 都看一眼（7860 是工作台/GUI 的默认端口，也是 IndexTTS 的默认端口，
#       所以只"复用"不"占用"）。
DISCOVERY_PORTS = tuple(range(7860, 7870))
# 自己拉起后端时**从 7861 起**：7860 留给桌面工作台与 IndexTTS，
# 不跟别的服务抢端口（2026-11-08 用户明确要求）。
AGENT_PORT = int(os.environ.get("MOSS_AGENT_PORT") or 7861)
START_PORTS = (AGENT_PORT,) + tuple(p for p in range(7862, 7880) if p != AGENT_PORT)
DEFAULT_TIMEOUT = 1800.0                    # 单张默认上限（turbo 1MP 实测约 600s）
POLL_INTERVAL = 1.0

# 官方负面词（与 app.py 的 OFFICIAL_NEG 保持一致；capabilities() 会以后端返回的为准）
DEFAULT_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"

# 意图档位：agent 只需要选意图，不必记引擎/步数/LoRA 这些内部细节。
# 数值全部来自本机实测（见 web/README.md），标 estimated 的按像素折算。
PRESETS = {
    "draft": {
        "label": "草稿 · 最快",
        "eta_s": 56,
        "desc": "4 步 akk4，512x512，实测约 44s。用于验证构图/措辞，确定后再升档重跑。",
        "engine": "mlx8", "lora_mode": "akk4", "width": 512, "height": 512,
        "steps": None, "cfg": None,
    },
    "standard": {
        "label": "标准 · 日常出片",
        "eta_s": 189,
        "desc": "4 步 akk4，1184x896（约 1MP，模型的训练比例），文字最稳。负面词不生效。",
        "engine": "mlx8", "lora_mode": "akk4", "width": 1184, "height": 896,
        "steps": None, "cfg": None,
    },
    "quality": {
        "label": "高质量 · Turbo",
        "eta_s": 600,
        "desc": "Turbo 固定调度（15 次前向，追 50 步质量），1184x896，实测约 10 分钟。负面词生效。",
        "engine": "msmlx", "lora_mode": "turbo", "width": 1184, "height": 896,
        "steps": None, "cfg": None,
    },
    "precise": {
        "label": "精细 · 25 步",
        "eta_s": 1160,
        "desc": "25 步 CFG 3.0 + 自适应跳步，负面词生效，适合复杂 Layout JSON。比 draft 慢一个量级。",
        "engine": "mlx8", "lora_mode": "native25", "width": 1184, "height": 896,
        "steps": None, "cfg": None,
    },
}

# 每百万像素的秒数（本机实测折算；只用于给 agent 一个量级判断，不承诺精度）
_SEC_PER_MP = {
    ("mlx8", "akk4"): 169.0,       # 44.4s @ 512² = 0.262MP
    ("mlx8", "native25"): 1090.0,  # 334.7s @ 640x480 = 0.307MP
    ("mlx8", "none"): 1717.0,      # 450s @ 512²
    ("mlx8", "full"): 2476.0,      # 760.3s @ 640x480
    ("msmlx", "turbo"): 566.0,     # ~600s @ 1184x896 = 1.060MP
}

ERROR_CODES = {
    "E_VALIDATION": "参数不合法（见 hint / detail 字段）",
    "E_PROMPT_EMPTY": "prompt 为空",
    "E_PROMPT_TOO_LONG": "prompt 超出编码器上下文（4096 token），会被截断或退化",
    "E_BAD_LAYOUT_JSON": "Layout JSON 解析失败或结构不符合四键规范",
    "E_SIZE": "width/height 必须是 16 的倍数，且在 256..4096 之间",
    "E_UNKNOWN_PRESET": "未知的意图档位，见 capabilities().presets",
    "E_UNKNOWN_LORA_MODE": "未知档位 id，见 capabilities().lora_modes",
    "E_UNKNOWN_ENGINE": "未知引擎 id，见 capabilities().engines",
    "E_ENGINE_UNAVAILABLE": "该引擎在本机不可用（缺 venv/权重）",
    "E_UNKNOWN_LORA": "角色 LoRA 文件不存在，见 capabilities().char_loras",
    "E_LORA_UNSUPPORTED": "该引擎/档位不支持叠加角色 LoRA",
    "E_QUEUE_FULL": "后端队列已满（MOSS_MAX_QUEUE），稍后重试",
    "E_BACKEND_DOWN": "后端没起、已退出或健康检查失败",
    "E_BACKEND_START_TIMEOUT": "拉起后端超时（默认 120s）",
    "E_JOB_TIMEOUT": "等待任务超时；job_id 仍有效，可 wait_job 续等",
    "E_JOB_FAILED": "后端任务失败（error 字段有子进程日志尾部）",
    "E_CANCELLED": "任务已被取消",
    "E_OUTPUT_MISSING": "任务成功但产物路径找不到（检查 MOSS_OUT 是否与后端一致）",
    "E_UPSCALE_SRC": "超分源图不存在",
    "E_HTTP": "HTTP 层错误（连接/协议/非 JSON 响应）",
    # 以下两个主要由 MCP/CLI 层产生，但属于"接口可能返回的码"，统一登记以便 agent 分支
    "E_BAD_ARGS": "MCP/CLI 层参数类型或取值不合法（在碰后端之前拦下）",
    "E_INTERNAL": "客户端内部异常（出现即 bug，请带上下文反馈）",
}


class MossError(Exception):
    """所有错误的基类。agent 只需看 code 与 hint 就能决定下一步。"""

    def __init__(self, code, message="", hint="", detail=None):
        self.code = code
        self.message = message or ERROR_CODES.get(code, code)
        self.hint = hint
        self.detail = detail
        super().__init__("%s: %s" % (code, self.message))

    def to_dict(self):
        d = {"ok": False, "code": self.code, "message": self.message}
        if self.hint:
            d["hint"] = self.hint
        if self.detail is not None:
            d["detail"] = self.detail
        return d


class ValidationError(MossError):
    def __init__(self, message, code="E_VALIDATION", hint="", detail=None):
        super().__init__(code, message, hint, detail)


# ── 小工具 ────────────────────────────────────────────────────────────────
def _now():
    return time.time()


def _mkdirs():
    for d in (AGENT_HOME, LOG_DIR, CACHE_DIR):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass


@contextlib.contextmanager
def _file_lock(path, timeout=120.0):
    """跨进程互斥（flock）。拿不到就退化成无锁执行，不让锁本身成为单点故障。"""
    _mkdirs()
    fh = None
    try:
        fh = open(path, "a+")
        try:
            import fcntl
        except ImportError:
            yield
            return
        deadline = time.time() + timeout
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() > deadline:
                    break
                time.sleep(0.2)
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_get(url, timeout=10.0, ok_404=False):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        if ok_404 and e.code == 404:
            return {"error": "not found"}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:
            obj = {"error": raw[:200].decode("utf-8", "replace")}
        raise _backend_error(obj.get("error") or ("HTTP %d" % e.code), e.code)
    except urllib.error.URLError as e:
        raise MossError("E_BACKEND_DOWN", "连不上后端：%s" % getattr(e, "reason", e),
                        "确认 web/app.py 在跑，或让客户端自动拉起")
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise MossError("E_HTTP", "响应不是 JSON：%r" % raw[:120])


def _json_post(url, body, timeout=30.0):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:
            obj = {"error": raw[:200].decode("utf-8", "replace")}
        msg = obj.get("error") or ("HTTP %d" % e.code)
        raise _backend_error(msg, e.code)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise MossError("E_HTTP", "响应不是 JSON：%r" % raw[:120])


def _backend_error(msg, status=None):
    """把后端的中文错误映射成稳定错误码，agent 才能分支处理。"""
    m = str(msg)
    table = [
        ("prompt empty", "E_PROMPT_EMPTY", "prompt 不能为空"),
        ("16 的倍数", "E_SIZE", "宽高取 16 的倍数，如 1184x896 / 1024x1024"),
        ("unknown lora_mode", "E_UNKNOWN_LORA_MODE", "先调 capabilities() 拿可用档位"),
        ("unknown engine", "E_UNKNOWN_ENGINE", "先调 capabilities() 拿可用引擎"),
        ("未知的角色 LoRA", "E_UNKNOWN_LORA", "文件名必须在 capabilities().char_loras 里"),
        ("暂不支持叠加角色 LoRA", "E_LORA_UNSUPPORTED", "换 akk4/native25 等非 turbo 档"),
        ("队列已满", "E_QUEUE_FULL", "稍等或先取消排队任务"),
        ("找不到源图", "E_UPSCALE_SRC", "name 必须是 list_outputs() 里的文件名"),
        ("unknown model", "E_VALIDATION", "超分模型见 capabilities().upscale_models"),
        ("未就绪", "E_ENGINE_UNAVAILABLE", "该模型/执行器在本机不可用"),
        ("权重缺失", "E_ENGINE_UNAVAILABLE", "权重文件缺失"),
        ("forbidden", "E_VALIDATION", "请求被后端写保护拒绝（Origin/Content-Type）"),
    ]
    table.append(("not found", "E_VALIDATION",
                  "job_id / 文件名不存在，或后端重启过；用 list_outputs() 重新找"))
    for needle, code, hint in table:
        if needle in m:
            return MossError(code, m, hint)
    return MossError("E_VALIDATION", m, "", {"http_status": status})


def _is_our_backend(health):
    return isinstance(health, dict) and health.get("app") == "moss-image"


# ── 后端发现 / 拉起 ───────────────────────────────────────────────────────
def _health(url, timeout=3.0):
    try:
        h = _json_get(url.rstrip("/") + "/api/health", timeout=timeout)
        return h if _is_our_backend(h) else None
    except Exception:
        return None


def _read_endpoint_file():
    try:
        with open(ENDPOINT_FILE, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_endpoint_file(info):
    _mkdirs()
    tmp = ENDPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    os.replace(tmp, ENDPOINT_FILE)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _adopt(url, port, health):
    """认领一个已在跑的后端。若端点文件里记的就是同一个 pid，保留它的 started_by / out_dir，
    否则会被降级成 external —— 那样之后 mossimg stop 会以"不是 agent 拉起的"拒绝关闭。"""
    prev = _read_endpoint_file() or {}
    same = prev.get("pid") and prev.get("pid") == health.get("pid")
    return {"url": url, "port": port, "pid": health.get("pid"),
            "out_dir": (prev.get("out_dir") if same else None) or _out_dir(),
            "started_by": (prev.get("started_by") if same else "external"),
            "version": health.get("version")}


def _free_port(preferred=START_PORTS):
    for p in preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _backend_python():
    """选一个能 import PIL 的解释器跑 app.py（超分链路的 PYSAFE_PYTHON 也取自它）。"""
    cands = []
    env = os.environ.get("MOSS_PYTHON")
    if env:
        cands.append(env)
    cands += [sys.executable,
              "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
              shutil.which("python3") or ""]
    for c in cands:
        if not c or not os.path.isfile(c):
            continue
        try:
            r = subprocess.run([c, "-c", "import PIL, sys; print('ok')"],
                               capture_output=True, timeout=20)
            if r.returncode == 0 and b"ok" in r.stdout:
                return c
        except Exception:
            continue
    raise MossError("E_BACKEND_DOWN", "找不到带 Pillow 的 python3 来启动后端",
                    "设置 MOSS_PYTHON=/path/to/python3")


def _out_dir(info=None):
    return (os.environ.get("MOSS_OUT")
            or (info or {}).get("out_dir")
            or DEFAULT_OUT)


def _start_backend(port=None, wait=120.0):
    """拉起一个 app.py 实例（独立会话，不随调用方退出）。返回 BackendInfo。"""
    if os.environ.get("MOSS_AGENT_AUTOSTART", "1") in ("0", "false", "False"):
        raise MossError("E_BACKEND_DOWN", "没有可用的 MOSS-Image 后端，且 MOSS_AGENT_AUTOSTART=0",
                        "先手动启动：cd ~/moss-image/web && python3 app.py")
    py = _backend_python()
    port = int(port or _free_port())
    _mkdirs()
    env = dict(os.environ)
    env["MOSS_PORT"] = str(port)
    env.setdefault("MOSS_OUT", _out_dir())
    # agent 拉起的实例空闲即退，避免长期占端口；正在跑任务时不会退（watchdog 会看 busy）
    env.setdefault("MOSS_IDLE_EXIT", os.environ.get("MOSS_AGENT_IDLE_EXIT", "1800"))
    env.pop("MOSS_PARENT_PID", None)
    log_path = os.path.join(LOG_DIR, "agentd-%d.log" % port)
    log = open(log_path, "ab")
    try:
        proc = subprocess.Popen([py, APP_PY], cwd=WEB, env=env,
                                stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    finally:
        log.close()
    url = "http://127.0.0.1:%d" % port
    deadline = time.time() + wait
    while time.time() < deadline:
        h = _health(url, timeout=2.0)
        if h:
            info = {"url": url, "port": port, "pid": h.get("pid") or proc.pid,
                    "out_dir": env["MOSS_OUT"], "started_at": _now(),
                    "started_by": "agent", "log": log_path,
                    "version": h.get("version")}
            _write_endpoint_file(info)
            return info
        if proc.poll() is not None:
            tail = ""
            with contextlib.suppress(Exception):
                with open(log_path, "rb") as f:
                    tail = f.read()[-800:].decode("utf-8", "replace")
            raise MossError("E_BACKEND_DOWN", "后端启动后立刻退出（rc=%s）" % proc.returncode,
                            "看日志：%s" % log_path, tail)
        time.sleep(0.4)
    with contextlib.suppress(Exception):
        proc.kill()
    raise MossError("E_BACKEND_START_TIMEOUT", "等待后端健康检查超时（%.0fs）" % wait,
                    "看日志：%s" % log_path)


def backend(autostart=True, wait=120.0):
    """返回可用后端信息：{url, pid, out_dir, ...}。优先复用，找不到再拉起。"""
    forced = os.environ.get("MOSS_AGENT_URL") or os.environ.get("MOSS_URL")
    if forced:
        h = _health(forced)
        if not h:
            raise MossError("E_BACKEND_DOWN", "MOSS_AGENT_URL 指向的后端不健康：%s" % forced,
                            "确认 app.py 在跑，或去掉该环境变量让它自动发现")
        info = _read_endpoint_file() or {}
        info.update({"url": forced.rstrip("/"), "pid": h.get("pid"),
                     "out_dir": info.get("out_dir") or _out_dir()})
        return info

    ep = _read_endpoint_file()
    if ep and ep.get("url") and _health(ep["url"]):
        ep["out_dir"] = ep.get("out_dir") or _out_dir()
        return ep
    if ep and ep.get("url"):
        with contextlib.suppress(Exception):
            os.remove(ENDPOINT_FILE)      # 过期端点，清掉

    for port in DISCOVERY_PORTS:
        url = "http://127.0.0.1:%d" % port
        h = _health(url, timeout=1.0)
        if h:
            info = _adopt(url, port, h)
            _write_endpoint_file(info)
            return info

    if not autostart:
        raise MossError("E_BACKEND_DOWN", "没有运行中的 MOSS-Image 后端",
                        "去掉 MOSS_AGENT_AUTOSTART=0 或手动启动 web/app.py")

    with _file_lock(START_LOCK):
        # 抢锁期间别的进程可能已经拉起来了
        ep = _read_endpoint_file()
        if ep and ep.get("url") and _health(ep["url"]):
            return ep
        for port in DISCOVERY_PORTS:
            url = "http://127.0.0.1:%d" % port
            h = _health(url, timeout=1.0)
            if h:
                info = _adopt(url, port, h)
                _write_endpoint_file(info)
                return info
        return _start_backend(wait=wait)


def backend_info():
    """后端概况（不抛异常，供 CLI status 用）。"""
    try:
        info = backend(autostart=False)
        h = _health(info["url"])
        return {"ok": True, "url": info["url"], "pid": info.get("pid"),
                "out_dir": _out_dir(info), "health": h}
    except MossError as e:
        return dict(e.to_dict(), backend_running=False)


def stop_backend(force=False):
    """关掉后端（需要本地 shutdown token；agent 自己拉起的实例才建议这么做）。"""
    info = backend(autostart=False)
    pid = info.get("pid")
    if not pid:
        raise MossError("E_BACKEND_DOWN", "拿不到后端 pid")
    if info.get("started_by") != "agent" and not force:
        raise MossError("E_VALIDATION", "这个后端不是 agent 拉起的（多半是桌面工作台）",
                        "别误关用户的界面：要强关传 force=True，或直接关工作台窗口")
    tok = ""
    for p in ("/tmp/moss-shutdown-%s.token" % pid, "/tmp/moss-shutdown.token"):
        with contextlib.suppress(Exception):
            with open(p, encoding="utf-8") as f:
                tok = f.read().strip()
            if tok:
                break
    if not tok:
        raise MossError("E_BACKEND_DOWN", "读不到 shutdown token，无法优雅关闭",
                        "直接 kill %s" % pid)
    req = urllib.request.Request(info["url"] + "/api/shutdown", method="GET",
                                 headers={"X-Moss-Token": tok})
    with urllib.request.urlopen(req, timeout=5) as r:
        r.read()
    with contextlib.suppress(Exception):
        os.remove(ENDPOINT_FILE)
    return True


# ── 能力发现 ──────────────────────────────────────────────────────────────
_caps_cache = {"ts": 0.0, "data": None}
_ver_cache = {"ts": 0.0, "data": None}


def _runtime_versions():
    """探测推理栈版本（mflux / mlx），用于把"同参数不同图"的隐藏变量显式化。"""
    if _ver_cache["data"] and _now() - _ver_cache["ts"] < 3600:
        return _ver_cache["data"]
    out = {"python": sys.version.split()[0]}
    # 开发版把 venv 叫 venv-mflux024，发布版叫 .venv-mflux；两个都探一下。
    py = next((c for c in (os.path.join(ROOT, ".venv-mflux", "bin", "python"),
                           os.path.join(ROOT, "venv-mflux024", "bin", "python"))
               if os.path.isfile(c)), None)
    if py:
        with contextlib.suppress(Exception):
            r = subprocess.run([py, "-c",
                                "from importlib.metadata import version as v;"
                                "print(v('mflux'), v('mlx'))"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.split()
                out["mflux"] = parts[0]
                if len(parts) > 1:
                    out["mlx"] = parts[1]
    _ver_cache.update({"ts": _now(), "data": out})
    return out


def capabilities(refresh=False, ttl=60.0):
    """机器可读的能力清单：引擎、档位、超分模型、角色 LoRA、尺寸规则、意图档位、错误码。"""
    if not refresh and _caps_cache["data"] and _now() - _caps_cache["ts"] < ttl:
        return _caps_cache["data"]
    info = backend()
    meta = _json_get(info["url"] + "/api/meta", timeout=10.0)
    h = _health(info["url"]) or {}
    data = {
        "ok": True,
        "client_version": __version__,
        "backend": {"url": info["url"], "pid": info.get("pid"),
                    "version": h.get("version"), "out_dir": _out_dir(info),
                    "running": h.get("running"), "queued": h.get("queued")},
        "engines": meta.get("engines", []),
        "lora_modes": meta.get("lora_modes", []),
        "default_engine": meta.get("default_engine"),
        "default_lora_mode": meta.get("default_lora"),
        "official_negative": meta.get("official_neg") or DEFAULT_NEGATIVE,
        "char_loras": meta.get("char_loras", []),
        "upscale_models": meta.get("upscale_models", []),
        "default_upscale": meta.get("default_upscale"),
        "presets": {k: dict(v, id=k) for k, v in PRESETS.items()},
        "size_rule": {"multiple_of": 16, "min": 256, "max": 4096,
                      "note": "宽高都必须是 16 的倍数；1MP 左右是模型的训练比例"},
        "steps_rule": {"min": 1, "max": 200,
                       "note": "不传则用档位默认；turbo 档忽略 steps/cfg（固定 15 次前向调度）"},
        "negative_rule": {"effective_when": "cfg != 1.0",
                          "note": "akk4 档 CFG=1.0，负面词完全不参与计算；turbo/native25/none/full 生效"},
        "prompt_rule": {"context_tokens": 4096, "recommended_chars": [1300, 2600],
                        "max_chars_warn": 3000,
                        "formats": ["caption", "layout_json"]},
        "error_codes": ERROR_CODES,
        # 版本三元组：复现时要对得上，否则"同参数"其实不同图
        "version": dict(_runtime_versions(),
                        client=__version__, app=h.get("version"),
                        model_weights="mlx_vi_8bit", quantize=8,
                        uncond_skip_layers=22),
        # 传了也不会生效的参数：与其让 agent 猜，不如显式列出来
        "unsupported": [
            {"param": "scheduler", "reason": "模型按训练调度（Flow-matching Euler）采样，传了被忽略"},
            {"param": "lora_style", "reason": "mflux 的 IGNORED_OPTIONS，传了被忽略"},
            {"param": "step_cache_ratio", "condition": "steps < 10", "reason": "步数太少，缓存无意义"},
            {"param": "two_stage", "reason": "引擎固有行为（mlx8 恒两阶段、msmlx 恒单进程），不接受外部开关"},
            {"param": "steps > 218", "reason": "mflux 硬边界（>218 直接报错）；本层已限制到 200"},
            {"param": "negative", "condition": "cfg == 1.0", "reason": "CFG=1.0 时负分支被跳过，负面词静默无效"},
            {"param": "char_lora", "condition": "lora_mode == turbo", "reason": "turbo 的调度来自 student 适配器，不叠角色 LoRA"},
        ],
        "notes": [
            "一次只跑一个任务（后端 GPU_LOCK 串行），排队上限默认 6。",
            "同 prompt + 同 seed + 同参数 = 同图，默认走幂等命中，不重复生成。",
            "产物是磁盘上的 PNG 绝对路径，另有同名 .json 参数快照；不要重新编码。",
        ],
    }
    _caps_cache.update({"ts": _now(), "data": data})
    return data


# ── 提示词校验 / 构造 ────────────────────────────────────────────────────
_LAYOUT_TOP = {"summary", "items", "bbox_coord_space", "bbox_format"}
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ]")


def _est_tokens(text):
    """粗估 token 数：CJK 约 1 字 1 token，其余按 4 字符 1 token。

    编码器上下文 4096 token（Qwen3.5 / Gated DeltaNet 训练长度），超了会退化。
    只做量级判断，不做精确分词。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return cjk + int(rest / 4) + 1


def validate_prompt(prompt, width=None, height=None, negative=None):
    """校验提示词（caption 或四键 Layout JSON），返回 {ok, kind, errors, warnings, stats}。

    后端不做任何提示词解析——JSON 写错键名不会报错，只会变成一段被模型忽略的废话。
    所以这一步是"正确调用"的关键：在提交前把结构性错误挑出来。
    """
    out = {"ok": True, "kind": "caption", "errors": [], "warnings": [], "stats": {}}
    if prompt is not None and not isinstance(prompt, str):
        out["ok"] = False
        out["errors"].append({"code": "E_VALIDATION",
                              "message": "prompt 必须是字符串（当前 %s）" % type(prompt).__name__,
                              "hint": "数字/对象先转成文本再传"})
        return out
    p = (prompt or "").strip()
    out["stats"]["chars"] = len(p)
    if not p:
        out["ok"] = False
        out["errors"].append({"code": "E_PROMPT_EMPTY", "message": "prompt 为空"})
        return out

    if p.startswith("["):
        # JSON 数组既不是 caption 也不是 Layout：别让它一路混到编码器里
        try:
            json.loads(p)
            out["ok"] = False
            out["errors"].append({
                "code": "E_BAD_LAYOUT_JSON",
                "message": "提示词以 [ 开头且是合法 JSON 数组；Layout 顶层必须是对象",
                "hint": "Layout 四键对象用 build_layout_json() 生成；纯 caption 不要以 [ 开头"})
            return out
        except Exception:
            pass                      # 不是合法 JSON，按普通 caption 处理
    if p.startswith("{"):
        out["kind"] = "layout_json"
        try:
            obj = json.loads(p)
        except Exception as e:
            out["ok"] = False
            out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                  "message": "JSON 解析失败：%s" % e,
                                  "hint": "常见原因：中文弯引号“”当引号、字符串没闭合、多余逗号"})
            return out
        if not isinstance(obj, dict):
            out["ok"] = False
            out["errors"].append({"code": "E_BAD_LAYOUT_JSON", "message": "顶层必须是对象"})
            return out
        extra = sorted(set(obj) - _LAYOUT_TOP)
        if extra:
            out["warnings"].append({"code": "W_LAYOUT_EXTRA_KEYS",
                                    "message": "多余的顶层键会被模型忽略：%s" % ", ".join(extra),
                                    "hint": "顶层只能是 summary/items/bbox_coord_space/bbox_format"})
        missing = sorted(_LAYOUT_TOP - set(obj))
        if missing:
            out["ok"] = False
            out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                  "message": "缺少顶层键：%s" % ", ".join(missing)})
            return out
        if obj.get("bbox_coord_space") != "abs_centralized":
            out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                  "message": "bbox_coord_space 必须是 abs_centralized"})
            out["ok"] = False
        if obj.get("bbox_format") != "yxyx":
            out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                  "message": "bbox_format 必须是 yxyx"})
            out["ok"] = False
        summary = obj.get("summary")
        if not isinstance(summary, dict) or not str(summary.get("caption") or "").strip():
            out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                  "message": "summary.caption 缺失或为空"})
            out["ok"] = False
        elif summary.get("canvas") not in (None, "single"):
            out["warnings"].append({"code": "W_LAYOUT_CANVAS",
                                    "message": "summary.canvas 应为 single（当前 %r）" % summary.get("canvas")})
        # 官方 compile_prompt.py 的 summary 允许键：caption / meta / canvas / palette
        _summary_extra = sorted(set(summary) - {"caption", "meta", "canvas", "palette"}) if isinstance(summary, dict) else []
        if _summary_extra:
            out["warnings"].append({"code": "W_LAYOUT_SUMMARY_KEYS",
                                    "message": "summary 里多余的键会被忽略：%s" % ", ".join(_summary_extra),
                                    "hint": "summary 只认 caption / meta / canvas / palette"})
        if isinstance(summary, dict) and summary.get("canvas") not in (None, "single", "multi_panel", "collage"):
            out["warnings"].append({"code": "W_LAYOUT_CANVAS",
                                    "message": "summary.canvas 应为 single / multi_panel / collage（当前 %r）"
                                               % summary.get("canvas")})
        items = obj.get("items")
        if not isinstance(items, list) or not items:
            out["errors"].append({"code": "E_BAD_LAYOUT_JSON", "message": "items 必须是非空数组"})
            out["ok"] = False
            items = []
        out["stats"]["items"] = len(items)
        if items and not (4 <= len(items) <= 7):
            out["warnings"].append({"code": "W_LAYOUT_ITEMS",
                                    "message": "items 建议 5–6 个（当前 %d）" % len(items)})
        half_w = (width / 2.0) if width else None
        half_h = (height / 2.0) if height else None
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                      "message": "items[%d] 不是对象" % i})
                out["ok"] = False
                continue
            for k in ("label", "caption", "bbox"):
                if k not in it:
                    out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                          "message": "items[%d] 缺 %s" % (i, k),
                                          "path": "items[%d].%s" % (i, k)})
                    out["ok"] = False
            # 官方 label 枚举：object / character / text；text 项必须给 text
            lab = it.get("label")
            if lab is not None and lab not in ("object", "character", "text"):
                out["warnings"].append({
                    "code": "W_LAYOUT_LABEL",
                    "message": "items[%d].label=%r 不在官方枚举 object/character/text 里" % (i, lab),
                    "hint": "官方 compile_prompt.py 会拒绝未知 label；模型侧也可能弱化该区域",
                    "path": "items[%d].label" % i})
            if lab == "text" and not str(it.get("text") or "").strip():
                out["errors"].append({
                    "code": "E_BAD_LAYOUT_JSON",
                    "message": "items[%d] 是 text 项但没有 text 字段（要渲染的字符）" % i,
                    "hint": '补 "text": "要出现的字"，换行用 \\n',
                    "path": "items[%d].text" % i})
                out["ok"] = False
            _item_extra = sorted(set(it) - {"label", "caption", "bbox", "text", "tags", "palette"})
            if _item_extra:
                out["warnings"].append({
                    "code": "W_LAYOUT_ITEM_KEYS",
                    "message": "items[%d] 多余的键会被忽略：%s" % (i, ", ".join(_item_extra)),
                    "hint": "item 只认 label / caption / bbox / text / tags / palette"})
            bb = it.get("bbox")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4
                    and all(isinstance(v, (int, float)) for v in bb)):
                out["errors"].append({"code": "E_BAD_LAYOUT_JSON",
                                      "message": "items[%d].bbox 必须是 4 个数字 [y1,x1,y2,x2]" % i})
                out["ok"] = False
                continue
            y1, x1, y2, x2 = bb
            if y1 > y2 or x1 > x2:
                out["warnings"].append({"code": "W_LAYOUT_BBOX_ORDER",
                                        "message": "items[%d].bbox 看起来反了（应为 y1<=y2, x1<=x2）" % i})
            if half_w and half_h:
                if max(abs(x1), abs(x2)) > half_w * 1.02 or max(abs(y1), abs(y2)) > half_h * 1.02:
                    out["warnings"].append({
                        "code": "W_LAYOUT_BBOX_RANGE",
                        "message": "items[%d].bbox 超出画布中心坐标系（±%d, ±%d）"
                                   % (i, int(half_w), int(half_h)),
                        "hint": "bbox 相对画布中心，单位像素；换尺寸后要按新画布等比缩放"})
        if "negative" in obj:
            out["warnings"].append({"code": "W_LAYOUT_NEGATIVE",
                                    "message": "negative 不能写进 Layout JSON，必须作为独立参数传"})
    else:
        n = out["stats"]["chars"]
        est = _est_tokens(p)
        out["stats"]["est_tokens"] = est
        if est > 4096:
            out["ok"] = False
            out["errors"].append({
                "code": "E_PROMPT_TOO_LONG",
                "message": "prompt 约 %d token，超出编码器 4096 上下文" % est,
                "hint": "压到 2600 字符左右（CJK 约 1 字 1 token）；关键内容别放最后"})
        elif est > 3200:
            out["warnings"].append({
                "code": "W_PROMPT_LONG",
                "message": "prompt 约 %d token，接近 4096 上限，可能被截断" % est,
                "hint": "建议 1300–2600 字符；关键内容别放最后"})
        if n < 40:
            out["warnings"].append({"code": "W_PROMPT_SHORT",
                                    "message": "prompt 只有 %d 字符，画面容易失控" % n,
                                    "hint": "补：风格/场景/主体外形/服装道具/动作表情/材质/光线"})

    # 尺寸也是"提交前就该知道"的事：后端只回一句 "16 的倍数"
    for label, val in (("width", width), ("height", height)):
        if val is None:
            continue
        try:
            iv = int(val)
        except (TypeError, ValueError):
            out["ok"] = False
            out["errors"].append({"code": "E_SIZE", "message": "%s 不是整数：%r" % (label, val)})
            continue
        if not _size_ok(iv):
            near = max(256, min(4096, int(round(iv / 16.0)) * 16))
            out["ok"] = False
            out["errors"].append({"code": "E_SIZE",
                                  "message": "%s=%d 不合法（必须是 16 的倍数，256..4096）" % (label, iv),
                                  "hint": "最近的合法值：%d" % near})
    return out


def build_layout_json(summary_caption, items, width, height,
                      canvas="single", validate=True):
    """把简单结构编成官方四键 Layout JSON，避免 agent 手写 JSON 时键名/坐标写错。

    items: [{"label": "character", "caption": "...", "bbox": [y1,x1,y2,x2]}, ...]
           bbox 可省略——省略时按 items 顺序均分画面（竖直方向）。
    """
    w, h = float(width), float(height)
    n = max(1, len(items or []))
    out_items = []
    for i, it in enumerate(items or []):
        bb = it.get("bbox")
        if not bb:
            top = -h / 2.0 + (h * i / n)
            bot = -h / 2.0 + (h * (i + 1) / n)
            bb = [round(top), round(-w / 2.0), round(bot), round(w / 2.0)]
        item = {"label": str(it.get("label") or "object"),
                "caption": str(it.get("caption") or ""),
                "bbox": [int(round(v)) for v in bb]}
        if it.get("text"):
            item["text"] = str(it["text"])
        if it.get("tags"):
            item["tags"] = it["tags"]
        out_items.append(item)
    obj = {"summary": {"caption": str(summary_caption or ""), "canvas": canvas},
           "items": out_items,
           "bbox_coord_space": "abs_centralized",
           "bbox_format": "yxyx"}
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if validate:
        r = validate_prompt(text, width=width, height=height)
        if not r["ok"]:
            raise ValidationError("生成的 Layout JSON 未通过校验",
                                  code="E_BAD_LAYOUT_JSON", detail=r)
    return text


# ── 参数规范化 ────────────────────────────────────────────────────────────
def _size_ok(v):
    return isinstance(v, int) and v % 16 == 0 and 256 <= v <= 4096


def normalize_params(prompt, negative=None, preset=None, width=None, height=None,
                     steps=None, cfg=None, seed=None, engine=None, lora_mode=None,
                     char_lora=None, char_lora_scale=None, enhance=None,
                     skip=None, stream=None, caps=None):
    """把 agent 的意图解析成后端认的完整参数（含档位默认值），并做本地校验。"""
    caps = caps or capabilities()
    modes = {m["id"]: m for m in caps.get("lora_modes", [])}
    engines = {e["id"]: e for e in caps.get("engines", [])}
    _steps_in, _cfg_in = steps, cfg      # 记住"用户显式传了吗"，用于警告文案
    _preset_given = bool(preset)

    if preset:
        if preset not in PRESETS:
            raise ValidationError("未知档位 %r" % preset, code="E_UNKNOWN_PRESET",
                                  hint="可用：%s" % ", ".join(sorted(PRESETS)))
        pv = PRESETS[preset]
        engine = engine or pv["engine"]
        lora_mode = lora_mode or pv["lora_mode"]
        width = width if width is not None else pv["width"]
        height = height if height is not None else pv["height"]
        steps = steps if steps is not None else pv["steps"]
        cfg = cfg if cfg is not None else pv["cfg"]

    if not preset and not lora_mode:
        # 没给档位就是"还不知道要什么"：默认走最便宜的草稿档，别一上来烧 1MP。
        preset = "draft"
        pv = PRESETS[preset]
        engine = engine or pv["engine"]
        lora_mode = pv["lora_mode"]
        width = width if width is not None else pv["width"]
        height = height if height is not None else pv["height"]
    lora_mode = lora_mode or caps.get("default_lora_mode") or "akk4"
    if lora_mode not in modes:
        raise ValidationError("未知档位 %r" % lora_mode, code="E_UNKNOWN_LORA_MODE",
                              hint="可用：%s" % ", ".join(sorted(modes)))
    mode = modes[lora_mode]

    # 档位可以绑定引擎（turbo 的调度只存在于 moss-image-mlx）
    bound = mode.get("engine")
    engine = bound or engine or caps.get("default_engine") or "mlx8"
    if engine not in engines:
        raise ValidationError("未知引擎 %r" % engine, code="E_UNKNOWN_ENGINE",
                              hint="可用：%s" % ", ".join(sorted(engines)))
    if not engines[engine].get("available", True):
        raise ValidationError(engines[engine].get("note") or "引擎不可用",
                              code="E_ENGINE_UNAVAILABLE",
                              hint="换 engine=mlx8 或补 venv")

    width = int(width if width is not None else 1184)
    height = int(height if height is not None else 896)
    if not _size_ok(width) or not _size_ok(height):
        near_w = max(256, min(4096, int(round(width / 16.0)) * 16))
        near_h = max(256, min(4096, int(round(height / 16.0)) * 16))
        raise ValidationError("width=%s height=%s 不合法" % (width, height), code="E_SIZE",
                              hint="取 16 的倍数且在 256..4096，例如 %dx%d" % (near_w, near_h))

    try:
        steps = int(steps if steps is not None else mode.get("steps") or 20)
    except (TypeError, ValueError):
        raise ValidationError("steps 必须是整数（当前 %r）" % (steps,), code="E_VALIDATION")
    if not (1 <= steps <= 200):
        raise ValidationError("steps 必须在 1..200（当前 %d）" % steps, code="E_VALIDATION")
    try:
        cfg = float(cfg if cfg is not None else mode.get("cfg") or 3.0)
    except (TypeError, ValueError):
        raise ValidationError("cfg 必须是数字（当前 %r）" % (cfg,), code="E_VALIDATION")
    if not (0.0 <= cfg <= 20.0):
        raise ValidationError("cfg 必须在 0..20（当前 %s）" % cfg, code="E_VALIDATION")

    warnings = []
    if not _preset_given and not _steps_in and not _cfg_in and width <= 640 and height <= 640:
        warnings.append({"code": "W_DRAFT_DEFAULT",
                         "message": "未指定 preset，已按 draft（512² 草稿档）出图",
                         "hint": "要成品请显式传 preset=standard / quality / precise"})
    if seed is None or seed == "" or seed == "random":
        seed = "random"
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValidationError("seed 必须是整数或 'random'", code="E_VALIDATION")
        if not (0 <= seed < 2 ** 31):
            raise ValidationError("seed 必须在 0..2^31-1", code="E_VALIDATION")

    neg = (negative or "").strip()
    if neg and cfg == 1.0:
        warnings.append({"code": "W_NEG_IGNORED",
                         "message": "档位 %s 的 CFG=1.0，负面词不参与计算（等于没填）" % lora_mode,
                         "hint": "要负面词生效请用 quality(precise) 档（CFG 3.0）"})
    if not neg and cfg != 1.0:
        neg = caps.get("official_negative") or DEFAULT_NEGATIVE
        warnings.append({"code": "W_NEG_DEFAULT",
                         "message": "未传 negative，已使用官方默认负面词",
                         "detail": neg})

    if mode.get("turbo") and (_steps_in is not None or _cfg_in is not None):
        warnings.append({"code": "W_TURBO_IGNORES_STEPS",
                         "message": "turbo 档忽略 steps/cfg（固定 15 次前向调度），传了也不生效"})

    char_lora = (char_lora or "").strip()
    if char_lora:
        known = {c if isinstance(c, str) else (c.get("id") or c.get("name"))
                 for c in caps.get("char_loras", [])}
        known = {x for x in known if x}
        if char_lora not in known:
            raise ValidationError("未知的角色 LoRA %r" % char_lora, code="E_UNKNOWN_LORA",
                                  hint="可用：%s" % (", ".join(sorted(known)) or "（models/loras/ 为空）"))
        if mode.get("turbo"):
            raise ValidationError("turbo 档不支持叠加角色 LoRA", code="E_LORA_UNSUPPORTED",
                                  hint="换 standard/draft/precise 档")
    if char_lora_scale is None:
        char_lora_scale = 1.0
    else:
        char_lora_scale = float(char_lora_scale)
        if not (0.0 <= char_lora_scale <= 2.0):
            raise ValidationError("char_lora_scale 必须在 0..2", code="E_VALIDATION")

    params = {
        "prompt": (prompt or "").strip(),
        "negative": neg,
        "preset": preset,
        "engine": engine,
        "lora_mode": lora_mode,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "char_lora": char_lora or None,
        "char_lora_scale": char_lora_scale if char_lora else None,
        "enhance": bool(enhance),
        "skip": skip,
        "stream": stream,
    }
    if not params["prompt"]:
        raise ValidationError("prompt 为空", code="E_PROMPT_EMPTY")
    return params, warnings


def _idem_key(params):
    """幂等键：只看真正影响像素的参数（跳过 enhance/stream/skip 这些流程开关）。"""
    core = {k: params.get(k) for k in
            ("prompt", "negative", "engine", "lora_mode", "width", "height",
             "steps", "cfg", "seed", "char_lora", "char_lora_scale")}
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def estimate_eta(params):
    """粗估耗时（秒）。优先用本地台账的实测中位数，没有才按每百万像素折算。"""
    key = (params.get("engine"), params.get("lora_mode"))
    samples = _ledger_etas(key, params.get("width"), params.get("height"))
    if samples:
        med = sorted(samples)[len(samples) // 2]
        return {"seconds": round(med, 1), "source": "ledger", "is_estimate": False,
                "samples": len(samples)}
    per_mp = _SEC_PER_MP.get(key)
    mp = (params.get("width", 0) * params.get("height", 0)) / 1e6
    if per_mp and mp:
        sec = per_mp * mp
        if params.get("lora_mode") == "akk4" and params.get("steps"):
            sec *= params["steps"] / 4.0
        return {"seconds": round(sec, 1), "source": "calibration", "is_estimate": True}
    return {"seconds": None, "source": "unknown", "is_estimate": True}


def _ledger_etas(key, w, h):
    out = []
    for e in _ledger_read():
        if e.get("kind") != "generate" or e.get("status") != "done":
            continue
        pr = e.get("params") or {}
        if (pr.get("engine"), pr.get("lora_mode")) != key:
            continue
        if pr.get("width") != w or pr.get("height") != h:
            continue
        if e.get("elapsed"):
            out.append(float(e["elapsed"]))
    return out[-9:]


# ── 台账（幂等 / 复现 / ETA）───────────────────────────────────────────────
def _ledger_read(limit=5000):
    entries = []
    try:
        with open(LEDGER_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return entries[-limit:]


def _fingerprint(caps=None):
    """推理栈指纹：版本/量化一变，同参数就不再是同一张图，幂等命中必须失效。"""
    caps = caps or capabilities()
    v = caps.get("version") or {}
    blob = "|".join(str(v.get(k)) for k in
                    ("app", "mflux", "mlx", "quantize", "uncond_skip_layers"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _ledger_append(entry):
    _mkdirs()
    entry = dict(entry, ts=_now())
    with _file_lock(LEDGER_FILE + ".lock", timeout=20):
        with open(LEDGER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _ledger_maybe_compact()


def _ledger_maybe_compact():
    """台账只增不减会拖慢每次查找；超过阈值就只留最近 3000 条（调用方已持锁）。"""
    try:
        if os.path.getsize(LEDGER_FILE) <= LEDGER_MAX_BYTES:
            return
        keep = _ledger_read(3000)
        tmp = LEDGER_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in keep:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, LEDGER_FILE)
    except Exception:
        pass


def _ledger_find(key, kind="generate", fp=None, ttl=None):
    """返回该幂等键最近一次**仍然有效**的完成条目。

    有效性 = 产物还在 + 指纹一致（版本/量化没变） + 没过期。
    返回 (entry, stale_reason)：stale_reason 非空表示"曾经命中但已失效"。
    """
    hit, stale = None, None
    ttl = LEDGER_TTL if ttl is None else ttl
    for e in _ledger_read():
        if e.get("kind") != kind or e.get("key") != key:
            continue
        if e.get("status") != "done" or not e.get("path") or not os.path.isfile(e["path"]):
            continue
        if fp and e.get("fp") and e["fp"] != fp:
            stale = "模型/引擎版本已变（%s → %s）" % (e.get("fp"), fp)
            continue
        if ttl > 0 and e.get("ts") and (_now() - float(e["ts"])) > ttl:
            stale = "台账条目已过期（>%.0f 天）" % (ttl / 86400.0)
            continue
        hit = e
    return hit, stale


def _ledger_find_inflight(key, ttl=None):
    """找一个"正在跑"的提交记录。超过 TTL 的孤儿行不再挂接（后端早就 prune 掉了）。"""
    ttl = INFLIGHT_TTL if ttl is None else ttl
    for e in reversed(_ledger_read(500)):
        if e.get("kind") != "generate" or e.get("key") != key:
            continue
        if e.get("status") != "submitted":
            continue
        if ttl > 0 and e.get("ts") and (_now() - float(e["ts"])) > ttl:
            continue
        return e
    return None


# ── 生成 ──────────────────────────────────────────────────────────────────
def _resolve_out_path(filename, info):
    cands = []
    for d in (os.environ.get("MOSS_OUT"), (info or {}).get("out_dir"), DEFAULT_OUT):
        if d:
            cands.append(os.path.join(d, filename))
    for p in cands:
        if os.path.isfile(p):
            return p
    # 超分产物在 upscaled/ 子目录
    for d in (os.environ.get("MOSS_OUT"), (info or {}).get("out_dir"), DEFAULT_OUT):
        if d:
            p = os.path.join(d, "upscaled", filename)
            if os.path.isfile(p):
                return p
    return None


def _read_meta(path):
    try:
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _settle(path, max_wait=None):
    """等产物大小稳定再算哈希。

    后端在"任务完成"之后会用 oxipng 做一次**无损**重压（8192² 要 70–80s），
    期间文件会被原地替换。若我们在重压前算 sha256，事后就对不上。
    这里按文件大小连续两次不变判定稳定；大图给更长的等待窗口。
    """
    try:
        size0 = os.path.getsize(path)
        px = 0
        with contextlib.suppress(Exception):
            from PIL import Image
            with Image.open(path) as im:
                px = im.size[0] * im.size[1]
        budget = max_wait if max_wait is not None else min(10.0, 1.5 + px / 4e6)
        deadline = time.time() + budget
        prev = size0
        stable = 0
        while time.time() < deadline:
            time.sleep(0.4)
            cur = os.path.getsize(path)
            if cur == prev:
                stable += 1
                if stable >= 3:
                    return True
            else:
                stable = 0
                prev = cur
        return False
    except Exception:
        return False


def _result(path, job, params, info, warnings=None, cached=False, identical=None):
    if not cached:
        _settle(path)          # 只有刚生成的产物才需要等 oxipng 落定；缓存命中不必再等
    st = os.stat(path)
    out_root = os.path.abspath(_out_dir(info))
    up_root = os.path.join(out_root, "upscaled")
    abspath = os.path.abspath(path)
    if abspath.startswith(up_root + os.sep):
        rel = "upscaled/" + os.path.basename(path)
    else:
        rel = os.path.basename(path)
    r = {
        "ok": True,
        "path": abspath,
        "filename": os.path.basename(path),
        "url": "%s/out/%s" % (info.get("url", ""), urllib.parse.quote(rel)),
        "bytes": st.st_size,
        "sha256": _sha256(path),
        "mtime": int(st.st_mtime),
        "width": (job or {}).get("width"),
        "height": (job or {}).get("height"),
        "elapsed": 0.0 if cached else (job or {}).get("elapsed"),
        "elapsed_original": (job or {}).get("elapsed"),
        "job_id": (job or {}).get("id"),
        "params": params,
        "meta": _read_meta(path),
        "warnings": warnings or [],
        "cached": bool(cached),
        "backend": {"url": info.get("url"), "pid": info.get("pid")},
    }
    if identical is not None:
        r["identical"] = identical
    return r


def _job_wait(info, job_id, timeout, on_progress=None):
    url = info["url"] + "/api/job?id=" + urllib.parse.quote(job_id)
    deadline = time.time() + float(timeout)
    last = None
    while True:
        try:
            j = _json_get(url, timeout=15.0, ok_404=True)
        except MossError as e:
            if e.code == "E_BACKEND_DOWN" or not _health(info["url"], timeout=2.0):
                raise MossError("E_BACKEND_DOWN", "等待任务期间后端失联",
                                "重启后端后 job 可能已丢失")
            raise
        if isinstance(j, dict) and j.get("error") == "not found":
            raise MossError("E_JOB_FAILED", "后端不认识该任务（可能已重启）",
                            "别重试同一个 job_id：用 generate() 重新提交，或用 list_outputs() 找已有产物",
                            detail=job_id)
        if j:
            if on_progress and (j.get("progress"), j.get("phase")) != last:
                last = (j.get("progress"), j.get("phase"))
                with contextlib.suppress(Exception):
                    on_progress(j)
            st = j.get("status")
            if st == "done":
                return j
            if st == "error":
                if j.get("cancelled"):
                    raise MossError("E_CANCELLED", "任务已取消", detail=j.get("error"))
                raise MossError("E_JOB_FAILED", "生成失败", detail=j.get("error"))
        if time.time() > deadline:
            e = MossError("E_JOB_TIMEOUT", "等待超时（%.0fs）" % timeout,
                          "job_id 仍有效：wait_job(job_id) 可续等，或 cancel(job_id)")
            e.job_id = job_id
            raise e
        time.sleep(POLL_INTERVAL)


def wait_job(job_id, timeout=DEFAULT_TIMEOUT, on_progress=None):
    """续等一个已知 job_id（E_JOB_TIMEOUT 之后用它接着等）。"""
    info = backend()
    j = _job_wait(info, job_id, timeout, on_progress)
    fn = j.get("output") or ""
    path = _resolve_out_path(fn, info)
    if not path:
        raise MossError("E_OUTPUT_MISSING", "任务成功但找不到产物 %r" % fn,
                        "确认 MOSS_OUT 与后端一致（capabilities().backend.out_dir）")
    params = {k: j.get(k) for k in
              ("prompt", "negative", "engine", "lora_mode", "width", "height",
               "steps", "cfg", "seed", "seed_requested", "char_lora",
               "char_lora_scale", "skip", "skip_mode") if k in j}
    return _result(path, j, params, info)


def _submit(params, info):
    """把参数交给后端，返回 (job_id, 响应)。"""
    body = {
        "prompt": params["prompt"],
        "negative": params["negative"],
        "width": params["width"],
        "height": params["height"],
        "steps": params["steps"],
        "cfg": params["cfg"],
        "engine": params["engine"],
        "lora_mode": params["lora_mode"],
        "seed": params["seed"],
        "enhance": bool(params["enhance"]),
    }
    if params["char_lora"]:
        body["char_lora"] = params["char_lora"]
        body["char_lora_scale"] = params["char_lora_scale"]
    if params["skip"] is not None:
        body["skip"] = params["skip"]
    if params["stream"] is not None:
        body["stream"] = bool(params["stream"])
    resp = _json_post(info["url"] + "/api/generate", body, timeout=60.0)
    jid = resp.get("id")
    if not jid:
        raise MossError("E_HTTP", "后端没返回 job id", detail=resp)
    params["filename"] = resp.get("filename")
    params["seed_requested"] = "random" if params["seed"] == "random" else params["seed"]
    return jid, resp


def _finish(jid, params, key, fp, info, warnings, timeout, on_progress, resp=None):
    """等任务结束、定位产物、写台账。"""
    try:
        j = _job_wait(info, jid, timeout, on_progress)
    except MossError as e:
        if e.code == "E_JOB_FAILED":
            _ledger_append({"kind": "generate", "key": key, "status": "error",
                            "job_id": jid, "params": params, "error": e.detail, "fp": fp})
        raise
    fn = j.get("output") or (resp or {}).get("filename")
    path = _resolve_out_path(fn, info)
    if not path:
        raise MossError("E_OUTPUT_MISSING", "任务成功但找不到产物 %r" % fn,
                        "确认 MOSS_OUT 与后端一致（capabilities().backend.out_dir）")
    res = _result(path, j, params, info, warnings)
    _ledger_append({"kind": "generate", "key": key, "status": "done", "job_id": jid,
                    "path": res["path"], "params": params,
                    "elapsed": j.get("elapsed"), "width": res["width"],
                    "height": res["height"], "sha256": res["sha256"], "fp": fp})
    return res


def generate(prompt, negative=None, preset=None, width=None, height=None, steps=None,
             cfg=None, seed=None, engine=None, lora_mode=None, char_lora=None,
             char_lora_scale=None, enhance=None, skip=None, stream=None,
             timeout=DEFAULT_TIMEOUT, idempotency_key=None, dedupe=True,
             on_progress=None, _validate=True):
    """同步生成一张图，返回 Result dict（path 是绝对路径，meta 是参数快照）。

    dedupe=True（默认）：同 prompt + 同 seed + 同参数 命中台账里已完成的图，直接返回；
    若同参数任务正在跑，则挂上去一起等，不会重复烧一次 GPU。

    并发说明：命中检查与"提交"在同一把跨进程锁里完成（毫秒级），**等待任务在锁外**，
    所以多个 agent 同时调用不会互相阻塞整个推理时长。
    """
    # 先做不需要后端的校验：后端挂了也要能报出"提示词写错了"，而不是一律 E_BACKEND_DOWN
    vr = None
    if _validate:
        vr = validate_prompt(prompt, width=width, height=height, negative=negative)
        if not vr["ok"]:
            first = vr["errors"][0]
            raise ValidationError(first.get("message", "prompt 不合法"),
                                  code=first.get("code", "E_VALIDATION"),
                                  hint=first.get("hint", ""), detail=vr)
    caps = capabilities()
    params, warnings = normalize_params(
        prompt, negative=negative, preset=preset, width=width, height=height,
        steps=steps, cfg=cfg, seed=seed, engine=engine, lora_mode=lora_mode,
        char_lora=char_lora, char_lora_scale=char_lora_scale, enhance=enhance,
        skip=skip, stream=stream, caps=caps)
    if vr and vr.get("warnings"):
        # validate_prompt 的结构性提醒（太长/太短/键名多余…）也要进结果，否则 agent 看不到
        warnings = list(vr["warnings"]) + list(warnings)
    if params["seed"] == "random":
        dedupe = False            # 随机种子没有幂等语义
        key = None
    else:
        key = idempotency_key or _idem_key(params)

    info = backend()
    fp = _fingerprint(caps) if key else None

    if not (dedupe and key):
        jid, resp = _submit(params, info)
        return _finish(jid, params, key, fp, info, warnings, timeout, on_progress, resp)

    attach, attach_params = None, None
    with _file_lock(GEN_LOCK, timeout=120):
        hit, stale = _ledger_find(key, fp=fp)
        if hit:
            job_like = {"id": hit.get("job_id"), "elapsed": hit.get("elapsed"),
                        "width": hit.get("width"), "height": hit.get("height")}
            return _result(hit["path"], job_like, hit.get("params") or params,
                           info, warnings, cached=True)
        if stale:
            warnings.append({"code": "W_STALE_LEDGER",
                             "message": "台账里有同参数旧图但已失效：%s" % stale,
                             "hint": "本次重新生成（旧图仍在原路径）"})
        inflight = _ledger_find_inflight(key)
        if inflight and inflight.get("job_id"):
            attach = inflight["job_id"]
            attach_params = inflight.get("params") or params
        else:
            jid, resp = _submit(params, info)
            _ledger_append({"kind": "generate", "key": key, "status": "submitted",
                            "job_id": jid, "params": params, "fp": fp})

    if attach:
        try:
            j = _job_wait(info, attach, timeout, on_progress)
            fn = j.get("output") or ""
            path = _resolve_out_path(fn, info)
            if path:
                res = _result(path, j, attach_params, info, warnings)
                _ledger_append({"kind": "generate", "key": key, "status": "done",
                                "job_id": attach, "path": res["path"],
                                "params": attach_params, "elapsed": j.get("elapsed"),
                                "width": res["width"], "height": res["height"],
                                "sha256": res["sha256"], "fp": fp})
                return res
        except MossError:
            # 挂不上（job 已被后端 prune / 后端重启过）。先回查台账：同 key 的产物
            # 可能已经落盘了，这时候重跑就是白烧一次 GPU。
            hit2, _ = _ledger_find(key, fp=fp)
            if hit2:
                job_like = {"id": hit2.get("job_id"), "elapsed": hit2.get("elapsed"),
                            "width": hit2.get("width"), "height": hit2.get("height")}
                return _result(hit2["path"], job_like, hit2.get("params") or params,
                               info, warnings, cached=True)
        jid, resp = _submit(params, info)
        _ledger_append({"kind": "generate", "key": key, "status": "submitted",
                        "job_id": jid, "params": params, "fp": fp})
        return _finish(jid, params, key, fp, info, warnings, timeout, on_progress, resp)

    return _finish(jid, params, key, fp, info, warnings, timeout, on_progress, resp)


def generate_async(prompt, **kw):
    """提交但不等待，返回 job_id（长任务/并发场景用；注意后端是串行的）。"""
    kw["timeout"] = 0.01
    try:
        generate(prompt, **kw)
    except MossError as e:
        if e.code == "E_JOB_TIMEOUT" and getattr(e, "job_id", None):
            return e.job_id
        raise
    raise MossError("E_HTTP", "任务在 10ms 内完成，请用 generate() 取结果")


def generate_batch(prompt, n=4, seeds=None, seed_base=None, on_progress=None, **kw):
    """同 prompt 生成 n 个变体（不同 seed），串行执行并逐个回调。"""
    n = int(n)
    if n < 1 or n > 16:
        raise ValidationError("n 必须在 1..16", code="E_VALIDATION")
    out = []
    for i in range(n):
        if seeds:
            seed = seeds[i]
        elif seed_base is not None:
            seed = int(seed_base) + i
        else:
            seed = "random"
        if on_progress:
            with contextlib.suppress(Exception):
                on_progress({"phase": "batch", "index": i, "total": n})
        out.append(generate(prompt, seed=seed, on_progress=on_progress, **kw))
    return out


def job_status(job_id):
    info = backend()
    return _json_get(info["url"] + "/api/job?id=" + urllib.parse.quote(job_id))


def cancel(job_id):
    info = backend()
    return _json_post(info["url"] + "/api/cancel", {"id": job_id}, timeout=15.0)


def current_jobs():
    info = backend()
    return _json_get(info["url"] + "/api/current", timeout=10.0).get("jobs", [])


# ── 产物：查看 / 复现 / 变体 / 超分 / 列表 ─────────────────────────────────
def inspect(path_or_name):
    """读一张图的参数快照（无损复现的依据）。"""
    path = path_or_name
    if not os.path.isfile(path):
        name = os.path.basename(path_or_name)
        path = _resolve_out_path(name, {})
        if not path:
            # 后端可能在别的 MOSS_OUT 上跑，问一下它
            with contextlib.suppress(MossError):
                path = _resolve_out_path(name, backend(autostart=False))
        if not path:
            raise ValidationError("找不到图片 %r" % path_or_name, code="E_OUTPUT_MISSING",
                                  hint="传绝对路径，或 list_outputs() 里的文件名")
    st = os.stat(path)
    meta = _read_meta(path)
    return {"ok": True, "path": os.path.abspath(path), "bytes": st.st_size,
            "sha256": _sha256(path), "mtime": int(st.st_mtime), "meta": meta,
            "reproducible": bool(meta and meta.get("seed") is not None
                                 and meta.get("prompt"))}


def replay(path_or_name, overrides=None, timeout=DEFAULT_TIMEOUT, compare=True,
           on_progress=None):
    """按快照原样重跑（同 prompt/seed/参数），默认比对 sha256 报告是否逐字节一致。"""
    src = inspect(path_or_name)
    meta = src.get("meta") or {}
    if not meta.get("prompt"):
        raise ValidationError("该图没有可用的参数快照，无法复现", code="E_VALIDATION",
                              hint="只有工作台生成的图才有同名 .json")
    kw = {
        "negative": meta.get("negative") or "",
        "width": meta.get("width"),
        "height": meta.get("height"),
        "steps": meta.get("steps"),
        "cfg": meta.get("cfg"),
        "seed": meta.get("seed"),
        "engine": meta.get("engine"),
        "lora_mode": meta.get("lora_mode"),
        "char_lora": meta.get("char_lora"),
        "char_lora_scale": meta.get("char_lora_scale"),
        "enhance": False,
    }
    if overrides:
        kw.update(overrides)
    res = generate(meta["prompt"], dedupe=False, timeout=timeout,
                   on_progress=on_progress, _validate=False, **kw)
    res["replayed_from"] = src["path"]
    if compare:
        res["identical"] = (res["sha256"] == src["sha256"])
        if not res["identical"]:
            res["warnings"] = res.get("warnings", []) + [{
                "code": "W_NOT_IDENTICAL",
                "message": "重跑结果与原图不一致（模型/引擎/跳步参数可能已变）",
                "hint": "看两张图的 params：steps/cfg/engine/skip 是否一致"}]
    return res


def vary(path_or_name, seeds=None, n=1, timeout=DEFAULT_TIMEOUT, on_progress=None):
    """同 prompt/参数换 seed 出变体（保持构图意图、换采样起点）。"""
    src = inspect(path_or_name)
    meta = src.get("meta") or {}
    if not meta.get("prompt"):
        raise ValidationError("该图没有参数快照，无法派生变体", code="E_VALIDATION")
    if not seeds:
        base = int(meta.get("seed") or 42)
        seeds = [base + i + 1 for i in range(int(n))]
    out = []
    for s in seeds:
        out.append(replay(src["path"], overrides={"seed": int(s)}, timeout=timeout,
                          compare=False, on_progress=on_progress))
    return out


def upscale(path_or_name, model=None, seed=None, sub="", timeout=DEFAULT_TIMEOUT,
            on_progress=None):
    """对已有出图做 x4 超分。model 缺省用后端默认（mewzoom，最快）。"""
    caps = capabilities()
    info = backend()
    name = os.path.basename(path_or_name)
    if not _resolve_out_path(name, info):
        raise ValidationError("超分源图不在输出目录里：%r" % name, code="E_UPSCALE_SRC",
                              hint="先把图放进 %s 或用 generate() 生成" % _out_dir(info))
    body = {"name": name, "sub": sub, "model": model or caps.get("default_upscale")}
    if seed is not None:
        body["seed"] = int(seed)
    resp = _json_post(info["url"] + "/api/upscale", body, timeout=60.0)
    jid = resp.get("id")
    j = _job_wait(info, jid, timeout, on_progress)
    fn = (j.get("output") or "").split("/")[-1] or resp.get("filename")
    path = _resolve_out_path(fn, info)
    if not path:
        raise MossError("E_OUTPUT_MISSING", "超分成功但找不到产物 %r" % fn)
    params = {"upscale_model": body["model"], "source": name}
    res = _result(path, j, params, info)
    _ledger_append({"kind": "upscale", "key": None, "status": "done", "job_id": jid,
                    "path": res["path"], "params": params, "elapsed": j.get("elapsed")})
    return res


def list_outputs(limit=20, since=None, prompt_contains=None, sub="", with_meta=False):
    """列出最近的产物（默认含原图与超分图）。"""
    info = backend(autostart=False)
    if not info:
        raise MossError("E_BACKEND_DOWN", "后端没在跑")
    data = _json_get(info["url"] + "/api/history", timeout=15.0)
    files = data.get("files", [])
    out = []
    for f in files:
        if sub and f.get("sub") != sub:
            continue
        if since and f.get("mtime", 0) < since:
            continue
        meta = f.get("meta") or {}
        if prompt_contains:
            if prompt_contains.lower() not in str(meta.get("prompt", "")).lower():
                continue
        item = {
            "filename": f["name"],
            "path": os.path.join(_out_dir(info), f.get("sub") or "", f["name"]),
            "bytes": f.get("size"),
            "mtime": f.get("mtime"),
            "sub": f.get("sub") or "",
            "topic": f.get("topic"),
            "type": meta.get("type"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "seed": meta.get("seed"),
            "engine": meta.get("engine"),
            "lora_mode": meta.get("lora_mode"),
            "prompt": (meta.get("prompt") or "")[:200],
        }
        if with_meta:
            item["meta"] = meta
        out.append(item)
        if len(out) >= int(limit):
            break
    return out


# ── 预览图（给 agent "看一眼"，不落盘到输出目录）────────────────────────────
def _prune_cache(keep=800):
    """预览缓存只增不减会慢慢吃掉几百 MB；超过上限就删最旧的一批。"""
    try:
        files = [os.path.join(CACHE_DIR, n) for n in os.listdir(CACHE_DIR)
                 if n.endswith(".jpg")]
        if len(files) <= keep:
            return
        files.sort(key=lambda p: os.path.getmtime(p))
        for p in files[:len(files) - keep]:
            with contextlib.suppress(Exception):
                os.remove(p)
    except Exception:
        pass


def thumbnail(path, max_side=768, quality=85):
    """返回 JPEG bytes（失败返回 None）。只读原图，缓存写到 ~/.moss-image/cache。"""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        st = os.stat(path)
        key = hashlib.md5(("%s|%d|%d|%d" % (os.path.abspath(path), st.st_mtime,
                                            max_side, quality)).encode()).hexdigest()
        _mkdirs()
        cache = os.path.join(CACHE_DIR, key + ".jpg")
        if os.path.isfile(cache):
            with open(cache, "rb") as f:
                return f.read()
        import io
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        with contextlib.suppress(Exception):
            with open(cache, "wb") as f:
                f.write(data)
            _prune_cache()
        return data
    except Exception:
        return None


def result_summary(res, include_meta=False):
    """给 agent 的紧凑结果：去掉体积大的字段（sha256 保留，用于校验无损）。"""
    d = {k: res.get(k) for k in
         ("ok", "path", "filename", "url", "bytes", "sha256", "width", "height",
          "elapsed", "elapsed_original", "job_id", "cached", "identical",
          "replayed_from", "wall")}
    d["params"] = {k: v for k, v in (res.get("params") or {}).items()
                   if k in ("prompt", "negative", "engine", "lora_mode", "width",
                            "height", "steps", "cfg", "seed", "seed_requested",
                            "char_lora", "char_lora_scale", "preset")}
    d["warnings"] = res.get("warnings") or []
    if include_meta:
        d["meta"] = res.get("meta")
    return d


if __name__ == "__main__":  # 便于 python3 agent/moss_client.py 自检
    print(json.dumps(backend_info(), ensure_ascii=False, indent=1))
