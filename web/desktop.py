#!/usr/bin/env python3
"""MOSS-Image 桌面壳 —— 用 macOS 原生 WKWebView 承载工作台，替代 Chrome。

为什么不用 Electron/Chrome：Chromium 内核开一个页面要吃掉几百 MB（本机实测
Chrome 全部实例合计 2.7 GB），而 WKWebView 是系统自带组件，同一页面只要 50-80 MB。
（Tauri 更小，但要 Rust 工具链；本机没有，且对这个单机工具属过度工程。）

设计要点：
  - 复用已在跑的 app.py；没跑就先拉起来
  - **壳活着，服务就必须活着**：看护线程每 5s 探一次端口，连续失败就重启服务并
    刷新窗口。以前靠 app.py 的"空闲 N 秒自退"兜底 Cmd+Q 残留，结果 macOS 的
    App Nap 会冻结后台窗口的 JS 定时器 → 前端轮询停摆 → 服务把自己判成"空闲"
    然后退出，窗口还开着，表现为"UI 卡死、点刷新也没用"。
  - **关窗即干净退出**：先请服务端杀推理子进程（/api/shutdown），再兜底 terminate/kill，
    这样 MLX 子进程不会残留在后台白占内存
  - 服务进程的 stdout/stderr 落到 logs/workbench.log，出事能查
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
# 端口统一从 MOSS_PORT 读（app.py / .command / 壳三处同源）
PORT = int(os.environ.get("MOSS_PORT") or "7860")
URL = "http://127.0.0.1:%d" % PORT
LOG_DIR = os.path.join(BASE, "logs")
# 壳事件与 app.py 的 stdout 分开写：原来混在一个文件里，壳的几行日志会把
# 服务端 traceback 冲散，排障时很难看清哪段是谁输出的。
LOG_PATH = os.path.join(LOG_DIR, "shell.log")
WORKBENCH_LOG = os.path.join(LOG_DIR, "workbench.log")


def _log(msg):
    """壳侧事件日志；写失败不影响主流程。"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except Exception:
        pass


def port_open(port=PORT, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(0.6)
        return s.connect_ex((host, port)) == 0


def health_ok(timeout=2.0):
    """应用级握手：确认 7860 上跑的确实是本工作台，而不是别的程序。

    2026-10-24 修 P1-1：以前只判断端口能连，Docker 端口映射、上个没退干净的
    app.py、别的小服务都会被当成"已有工作台"直接复用，然后把别人的页面显示出来。
    """
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return data.get("app") == "moss-image"
    except Exception:
        return False


def wait_for_server(timeout=30):
    """等端口 + 握手都成功。超时从 25s 提到 30s（与 .command 对齐）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_open() and health_ok(1.0):
            return True
        time.sleep(0.3)
    return False


def _server_pid(timeout=2.0):
    """问 /api/health 要服务进程 pid：token 文件名带 pid，壳按它去读。

    2026-11-04：token 文件原来是固定的 /tmp/moss-shutdown.token，两个实例会互相覆盖。
    """
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=timeout) as r:
            return int(json.loads(r.read().decode("utf-8", "replace")).get("pid") or 0)
    except Exception:
        return 0


def call_shutdown(pid=None):
    """请服务端自行收尾（杀推理子进程后退出）。超时/失败都不影响后续兜底。

    2026-10-24：/api/shutdown 加了 token（原来是无鉴权 GET，任意网页一行 img 标签
    就能杀掉服务和在跑的任务）。token 由 app.py 启动时写到 /tmp/moss-shutdown-<pid>.token。
    """
    try:
        tok = ""
        try:
            _pid = pid or _server_pid()
            if _pid:
                with open("/tmp/moss-shutdown-%d.token" % int(_pid), encoding="utf-8") as f:
                    tok = f.read().strip()
        except OSError:
            pass
        # token 走 header：避免出现在 ps aux 的 argv 里（P1-9）
        req = urllib.request.Request(URL + "/api/shutdown",
                                     headers={"X-Moss-Token": tok} if tok else {})
        urllib.request.urlopen(req, timeout=4).read()
        return True
    except Exception:
        return False


_SERVER_REF = None   # main() 里赋值，供信号处理用


def _alert(msg, title="MOSS-Image 工作台"):
    """失败时给用户一个看得见的出口（Finder 启动的 .app 没有终端可看）。"""
    try:
        script = 'display dialog %s with title %s buttons {"好"} default button 1 with icon caution' % (
            json.dumps(msg), json.dumps(title))
        subprocess.run(["osascript", "-e", script], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _install_signal_handlers():
    """Cmd+Q / Dock 退出会发 SIGTERM；默认行为不会走清理，这里接管。

    2026-10-24 修 P1-3：原来无条件 call_shutdown()，会杀掉**不是自己拉起**的服务
    （多窗口/复用了终端里那个 app.py 时，退出一个窗口会连带杀掉别人的推理）。
    现在统一走 server.stop()：只收自己拉起的，复用型服务不碰。
    """
    def _handler(signum, frame):
        try:
            if _SERVER_REF is not None:
                _SERVER_REF.stop()
            else:
                call_shutdown()
        except Exception:
            pass
        # Cocoa runloop 里 sys.exit() 不一定能立刻结束进程，硬退更可靠
        os._exit(0)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


class Server:
    """工作台服务的生命周期：启动 / 重启 / 收尾。"""

    def __init__(self):
        self.proc = None
        self.log_fp = None
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return True
            if port_open():
                # 端口被占：必须先确认对方是本工作台，否则任何占用 7860 的程序
                # （Docker 映射、别的项目）都会被当成工作台把页面显示出来。
                if health_ok(2.0):
                    # 终端里手动跑的 app.py：不抢，复用（它没有 MOSS_PARENT_PID，不随壳退出）
                    _log("端口 %d 已有工作台，复用（未接管）" % PORT)
                    self.proc = None
                    return True
                _log("端口 %d 被非工作台程序占用，拒绝复用" % PORT)
                _alert("7860 端口被其他程序占用，且不是 MOSS-Image 工作台。\n"
                       "请先关掉那个程序，或改用其他端口。")
                return False
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
                self.log_fp = open(WORKBENCH_LOG, "ab", buffering=0)
            except Exception:
                self.log_fp = subprocess.DEVNULL
            env = {**os.environ, "MOSS_PARENT_PID": str(os.getpid())}
            self.proc = subprocess.Popen(
                [sys.executable, os.path.join(BASE, "app.py")],
                cwd=BASE, stdout=self.log_fp, stderr=self.log_fp, env=env)
            _log("启动工作台服务 pid=%d" % self.proc.pid)
            ok = wait_for_server()
            if not ok:
                _log("服务启动超时（端口 %d）" % PORT)
            return ok

    def stop(self):
        with self.lock:
            if self.proc is None:
                return
            if self.proc.poll() is None:
                call_shutdown(self.proc.pid)
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=3)
                    except Exception:
                        self.proc.kill()
            self.proc = None
            try:
                if self.log_fp not in (None, subprocess.DEVNULL):
                    self.log_fp.close()
            except Exception:
                pass
            self.log_fp = None

    def restart(self):
        _log("服务不响应，重启中")
        self.stop()
        time.sleep(0.5)
        ok = self.start()
        _log("服务重启%s" % ("成功" if ok else "失败"))
        return ok


def _watch(server, win):
    """看护循环（跑在后台线程）：壳活着，服务就不能死。

    2026-10-24 修：以前只看端口 connect(0.6s)，服务稍忙（GIL 被长请求占住、
    正在序列化大响应）就被判死；连续两次 → restart → /api/shutdown 把**正在跑的
    推理** SIGKILL 掉。当天因此自我重启 8 次。
    现在：自己拉起的子进程只要 poll() 还活着就绝不重启；端口只在进程真退出后才当补充判据。
    """
    fails = 0
    while True:
        time.sleep(5)
        proc = server.proc
        if proc is not None:
            if proc.poll() is None:
                # 2026-10-24 修 P1-6：进程活着 ≠ 服务可用。_stream 卡死、GPU_LOCK 被永久占住时，
                # 进程照样在、端口照样能连，但 HTTP 永远不响应，界面就是一直转圈。
                # 所以这里加一道健康检查：连续 3 次（约 15s）不响应才判定需要重启。
                if health_ok(2.0):
                    if fails:
                        _log("健康检查恢复，失败计数清零")
                    fails = 0
                    continue
                fails += 1
                _log("健康检查失败（连续 %d 次）" % fails)
                if fails < 3:
                    continue
                _log("服务进程在但 HTTP 无响应，重启")
                fails = 0
                if server.restart():
                    try:
                        win.evaluate_js("location.reload()")
                    except Exception:
                        pass
                continue
            _log("检测到工作台子进程已退出（rc=%r）" % proc.poll())
        else:
            try:
                alive = port_open()
            except Exception:
                alive = False
            if alive:
                if fails:
                    _log("端口恢复，失败计数清零")
                fails = 0
                continue
        fails += 1
        _log("服务不可用（连续 %d 次）" % fails)
        if fails < 6:          # 2 次(10s) → 6 次(30s)
            continue
        fails = 0
        if server.restart():
            # 服务回来了，让已经死掉的页面重新加载
            try:
                win.evaluate_js("location.reload()")
                _log("已通知窗口重新加载")
            except Exception as e:
                _log("刷新窗口失败: %r" % e)


def main():
    try:
        import webview
    except ImportError:
        msg = "缺少 pywebview，先装：\n  %s -m pip install pywebview" % sys.executable
        sys.stderr.write(msg + "\n")
        _alert(msg)
        return 2

    _install_signal_handlers()

    global _SERVER_REF
    server = Server()
    _SERVER_REF = server
    if not server.start():
        msg = ("工作台服务没能启动（端口 %d 不可用或启动超时）。\n"
               "日志：%s" % (PORT, LOG_PATH))
        sys.stderr.write(msg + "\n")
        _alert(msg)
        server.stop()
        return 1

    win = webview.create_window("MOSS-Image 工作台", URL,
                                width=1440, height=940, min_size=(900, 640))

    def _on_closing():
        """窗口关闭按钮 / 系统退出时触发。这里放干净收尾。

        2026-10-24 修（P0-1）：原来只 server.stop() 然后 return True。
        但 webview.start(_watch, ...) 把 _watch 跑在**非 daemon 线程**里，
        窗口一关 Cocoa 停掉 runloop，主线程要 join 那个 while True 的看护线程，
        进程于是卡住不退；30 秒后残留看护判定"服务挂了"，又把服务重新拉起来（无窗口）。
        这里收尾完直接 os._exit，绕开 join。
        另外（P1-7）服务忙时 /api/shutdown 自己也可能等几秒，所以收尾动作放到
        后台线程，窗口先关掉，用户不会看到"点了没反应"。
        """
        def _finish():
            try:
                server.stop()
            finally:
                os._exit(0)
        threading.Thread(target=_finish, daemon=True).start()
        return True

    try:
        win.events.closing += _on_closing
    except Exception:
        pass

    # start(func) 会把看护函数放到 GUI 就绪后的独立线程里跑
    webview.start(_watch, (server, win))

    # process 收尾（事件没走到时的兜底）
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
