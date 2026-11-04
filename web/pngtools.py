#!/usr/bin/env python3
"""输出目录 + PNG 无损重压 + 参数快照 —— 无副作用的纯工具模块。

为什么单独一个文件：compress_history.py 要复用这几个函数，但原来它 `import app`
会连带执行 app.py 的模块级副作用（写 shutdown token、建目录、准备引擎路径）。
这里只留路径常量与纯函数，app.py 与 compress_history.py 共用同一份实现。

不许 import torch / mlx：app.py 主进程会 import 本模块。
"""
import json
import os
import sys
import threading

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MOSS_OUT") or os.path.expanduser("~/Pictures/MOSS-Image")
# 尾斜杠会让 "/out/" 的包含检查（OUT + os.sep）永远不成立，启动时统一归一
OUT = OUT.rstrip(os.sep) or os.sep
# 超分成图单独放子文件夹，不跟原图混在一起
UPSCALE_DIR = os.path.join(OUT, "upscaled")

# ── PNG 无损重压（oxipng）────────────────────────────────────────────────
# 生成/超分出来的图动辄 20–55MB。PNG 本身无损，重压不碰任何像素，只是把
# filter 组合和 deflate 重新选一遍：实测 8192² 省 15%、4096² 省 6%、噪点多的省 14%。
# pyoxipng 的 wheel 装在 web/_pylibs（不动系统 Python），缺了就静默跳过。
sys.path.insert(0, os.path.join(BASE, "_pylibs"))
try:
    import oxipng as _oxi
except Exception:
    _oxi = None
_PNG_OPT_LEVEL = int(os.environ.get("MOSS_PNG_OPT", "2") or 0)
_OPT_LOCK = threading.Lock()
_OPT_BUSY = set()


def _write_meta(img_path, meta):
    """在图片旁写同名 .json，记录本次的完整生成参数，便于日后复现/对比。
    失败不影响出图（老图没有快照是正常的），静默忽略。"""
    try:
        p = os.path.splitext(img_path)[0] + ".json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _read_meta(img_path):
    """读同名 .json 快照；没有或坏了返回 None。"""
    try:
        p = os.path.splitext(img_path)[0] + ".json"
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _optimize_png(path, level=None):
    """无损重压一个 PNG，返回 (before, after)；失败/无收益返回 None。

    保留原 mtime —— 历史列表按 mtime 排序，不保留会把压缩过的图全顶到最前面。
    """
    if _oxi is None:
        return None
    level = _PNG_OPT_LEVEL if level is None else level
    if level <= 0 or not path.lower().endswith(".png") or not os.path.isfile(path):
        return None
    tmp = path + ".opt"
    try:
        st = os.stat(path)
        before = st.st_size
        with open(path, "rb") as f:
            data = f.read()
        out = _oxi.optimize_from_memory(data, level=level)
        if len(out) >= before:
            return (before, before)
        with open(tmp, "wb") as f:
            f.write(out)
        os.replace(tmp, path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        return (before, len(out))
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return None


def _optimize_async(path):
    """后台线程压缩：8192² 的图要 70–80s，不该让任务完成状态等它。"""
    if _oxi is None or _PNG_OPT_LEVEL <= 0:
        return
    with _OPT_LOCK:
        if path in _OPT_BUSY:
            return
        _OPT_BUSY.add(path)

    def work():
        try:
            res = _optimize_png(path)
            if res and res[1] < res[0]:
                meta = _read_meta(path)
                if isinstance(meta, dict):
                    meta["size"] = res[1]
                    meta["opt"] = {"before": res[0], "after": res[1],
                                   "level": _PNG_OPT_LEVEL}
                    _write_meta(path, meta)
        finally:
            with _OPT_LOCK:
                _OPT_BUSY.discard(path)

    threading.Thread(target=work, daemon=True).start()
