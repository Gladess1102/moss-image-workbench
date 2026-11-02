#!/usr/bin/env python3
"""把历史出图做一遍 PNG 无损重压（oxipng）。

- 无损：像素一个不差，只重选 filter + deflate 编码
- 保留 mtime：历史列表按时间排序，不会因为压缩而乱序
- 幂等：已经压过的再跑一遍基本没有收益（结果更大时保留原文件）

用法：
  python3 compress_history.py                 # 压缩全部
  python3 compress_history.py --limit 5       # 只压前 5 张（试水）
  python3 compress_history.py --level 4       # 更慢更小（默认 2）
  python3 compress_history.py --min-mb 0.5    # 跳过小于 0.5MB 的
"""
import argparse
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import app  # 复用 _optimize_png / _write_meta / _read_meta / OUT / UPSCALE_DIR


def iter_pngs(min_bytes):
    for d in (app.OUT, app.UPSCALE_DIR):
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if name.lower().endswith(".png") and os.path.isfile(p) \
                    and os.path.getsize(p) >= min_bytes:
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="最多处理几张（0=全部）")
    ap.add_argument("--level", type=int, default=app._PNG_OPT_LEVEL)
    ap.add_argument("--min-mb", type=float, default=0.3)
    args = ap.parse_args()
    if app._oxi is None:
        print("缺少 pyoxipng：pip3 install --target _pylibs pyoxipng")
        return 2
    if args.level <= 0:
        print("level 必须 > 0")
        return 2

    files = list(iter_pngs(int(args.min_mb * 1e6)))
    if args.limit:
        files = files[:args.limit]
    total_before = sum(os.path.getsize(p) for p in files)
    print("待处理 %d 张，共 %.1f MB（level=%d）" % (
        len(files), total_before / 1e6, args.level), flush=True)

    saved = 0
    done = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        res = app._optimize_png(p, level=args.level)
        if res and res[1] < res[0]:
            saved += res[0] - res[1]
            done += 1
            print("[%d/%d] %-46s %6.1f -> %6.1f MB  -%.1f%%" % (
                i, len(files), os.path.basename(p)[:46], res[0] / 1e6,
                res[1] / 1e6, (1 - res[1] / res[0]) * 100), flush=True)
            # 同步更新同名快照里的 size，灯箱/历史里看到的就是压缩后的
            meta = app._read_meta(p)
            if isinstance(meta, dict):
                meta["size"] = res[1]
                meta["opt"] = {"before": res[0], "after": res[1],
                               "level": args.level}
                app._write_meta(p, meta)
        else:
            print("[%d/%d] %-46s 跳过（无收益）" % (
                i, len(files), os.path.basename(p)[:46]), flush=True)

    print("完成：%d 张有收益，共省 %.1f MB（%.1f%%），耗时 %.0fs" % (
        done, saved / 1e6, saved / max(total_before, 1) * 100,
        time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
