#!/usr/bin/env python3
"""mossimg —— MOSS-Image 的命令行入口（agent 与人都能用）。

约定：默认输出**单行 JSON**（agent 直接解析），--human 给人看。
所有产物返回绝对路径；不会对图片做任何重编码。

常用：
  mossimg capabilities                          # 能力清单（先看这个）
  mossimg generate "提示词" --preset draft      # 出图（阻塞到完成）
  mossimg generate "提示词" --preset quality --seed 42
  mossimg batch "提示词" -n 4                    # 4 个变体
  mossimg list -n 10                            # 最近产物
  mossimg inspect <图>                          # 参数快照
  mossimg replay <图>                           # 原样重跑并比对 sha256
  mossimg vary <图> -n 3                        # 换 seed 出变体
  mossimg upscale <图> --model mewzoom          # x4 超分
  mossimg validate-prompt -f prompt.txt         # 提交前校验提示词
  mossimg status | cancel <job_id> | stop       # 后端管理
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moss_client as mc  # noqa: E402


PROMPT_GUIDE = """\
# MOSS-Image 提示词要点（agent 速查，完整版见 prompts/README.md）

两种正向提示词，二选一：
  1) 纯文本 caption
  2) 四键 Layout JSON：{"summary":{...},"items":[...],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
     · 顶层只能这四个键，多写 schema/width/negative 之类会被模型当废话忽略
     · bbox 相对画布中心、单位像素、顺序 [y1,x1,y2,x2]；换尺寸要按新画布等比缩放
     · items 建议 5–6 个；同一件东西只在一个 item 里详写（重复会出双份）
     · 拿不准就用 mossimg layout 生成，别手写 JSON

写 caption 的顺序：位置锚点 → 主体外形 → 服装与道具 → 动作表情 → 材质细节 → 光。
长度：建议 1300–2600 字符（编码器 4096 token）；关键内容别放最后。
年龄：写「三十岁上下」；要大叔感靠体型/胡茬/油污/疲惫，不靠年龄词。
姿态要动：蹲、扑、跳、够、拧——比"坐着看镜头"出片率高。

负面词是**独立参数**（--negative），不能塞进 JSON；且只在 CFG≠1.0 时生效：
  akk4(4步) CFG=1.0 → 负面词完全不参与计算
  turbo / native25 / none / full CFG=3.0 → 生效
角色 LoRA 的触发词要自己写进 caption 开头，下拉只挂权重。
"""

HUMAN = False


def _emit(obj, human=None):
    if HUMAN:
        if human:
            print(human)
        else:
            print(json.dumps(obj, ensure_ascii=False, indent=1))
    else:
        print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _err(e):
    d = e.to_dict() if isinstance(e, mc.MossError) else {"ok": False, "code": "E_INTERNAL",
                                                         "message": repr(e)}
    if HUMAN:
        print("✗ %s: %s" % (d.get("code"), d.get("message")), file=sys.stderr)
        if d.get("hint"):
            print("  → %s" % d["hint"], file=sys.stderr)
        if d.get("detail") and not isinstance(d["detail"], (dict, list)):
            print("  detail: %s" % str(d["detail"])[:800], file=sys.stderr)
    else:
        print(json.dumps(d, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
    return 1 if not isinstance(e, mc.MossError) else 2


def _read_prompt(args):
    if getattr(args, "prompt_file", None):
        try:
            with open(args.prompt_file, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            raise mc.ValidationError("读不到提示词文件：%s" % e, code="E_VALIDATION",
                                     hint="检查 -f 的路径是否存在、是否有读权限")
    parts = args.prompt or []
    return " ".join(parts) if isinstance(parts, list) else (parts or "")


def _progress(j):
    ph = j.get("phase") or ""
    pr = j.get("progress")
    step = j.get("step") or ""
    eta = j.get("eta")
    line = "  [%s] %s %s%s" % (j.get("status"), ph,
                               ("%s%%" % pr) if pr is not None else "",
                               ("  %s" % step) if step else "")
    if eta:
        line += "  剩余~%ss" % eta
    print(line, file=sys.stderr)


def cmd_capabilities(args):
    caps = mc.capabilities(refresh=args.refresh)
    if not args.full:
        modes = [{k: m.get(k) for k in ("id", "label", "steps", "cfg", "engine", "turbo", "desc")}
                 for m in caps["lora_modes"]]
        caps = {"ok": caps["ok"], "backend": caps["backend"],
                "engines": caps.get("engines"),
                "error_codes": caps.get("error_codes"),
                "default_engine": caps["default_engine"],
                "default_lora_mode": caps["default_lora_mode"],
                "presets": caps["presets"], "lora_modes": modes,
                "char_loras": [c.get("id") if isinstance(c, dict) else c
                               for c in caps["char_loras"]],
                "upscale_models": [u["id"] for u in caps["upscale_models"]],
                "size_rule": caps["size_rule"], "steps_rule": caps["steps_rule"],
                "negative_rule": caps["negative_rule"], "prompt_rule": caps["prompt_rule"],
                "version": caps.get("version"), "unsupported": caps.get("unsupported")}
    _emit(caps)
    return 0


def cmd_generate(args):
    prompt = _read_prompt(args)
    eta = None
    if HUMAN and not args.quiet:
        try:
            p, _w = mc.normalize_params(prompt, preset=args.preset, width=args.w,
                                        height=args.h, steps=args.steps, cfg=args.cfg,
                                        seed=args.seed, engine=args.engine,
                                        lora_mode=args.lora_mode)
            eta = mc.estimate_eta(p)
            print("# 预计 %ss（%s）→ 生成中…" % (eta.get("seconds"), eta.get("source")),
                  file=sys.stderr)
        except mc.MossError:
            pass
    t0 = time.time()
    res = mc.generate(
        prompt, preset=args.preset, width=args.w, height=args.h,
        negative=args.negative, steps=args.steps, cfg=args.cfg, seed=args.seed,
        engine=args.engine, lora_mode=args.lora_mode,
        char_lora=args.char_lora, char_lora_scale=args.char_lora_scale,
        enhance=args.enhance, timeout=args.timeout,
        dedupe=not args.no_dedupe, idempotency_key=args.idem_key,
        on_progress=(_progress if HUMAN and not args.quiet else None),
    )
    out = mc.result_summary(res)
    out["wall"] = round(time.time() - t0, 1)
    if eta:
        out["eta"] = eta
    if args.preview:
        data = mc.thumbnail(res["path"], max_side=args.preview_max)
        if data:
            with open(args.preview, "wb") as f:
                f.write(data)
            out["preview"] = os.path.abspath(args.preview)
    human = "✓ %s  (%sx%s, %.1fs%s)\n%s" % (
        res["path"], res["width"], res["height"], res.get("elapsed") or 0,
        ", 幂等命中" if res.get("cached") else "", res["path"])
    _emit(out, human)
    return 0


def cmd_batch(args):
    prompt = _read_prompt(args)
    res = mc.generate_batch(prompt, n=args.n, seed_base=args.seed_base,
                            preset=args.preset, width=args.w, height=args.h,
                            negative=args.negative, timeout=args.timeout,
                            on_progress=(_progress if HUMAN else None))
    _emit([mc.result_summary(r) for r in res],
          "\n".join("✓ %s (%ss)" % (r["path"], r.get("elapsed")) for r in res))
    return 0


def cmd_upscale(args):
    res = mc.upscale(args.image, model=args.model, seed=args.seed, sub=args.sub,
                     timeout=args.timeout,
                     on_progress=(_progress if HUMAN else None))
    _emit(mc.result_summary(res), "✓ %s (%sx%s, %ss)" % (res["path"], res["width"],
                                                         res["height"], res.get("elapsed")))
    return 0


def cmd_list(args):
    items = mc.list_outputs(limit=args.n, since=(time.time() - args.since * 60
                                                if args.since else None),
                            prompt_contains=args.prompt_contains, sub=args.sub,
                            with_meta=args.full)
    _emit(items, "\n".join("%s\t%s\t%sx%s" % (i["filename"], i["type"] or "-",
                                              i["width"], i["height"]) for i in items))
    return 0


def cmd_inspect(args):
    _emit(mc.inspect(args.image))
    return 0


def cmd_replay(args):
    res = mc.replay(args.image, compare=not args.no_compare, timeout=args.timeout,
                    on_progress=(_progress if HUMAN else None))
    _emit(mc.result_summary(res),
          "✓ %s\n  原图 %s\n  identical=%s" % (res["path"], res.get("replayed_from"),
                                               res.get("identical")))
    return 0


def cmd_vary(args):
    res = mc.vary(args.image, n=args.n, timeout=args.timeout,
                  on_progress=(_progress if HUMAN else None))
    _emit([mc.result_summary(r) for r in res],
          "\n".join("✓ %s (seed %s)" % (r["path"], (r["params"] or {}).get("seed"))
                    for r in res))
    return 0


def cmd_validate(args):
    prompt = _read_prompt(args)
    r = mc.validate_prompt(prompt, width=args.w, height=args.h, negative=args.negative)
    _emit(r, ("✓ 通过（%s, %d 字符）" % (r["kind"], r["stats"]["chars"])) if r["ok"]
          else "✗ 不通过：%s" % r["errors"])
    return 0 if r["ok"] else 2


def cmd_layout(args):
    items = [json.loads(s) for s in (args.item or [])]
    text = mc.build_layout_json(args.summary, items, args.w, args.h)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        _emit({"ok": True, "path": os.path.abspath(args.out), "chars": len(text)})
    else:
        print(text)
    return 0


def cmd_status(args):
    info = mc.backend_info()
    if info.get("ok"):
        info["current"] = [{"id": j.get("id"), "status": j.get("status"),
                            "phase": j.get("phase"), "progress": j.get("progress"),
                            "prompt": (j.get("prompt") or "")[:60]}
                           for j in mc.current_jobs()]
        info["eta_note"] = "同机只跑一个任务（后端 GPU_LOCK 串行）"
    _emit(info)
    return 0 if info.get("ok") else 1


def cmd_cancel(args):
    _emit(mc.cancel(args.job_id))
    return 0


def cmd_stop(args):
    _emit({"ok": mc.stop_backend(force=args.force)})
    return 0


def cmd_guide(args):
    print(PROMPT_GUIDE, end="")
    return 0


def main(argv=None):
    global HUMAN
    ap = argparse.ArgumentParser(prog="mossimg",
                                 description="MOSS-Image 本地出图 CLI（默认输出单行 JSON）")
    ap.add_argument("--human", action="store_true", help="人类可读输出（默认单行 JSON）")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("capabilities", help="能力清单")
    p.add_argument("--full", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_capabilities)

    def add_gen(sp):
        sp.add_argument("prompt", nargs="*", help="提示词（或 -f 文件）")
        sp.add_argument("-f", "--prompt-file", help="从文件读提示词（长 Layout JSON 用）")
        sp.add_argument("--preset", help="意图档位：%s" % " / ".join(sorted(mc.PRESETS)))
        sp.add_argument("--w", type=int, help="宽（16 的倍数）")
        sp.add_argument("--h", type=int, help="高（16 的倍数）")
        sp.add_argument("--negative", default=None, help="负面词（CFG≠1 才生效）")
        sp.add_argument("--steps", type=int)
        sp.add_argument("--cfg", type=float)
        sp.add_argument("--seed", default=None, help="整数或 random")
        sp.add_argument("--engine", help="引擎：mlx8（默认）/ msmlx")
        sp.add_argument("--lora-mode", dest="lora_mode")
        sp.add_argument("--char-lora")
        sp.add_argument("--char-lora-scale", dest="char_lora_scale", type=float)
        sp.add_argument("--enhance", action="store_true", help="提示词强化（+10~90s）")
        sp.add_argument("--timeout", type=float, default=mc.DEFAULT_TIMEOUT)
        sp.add_argument("--no-dedupe", action="store_true", help="同参数也重新生成")
        sp.add_argument("--idem-key", dest="idem_key", help="自定义幂等键")
        sp.add_argument("--quiet", action="store_true")

    p = sub.add_parser("generate", help="出图（阻塞）")
    add_gen(p)
    p.add_argument("--preview", help="另存一张缩略图到该路径")
    p.add_argument("--preview-max", type=int, default=768)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("batch", help="同 prompt 多变体")
    add_gen(p)
    p.add_argument("-n", type=int, default=4)
    p.add_argument("--seed-base", type=int)
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("upscale", help="x4 超分")
    p.add_argument("image")
    p.add_argument("--model")
    p.add_argument("--seed", type=int)
    p.add_argument("--sub", default="")
    p.add_argument("--timeout", type=float, default=mc.DEFAULT_TIMEOUT)
    p.set_defaults(fn=cmd_upscale)

    p = sub.add_parser("list", help="最近产物")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--since", type=float, help="最近 N 分钟")
    p.add_argument("--prompt-contains")
    p.add_argument("--sub", default="")
    p.add_argument("--full", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("inspect", help="读参数快照")
    p.add_argument("image")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("replay", help="原样重跑并比对")
    p.add_argument("image")
    p.add_argument("--no-compare", action="store_true")
    p.add_argument("--timeout", type=float, default=mc.DEFAULT_TIMEOUT)
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("vary", help="换 seed 出变体")
    p.add_argument("image")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--timeout", type=float, default=mc.DEFAULT_TIMEOUT)
    p.set_defaults(fn=cmd_vary)

    p = sub.add_parser("validate-prompt", help="提交前校验提示词")
    p.add_argument("prompt", nargs="*")
    p.add_argument("-f", "--prompt-file")
    p.add_argument("--w", type=int)
    p.add_argument("--h", type=int)
    p.add_argument("--negative")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("layout", help="生成合规 Layout JSON")
    p.add_argument("--summary", required=True)
    p.add_argument("--item", action="append",
                   help='JSON: {"label":"character","caption":"...","bbox":[y1,x1,y2,x2]}')
    p.add_argument("--w", type=int, default=1184)
    p.add_argument("--h", type=int, default=896)
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_layout)

    p = sub.add_parser("status", help="后端状态")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("cancel", help="取消任务")
    p.add_argument("job_id")
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser("stop", help="关掉后端（需 token）")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("prompt-guide", help="打印提示词规范")
    p.set_defaults(fn=cmd_guide)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    HUMAN = bool(args.human)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:            # noqa: BLE001
        return _err(e)


if __name__ == "__main__":
    sys.exit(main())
