#!/usr/bin/env python3
"""Reproduce the step-cache matrix reported in mflux-community/mflux#950.

Runs MOSS-Image-SFT at one fixed seed/prompt and compares the decoded image against
an uncached baseline (PSNR), for:

  baseline                no reuse
  stepcache-pf010         mflux StepCache, ratio 0.3, protected_fraction 0.10 (default)
  stepcache-pf032         same, protected_fraction 0.32
  stepcache-pf040         same, protected_fraction 0.40
  spread8                 8 skips every other step (caller-supplied set, no heuristic)
  adaptive                this workbench's online cosine policy (mflux 0.24 adapter)

The protected_fraction argument exists only on mflux main after #960; see README.md
for the environment this was measured in.

    MOSS_MODEL=/path/to/mlx_vi_8bit python retest.py --outdir ./out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB = REPO_ROOT / "web"

DEFAULT_PROMPT = (
    "A lone grey wolf standing on a snowy ridge at dusk, sharp fur detail, "
    "crisp whiskers, distant pine forest, low sun behind clouds, "
    "high dynamic range, photographic"
)
SPREAD8 = frozenset([8, 10, 12, 14, 16, 18, 20, 22])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MOSS_MODEL", ""),
                   help="mflux-format MOSS-Image dir containing transformer/, "
                        "text_encoder/, vae/, tokenizer/ (or set MOSS_MODEL)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--size", default="640x480", help="WxH, both multiples of 16")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratio", type=float, default=0.3, help="StepCache skip ratio")
    p.add_argument("--outdir", default="out")
    a = p.parse_args()
    if not a.model:
        p.error("--model (or MOSS_MODEL) is required")
    a.width, a.height = (int(x) for x in a.size.lower().split("x"))
    a.outdir = Path(a.outdir)
    a.outdir.mkdir(parents=True, exist_ok=True)
    return a


def psnr(a, b) -> float:
    import numpy as np
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return 99.0 if mse <= 0 else float(10.0 * np.log10(255.0 * 255.0 / mse))


def check_step_cache() -> None:
    """Fail early with a useful message when the mflux build predates #960."""
    import inspect

    from mflux.models.common.step_cache.step_cache import StepCache
    params = inspect.signature(StepCache.for_run).parameters
    if "protected_fraction" not in params:
        raise SystemExit(
            "this mflux build has no StepCache protected_fraction (pre-#960).\n"
            "Install mflux main ahead of the environment's mflux:\n"
            "  pip install --no-deps --target /tmp/mfluxmain "
            "\"git+https://github.com/mflux-community/mflux@main\"\n"
            "  PYTHONPATH=/tmp/mfluxmain python retest.py ..."
        )


def encode(a, adapter):
    import mlx.core as mx
    import numpy as np

    t0 = time.time()
    te = adapter.load_text_encoder(a.model)
    tok = adapter.load_tokenizer(a.model)
    pos, neg = adapter.encode_prompts(te, tok, a.prompt, "")
    for name, arr in (("pos", pos), ("neg", neg)):
        if arr is not None and not np.isfinite(np.array(arr.astype(mx.float32))).all():
            raise SystemExit("%s embeddings contain non-finite values" % name)
    del te
    mx.clear_cache()
    print("encoded in %.1fs" % (time.time() - t0), flush=True)
    return pos, neg


def main() -> None:
    a = parse_args()
    sys.path.insert(0, str(WEB))

    import numpy as np
    from PIL import Image

    import mflux024_adapter as adapter
    from mflux.models.common.step_cache.step_cache import StepCache

    check_step_cache()
    pos, neg = encode(a, adapter)

    dit = adapter.load_transformer(a.model, streaming=True)
    vae = adapter.load_vae(a.model)

    orig_for_run = StepCache.for_run

    def cache_with(protected_fraction):
        def patched(config, ratio, signal_fn=None, *, protected_fraction=None):
            return orig_for_run(config, ratio, signal_fn,
                                protected_fraction=(cache_with.pf if protected_fraction is None
                                                    else protected_fraction))
        return patched

    configs = [
        ("baseline", {"skip": 0.0}),
        ("stepcache-pf010", {"skip": a.ratio, "pf": 0.10}),
        ("stepcache-pf032", {"skip": a.ratio, "pf": 0.32}),
        ("stepcache-pf040", {"skip": a.ratio, "pf": 0.40}),
        ("spread8", {"fixed": SPREAD8}),
        ("adaptive", {"skip": a.ratio, "mode": "adaptive"}),
    ]

    results, base_img = [], None
    for name, cfg in configs:
        StepCache.for_run = orig_for_run
        if "fixed" in cfg:
            fixed = cfg["fixed"]
            StepCache.for_run = staticmethod(
                lambda config, ratio, signal_fn=None, **kw: StepCache(fixed))
        elif "pf" in cfg:
            patched = cache_with(cfg["pf"])
            patched.pf = cfg["pf"]
            StepCache.for_run = staticmethod(patched)

        t0 = time.time()
        kw = dict(seed=a.seed, height=a.height, width=a.width, steps=a.steps,
                  guidance=a.guidance)
        if cfg.get("mode") == "adaptive":
            img, stats = adapter.generate(dit, vae, pos, neg, **kw, skip=a.ratio,
                                          skip_mode="adaptive", skip_cos=0.972,
                                          skip_warmup=3, skip_tail=3)
        else:
            img, stats = adapter.generate(dit, vae, pos, neg, **kw,
                                          skip=cfg.get("skip", a.ratio),
                                          skip_mode="official")
        seconds = time.time() - t0
        if name == "baseline":
            base_img = img
        Image.fromarray(img).save(a.outdir / (name + ".png"))
        rec = {"name": name, "seconds": round(seconds, 1),
               "skipped": sorted(stats["skipped"]), "n_skipped": len(stats["skipped"])}
        if base_img is not None and name != "baseline":
            rec["psnr_vs_baseline"] = round(psnr(img, base_img), 2)
            rec["mae"] = round(float(np.abs(img.astype(np.float64)
                                            - base_img.astype(np.float64)).mean()), 2)
        results.append(rec)
        print("RESULT " + json.dumps(rec, ensure_ascii=False), flush=True)
        (a.outdir / "results.json").write_text(json.dumps(results, indent=2))

    StepCache.for_run = orig_for_run
    print("\n%-18s %10s %6s  %s" % ("config", "seconds", "skips", "PSNR vs baseline"))
    for r in results:
        print("%-18s %10.1f %6d  %s" % (r["name"], r["seconds"], r["n_skipped"],
                                        r.get("psnr_vs_baseline", "-")))


if __name__ == "__main__":
    main()
