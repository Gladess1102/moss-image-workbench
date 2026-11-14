#!/usr/bin/env python3
"""mflux 0.24 适配层：让 0.24 跑本机 compact 权重，并补齐 0.24 缺的能力。

背景
----
本机权重来自 vanch007/mflux 的 fork（feature/moss-image-sft），0.24 上游是另一套
权重加载框架。两边模型类参数名几乎一致，需要桥接四处：

  1. time_embedding.1.*  →  time_embedding.2.*（本机 fork 的编号 vs 0.24）
  2. quantization_scope = "proj_only"（0.24 只有全局谓词）
  3. TE 的 language_model.model. 前缀 + conv1d 布局 + RMSNorm 的 (1+w) 反偏移
  4. VAE 的 OIHW→OHWI、bn 前缀、to_out.0

0.24 缺的能力（fork 有、上游没有），本层补齐：
  * skip_text_encoder  → 用 encode_prompts() / generate() 两阶段代替
  * 自适应跳步          → sample(skip_mode="adaptive")
  * time_embedding_path → apply_time_embedding_override()

用法
----
    # 两阶段（省内存：编码阶段只装 TE，出图阶段只装 DiT）
    from mflux024_adapter import load_text_encoder, load_transformer, load_vae
    from mflux024_adapter import encode_prompts, save_embeds, load_embeds, generate

    te = load_text_encoder(MODEL)
    pos, neg = encode_prompts(te, MODEL, prompt, negative)
    save_embeds("job.embeds", pos, neg)
    del te

    pos, neg = load_embeds("job.embeds")
    dit, vae = load_transformer(MODEL), load_vae(MODEL)
    img = generate(dit, vae, pos, neg, seed=42, width=512, height=512, steps=4, guidance=1.0)

    # 单阶段（简单）
    img = run(MODEL, prompt, seed=42, width=512, height=512, steps=4, guidance=1.0)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten


# ---------------------------------------------------------------- 权重读取

def load_mlx_weights(directory: str | Path) -> dict:
    """读一个目录下所有 *.safetensors（跳过 macOS 的 ._* AppleDouble）。"""
    directory = Path(directory)
    weights: dict = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".safetensors") or fn.startswith("._"):
            continue
        weights.update(mx.load(str(directory / fn)))
    return weights


def _rename_time_embedding(weights: dict) -> dict:
    """本机 fork 的 time_embedding.1.* → 0.24 的 time_embedding.2.*"""
    out = {}
    for k, v in weights.items():
        if k.startswith("time_embedding.1."):
            out["time_embedding.2." + k[len("time_embedding.1."):]] = v
        else:
            out[k] = v
    return out


def _strip_te_prefix(weights: dict) -> dict:
    """language_model.model.* → layers.*，并做三处布局/语义转换。

      * conv1d：[C, K, 1] → [C, 1, K]（本机 mlx_lm 布局 → 0.24 的 checkpoint 布局）
      * RMSNorm：本机权重已折入 (1+w)，0.24 的 Qwen35RMSNorm 会再算 (1+weight)，
        所以这里反偏移 w-1。不修的话 TE 输出余弦只有 0.126（文本完全失效）。
      * linear_attn.norm 用 Qwen35GatedRMSNorm（直接用 weight），保持原样。
    """
    OFFSET_SUFFIXES = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    out = {}
    for k, v in weights.items():
        if k.startswith("language_model.model."):
            k2 = k[len("language_model.model."):]
        elif k.startswith("language_model."):
            k2 = k[len("language_model."):]
        else:
            continue  # vision tower 等，丢弃
        if k2.startswith("lm_head."):
            continue  # 文本编码器不需要 lm_head
        if k2.endswith("conv1d.weight") and v.ndim == 3 and v.shape[-1] == 1:
            v = v.transpose(0, 2, 1)
        elif k2 == "norm.weight" or k2.endswith(OFFSET_SUFFIXES):
            v = (v.astype(mx.float32) - 1.0).astype(v.dtype)
        out[k2] = v
    return out


# ---------------------------------------------------------------- 量化结构

def _quant_predicate(scope: str, group_size: int):
    """按本机 config 的 quantization_scope 建量化谓词（复刻 fork 的 proj_only 语义）。"""
    def pred(path: str, module):
        if not hasattr(module, "to_quantized"):
            return False
        w = getattr(module, "weight", None)
        if w is None or w.shape[-1] % group_size:
            return False
        if scope == "proj_only":
            return (".proj." in path) or ("text_embedder.blocks" in path)
        return True
    return pred


def _build_quantized(model: nn.Module, bits: int, group_size: int, scope: str) -> None:
    nn.quantize(model, group_size=group_size, bits=bits,
                class_predicate=_quant_predicate(scope, group_size))



# ---------------------------------------------------------------- 数值防护

_DELTA_GUARD = {"on": False}


def _install_delta_guard():
    """给 Qwen3.5 的 Gated DeltaNet 加数值夹紧。

    问题（2026-10-29 定位）：长 prompt 上 chunk 内 (I - a)^{-1} 的幂连乘会炸。
    实测 layer 25：a 的元素 absmax 0.98（接近满下三角），power 逐次 45 → 1.8e4
    → 1.1e8 → 6.5e12 → 4.5e15，inverse 峰值 2.4e15；state 递推后整段变 NaN，
    最终出图是 117 字节的纯黑图。短 prompt（<300 token）走不到这个病态 chunk。

    处理：把 inverse / state / 输出夹到 1e4。正常 chunk 本来就在 1e3 内，不受影响；
    病态 chunk 只损失精度，不会再污染整张图。这是工程折中，不是数值精确解。
    """
    if _DELTA_GUARD["on"]:
        return
    from mflux.models.common_models.qwen3_5.qwen3_5_gated_delta_net import (
        Qwen35GatedDeltaNet as _G,
    )

    def _guarded(query, key, value, g, beta):
        chunk = _G.CHUNK_SIZE
        query = _G._l2_normalize(query.astype(mx.float32))
        key = _G._l2_normalize(key.astype(mx.float32))
        query, key, value = (x.astype(mx.float32).transpose(0, 2, 1, 3)
                             for x in (query, key, value))
        g, beta = (x.astype(mx.float32).transpose(0, 2, 1) for x in (g, beta))

        batch, heads, seq_len, key_dim = key.shape
        value_dim = value.shape[-1]
        pad = (chunk - seq_len % chunk) % chunk
        query, key, value = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)])
                             for x in (query, key, value))
        g, beta = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (g, beta))
        chunks = (seq_len + pad) // chunk

        query = query * (key_dim**-0.5)
        value_beta = value * beta[..., None]
        key_beta = key * beta[..., None]
        query, key, value_beta, key_beta = (
            x.reshape(batch, heads, chunks, chunk, -1)
            for x in (query, key, value_beta, key_beta)
        )
        g = mx.cumsum(g.reshape(batch, heads, chunks, chunk), axis=-1)

        lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
        strictly_lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), k=-1)
        decay = mx.where(lower, mx.exp(mx.where(
            lower, g[..., :, None] - g[..., None, :], 0.0)), 0.0)

        a = mx.where(strictly_lower,
                     -((key_beta @ key.swapaxes(-1, -2)) * decay), 0.0)
        identity = mx.eye(chunk, dtype=mx.float32)
        # 对齐 transformers 参考实现（torch_chunk_gated_delta_rule）的导出兜底：
        # 参考实现在 GPU/导出场景也是"逐行前代"，而不是幂连乘。
        # mflux 的 MLX 移植用幂连乘 (I+a)(I+a²)...，中间量 a^32 能到 4.5e15，
        # fp32 只有 7 位有效数字，病态 chunk 上直接被吃成 NaN。
        # MLX 的 mx.linalg.solve_triangular 只能在 CPU 跑，所以这里手写前代：
        # X_i = B_i + Σ_{j<i} a_ij X_j，纯矩阵乘，GPU 友好且不产生巨大中间量。
        def _solve_unit_lower(rhs):
            rows = []
            for i in range(chunk):
                r = rhs[..., i, :]
                if i:
                    prev = mx.stack(rows, axis=-2)
                    r = r + mx.sum(a[..., i, :i][..., :, None] * prev, axis=-2)
                rows.append(r)
            return mx.stack(rows, axis=-2)

        value_chunks = mx.clip(_solve_unit_lower(value_beta), -1e5, 1e5)
        key_cum = mx.clip(_solve_unit_lower(key_beta * mx.exp(g)[..., None]), -1e5, 1e5)

        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
        outputs = []
        for i in range(chunks):
            q_i, k_i, g_i = query[:, :, i], key[:, :, i], g[:, :, i]
            attn = (q_i @ k_i.swapaxes(-1, -2)) * decay[:, :, i]
            v_new = value_chunks[:, :, i] - key_cum[:, :, i] @ state
            outputs.append((q_i * mx.exp(g_i)[..., None]) @ state + attn @ v_new)
            g_last = g_i[..., -1:]
            state = state * mx.exp(g_last)[..., None] + (
                k_i * mx.exp(g_last - g_i)[..., None]).swapaxes(-1, -2) @ v_new
            state = mx.clip(state, -1e4, 1e4)

        out = mx.stack(outputs, axis=2).reshape(
            batch, heads, chunks * chunk, value_dim)[:, :, :seq_len]
        return mx.clip(out, -1e4, 1e4).transpose(0, 2, 1, 3)

    _G._gated_delta_rule = staticmethod(_guarded)
    _DELTA_GUARD["on"] = True


# ---------------------------------------------------------------- 组件构造

def load_transformer(model_dir: str | Path, streaming: bool = False,
                     lora_paths=None, lora_scales=None):
    """建 0.24 的 MOSSImageTransformer，按本机 config 量化，加载权重。

    streaming=True 返回 StreamingTransformer：42 层按层 mmap，常驻内存 5.5 GB -> ~0.5 GB
    （带 LoRA 约 1.7 GB），单步约 +1.7%，输出与非流式逐比特一致。
    流式下 LoRA 必须走这里传（层是动态构建的，事后 apply_lora 找不到 layers）。
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import (
        MOSSImageTransformer,
    )

    model_dir = Path(model_dir)
    tdir = model_dir / "transformer"
    cfg = json.loads((tdir / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("moss_image_mlx", {}).get("quantization")
    scope = cfg.get("quantization_scope", "all")

    mc = ModelConfig.moss_image_sft()
    model = MOSSImageTransformer(**mc.transformer_overrides)
    if quant:
        _build_quantized(model, quant["bits"], quant["group_size"], scope)

    if streaming:
        import gc

        from mflux_stream import StreamingTransformer, lora_specs

        shards = [tdir / f for f in sorted(os.listdir(tdir))
                  if f.endswith(".safetensors") and not f.startswith("._")]
        non = {}
        for shard in shards:
            w = mx.load(str(shard))
            non.update({k: v for k, v in w.items() if not k.startswith("layers.")})
            del w
        non = _rename_time_embedding(non)
        # layers 先置空再装非层权重：42 层 bf16 空壳白占 ~5 GB
        model.layers = []
        gc.collect()
        mx.clear_cache()
        model.update(tree_unflatten(list(non.items())), strict=True)
        model.eval()
        del non
        gc.collect()
        mx.clear_cache()
        lora = lora_specs(lora_paths, lora_scales) if lora_paths else None
        return StreamingTransformer(
            model, shards, lora,
            dim=cfg.get("dim", 3072), ffn_dim=cfg.get("ffn_dim", 8192),
            num_heads=cfg.get("num_heads", 24), num_layers=cfg.get("num_layers", 42),
            bits=(quant or {}).get("bits", 8), group_size=(quant or {}).get("group_size", 64),
            quantized=bool(quant),
        )

    weights = _rename_time_embedding(load_mlx_weights(tdir))
    model.update(tree_unflatten(list(weights.items())), strict=True)
    model.eval()
    return model


def load_text_encoder(model_dir: str | Path, quantize_bits: int | None = None):
    """建 0.24 的 Qwen35TextModel，加载本机 TE 权重（4bit 或 8bit）。"""
    from mflux.models.common_models.qwen3_5 import Qwen35TextModel

    model_dir = Path(model_dir)
    te_dir = model_dir / "text_encoder"
    if not te_dir.exists():
        te_dir = model_dir          # 允许直接传 TE 目录（如 mflux q8 的下载目录）
    cfg_path = te_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    quant = cfg.get("quantization") or {}
    # 没有 config.json（mflux 原生格式）时留 None，交给 load_text_encoder_mflux 从 index 读
    bits = quantize_bits or quant.get("bits")
    group_size = quant.get("group_size", 64)

    raw = load_mlx_weights(te_dir)
    # 自动识别两种格式：
    #   本机 fork 格式 —— 键名带 language_model. 前缀，已折入 (1+w)，conv1d 是 [C,K,1]
    #   mflux 原生格式 —— 键名无前缀，norm 存偏移量，conv1d 已是 [C,1,K]（跟 0.24 天然对齐）
    is_local = any(k.startswith("language_model.") for k in raw)
    if not is_local:
        _install_delta_guard()
        return load_text_encoder_mflux(te_dir, bits=bits, group_size=group_size)

    model = Qwen35TextModel()
    if bits:
        def pred(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            w = getattr(module, "weight", None)
            return w is not None and w.shape[-1] % group_size == 0
        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=pred)

    model.update(tree_unflatten(list(_strip_te_prefix(raw).items())), strict=True)
    model.eval()
    _install_delta_guard()
    return model


def load_text_encoder_mflux(te_dir: str | Path, bits: int | None = None, group_size: int = 64):
    """加载 mflux 原生格式的 TE（如 mflux-community/moss-image-sft-mflux-q8）。

    与 load_text_encoder 的区别：这类权重是 mflux 直接从官方 bf16 量化的，
    格式天然对齐 0.24，**不需要任何转换**：
      * 键名无 language_model. 前缀（已是 embed_tokens.* / layers.*）
      * conv1d 已是 [C, 1, K]
      * norm 存偏移量，0.24 的 Qwen35RMSNorm 正好做 (1+w)
    """
    from mflux.models.common_models.qwen3_5 import Qwen35TextModel

    te_dir = Path(te_dir)
    # 从 index 的 metadata 读量化等级
    if bits is None:
        idx_path = te_dir / "model.safetensors.index.json"
        if idx_path.exists():
            meta = json.loads(idx_path.read_text()).get("metadata", {})
            bits = int(meta.get("quantization_level", 8))
        else:
            bits = 8

    model = Qwen35TextModel()
    if bits:
        def pred(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            w = getattr(module, "weight", None)
            return w is not None and w.shape[-1] % group_size == 0
        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=pred)

    weights = load_mlx_weights(te_dir)
    # 丢掉 index 里的辅助键（如果有）
    weights = {k: v for k, v in weights.items() if not k.startswith("__")}
    model.update(tree_unflatten(list(weights.items())), strict=True)
    model.eval()
    return model


def load_vae(model_dir: str | Path):
    """建 0.24 的 Flux2VAE（small decoder），加载本机 vae 目录。"""
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from mflux.models.moss_image.moss_image_initializer import (
        MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS,
    )

    model_dir = Path(model_dir)
    vae_dir = model_dir / "vae"
    model = Flux2VAE(decoder_block_out_channels=MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS)

    raw = {}
    for fn in sorted(os.listdir(vae_dir)):
        if fn.endswith(".safetensors") and not fn.startswith("._"):
            raw.update(mx.load(str(vae_dir / fn)))

    fixed = {}
    for k, v in raw.items():
        if k.startswith("bn."):
            if k.endswith(("running_mean", "running_var")):
                fixed[k] = v.astype(mx.float32)
            continue
        if not k.startswith(("decoder.", "post_quant_conv.")):
            continue
        if v.ndim == 4:
            v = v.transpose(0, 2, 3, 1)  # OIHW -> OHWI
        fixed[k.replace(".to_out.0.", ".to_out.")] = v.astype(mx.float32)

    def _collect(prefix, tree, acc):
        for k, v in tree.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _collect(p, v, acc)
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    _collect(f"{p}.{i}", x, acc)
            else:
                acc.add(p)

    model_keys: set = set()
    _collect("", model.parameters(), model_keys)
    loadable = {k: v for k, v in fixed.items() if k in model_keys}
    needed = {k for k in model_keys
              if k.startswith(("decoder.", "post_quant_conv.")) or k.startswith("bn.")}
    missing = sorted(needed - set(loadable))
    if missing:
        raise ValueError(f"VAE decoder 缺 {len(missing)} 个权重: {missing[:6]}")
    model.load_weights(list(loadable.items()), strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------- LoRA

def apply_lora(transformer, paths, scales=None, bake: bool = False) -> int:
    """在 DiT 上挂 LoRA（运行时层，不 merge）。

    bake=False 很关键：蒸馏 LoRA 的 delta 比 bf16 的舍入步长还小，
    merge 进 8bit/bf16 底模会丢掉大部分更新。
    """
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader
    from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping

    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    if scales is None:
        scales = [1.0] * len(paths)
    elif not isinstance(scales, list):
        scales = [float(scales)] * len(paths)
    LoRALoader.load_and_apply_lora(
        lora_mapping=MOSSImageLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(p) for p in paths],
        lora_scales=list(scales),
        bake_lora=bake,
    )
    return len(paths)


def apply_time_embedding_override(transformer, path: str | Path) -> int:
    """把少步适配器（如 darask0 DMAD）的时间嵌入/调制张量覆盖进去。

    对应 fork 的 time_embedding_path 参数。适配器用 PyTorch 的 0/1/2 编号，
    这里映射到 0.24 的 time_embedding.0 / time_embedding.2。
    """
    raw = mx.load(str(path))
    prefixes = (
        ("time_embedding.0.", "time_embedding.0."),
        ("time_embedding.1.", "time_embedding.2."),
        ("time_embedding.2.", "time_embedding.2."),
    )
    renamed = {}
    for key, value in raw.items():
        for src, dst in prefixes:
            if key.startswith(src):
                renamed[dst + key[len(src):]] = value.astype(mx.bfloat16)
                break
        else:
            renamed[key] = value.astype(mx.bfloat16)
    transformer.update(tree_unflatten(list(renamed.items())), strict=False)
    return len(renamed)


# ---------------------------------------------------------------- 编码（两阶段的 encode 侧）

def load_tokenizer(model_dir: str | Path):
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.moss_image.weights.moss_image_weight_definition import (
        MOSSImageWeightDefinition,
    )
    return TokenizerLoader.load_all(
        definitions=MOSSImageWeightDefinition.get_tokenizers(), model_path=str(model_dir))


def encode_prompts(text_encoder, tokenizer, prompt: str, negative: str | None = None):
    """返回 (pos, neg)。neg 为 None 表示不做 CFG。

    注意：每个 prompt 单独过 chat template、不 padding——padding 会改变
    Gated DeltaNet 的状态递推。空 prompt 也要过模板（CFG 的负支需要文本 token）。
    """
    from mflux.models.moss_image.model.moss_image_text_encoder.moss_image_prompt_encoder import (
        MOSSImagePromptEncoder,
    )
    # TokenizerLoader.load_all 返回 {name: tokenizer} 的 dict，取出 qwen3_5
    if isinstance(tokenizer, dict):
        tokenizer = tokenizer.get("qwen3_5") or next(iter(tokenizer.values()))
    pos = MOSSImagePromptEncoder.encode(prompt, tokenizer, text_encoder)
    mx.eval(pos)
    neg = None
    if negative is not None:
        neg = MOSSImagePromptEncoder.encode(negative, tokenizer, text_encoder)
        mx.eval(neg)
    return pos, neg


def save_embeds(path: str | Path, pos, neg=None) -> Path:
    """存 embedding。注意 mx.save_safetensors 会给非 .safetensors 后缀自动追加，
    所以这里先写成 .safetensors 再改名，保证落盘路径与调用方给的一致。"""
    payload = {"pos": pos.astype(mx.float32)}
    if neg is not None:
        payload["neg"] = neg.astype(mx.float32)
    p = Path(path)
    if p.suffix.lower() == ".safetensors":
        mx.save_safetensors(str(p), payload)
        return p
    tmp = p.with_name(p.name + ".safetensors")
    mx.save_safetensors(str(tmp), payload)
    os.replace(str(tmp), str(p))
    return p


def load_embeds(path: str | Path):
    # mx.load 靠扩展名判断格式：.embeds 这类自定义后缀要复制成 .safetensors 再读
    p = Path(path)
    if not p.exists():
        # 兼容早期被 mx.save_safetensors 自动加了后缀的文件
        alt = p.with_name(p.name + ".safetensors")
        if alt.exists():
            p = alt
    if p.suffix.lower() == ".safetensors":
        store = mx.load(str(p))
    else:
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            # mx.load 不跟随符号链接，用复制（embeds 只有几百 KB）
            copy = os.path.join(td, "embeds.safetensors")
            shutil.copyfile(str(p), copy)
            store = mx.load(copy)
    pos = mx.array(store["pos"])
    neg = mx.array(store["neg"]) if "neg" in store else None
    return pos, neg


# ---------------------------------------------------------------- 采样（两阶段的 run 侧）

def _cos(x, y) -> float:
    na = mx.sqrt(mx.sum(x * x))
    nb = mx.sqrt(mx.sum(y * y))
    return float(mx.sum(x * y) / (na * nb + 1e-9))


def _rel_drift(x, y) -> float:
    """相对漂移：mean|x-y| / mean|y|。

    跳步判据要的是"轨迹在多大程度上还在动"，余弦相似度对**幅度**不敏感
    （方向不变、幅度翻倍时 cos 仍接近 1），所以 MG-Cache 用的是输出漂移。
    这里归一化到当前 pred 的均值幅度，使阈值与分辨率、步数无关。
    """
    diff = float(mx.mean(mx.abs(x - y)))
    base = float(mx.mean(mx.abs(y)))
    return diff / (base + 1e-8)


def sample(transformer, pos, neg, *, seed: int, height: int, width: int,
           steps: int = 4, guidance: float = 1.0, shift: float = 5.0,
           uncond_skip_layers=(22,), skip: float = 0.0, skip_mode: str = "adaptive",
           skip_cos: float = 0.972, skip_warmup: int = 3, skip_tail: int = 3,
           skip_drift: float = 0.03,
           extrapolate: bool = False, on_step=None, progress: bool = False):
    """Euler 采样，支持 CFG / 层跳过 / 自适应跳步（对应 fork 的 mlx_runner 主干）。

    skip_mode="adaptive"：按相邻预测的余弦相似度决定是否跳步——
      中段轨迹平缓时跳过、陡峭时自动恢复，尾部自动收手。
      实测 25 步跳 7 步 PSNR 33.5 dB（固定比例同样跳 7 步只有 26.0 dB）。
    skip_mode="drift"：漂移预算（MG-Cache / arXiv 2611.11365 的 timestep cache）。
      累计"自上次真实计算以来的输出漂移"，只要加上下一次的预估漂移仍在
      skip_drift 内就继续跳。与 adaptive 的区别：adaptive 是"逐点"判据，
      单步漂移小就跳，但连续多步的小漂移叠加起来仍会累积误差；drift 是
      "预算"判据，跳得越多下一次越难再跳，天然防止漂移累积。
      skip 在此模式下表示**最大跳步比例**（对可跳区间），例如 0.5 = 最多跳一半。
    skip_mode="hybrid"：余弦判据 AND 漂移预算——单步够平 (skip_cos) 且累计
      漂移没超支 (skip_drift) 才跳。实测 25 步档：adaptive 跳 10 步 PSNR 36.3 dB，
      纯 drift 跳 11 步只剩 34.3 dB（连续跳成簇会累积误差且更快退化），
      所以把逐点判据和预算 AND 起来。
    skip_mode="fixed"：按 skip 比例固定跳（不推荐，陡峭区会跳糊）。
    skip_mode="official"：用 mflux 自带的 StepCache（按 sigma 选点）。
    """
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.moss_image.latent_creator.moss_image_latent_creator import (
        MOSSImageLatentCreator,
    )

    config = Config(width=width, height=height, guidance=guidance, scheduler="moss_image",
                    model_config=ModelConfig.moss_image_sft(), num_inference_steps=steps)
    timesteps = config.scheduler.timesteps.tolist()
    skip_set = frozenset(uncond_skip_layers)
    do_cfg = guidance != 1.0 and neg is not None

    latents = MOSSImageLatentCreator.create_noise(seed=seed, height=height, width=width)
    mx.eval(latents)

    idx = list(config.time_steps)
    n = len(idx)
    lo = max(1, skip_warmup)
    hi = n - max(1, skip_tail)
    skipped, pred_prev, last_cos, just_skipped = [], None, None, False
    cos_log = []          # 每步 pred 与上一步的余弦，用于和旧 fork 的曲线对比
    hist = []             # 最近几个"真实计算"的 pred，供跳步时做外推
    acc_drift = 0.0       # drift 模式：自上次真实计算以来的累计漂移
    last_drift = None     # drift 模式：最近一次真实相邻步的相对漂移
    drift_log = []        # 每步跳不跳时的（累计漂移, 预估漂移），供标定曲线
    drift_cap = max(1, round(skip * max(1, hi - lo))) if skip_mode in ("drift", "hybrid") else 0

    # skip_mode="official"：直接用 mflux 0.24 自带的 StepCache（TeaCache 风格）。
    # 它按 **schedule 的 sigma 相邻差** 选跳步点（信号变化最小的那 ratio 比例），
    # 保护首尾各 10%，<10 步不跳。与手写 cos 判据的区别：官方选点只看 schedule，
    # 与模型实际输出无关，所以不会因为轨迹噪声把跳步放到陡峭区。
    # 2026-10-28：手写 adaptive 在 0.24 上跳 7/25 只有 PSNR 26.33 dB（旧 fork 33.50），
    # 改用官方 StepCache 重新标定。
    step_cache = None
    if skip > 0 and skip_mode == "official":
        from mflux.models.common.step_cache.step_cache import StepCache
        step_cache = StepCache.for_run(config, ratio=skip)
        if step_cache.active:
            print("官方 StepCache：跳 %d/%d 步 %s"
                  % (len(step_cache.skip_steps), n, sorted(step_cache.skip_steps)), flush=True)

    for i in idx:
        do_skip = False
        reused = None
        if step_cache is not None:
            reused = step_cache.reuse(i)
            do_skip = reused is not None
        elif skip > 0 and lo <= i < hi and pred_prev is not None:
            if skip_mode == "drift":
                # 预估下一步漂移：用最近一次真实漂移打 9 折（sigma 递减，漂移通常变小）
                est = (last_drift or 0.0) * 0.9
                do_skip = (last_drift is not None and not just_skipped
                           and acc_drift + est <= skip_drift
                           and len(skipped) < drift_cap)
            elif skip_mode == "hybrid":
                # 逐点判据（余弦）+ 累计预算双重约束：单步够平，且累计漂移没超支
                est = (last_drift or 0.0) * 0.9
                do_skip = (last_cos is not None and last_cos >= skip_cos
                           and not just_skipped
                           and (last_drift is None or acc_drift + est <= skip_drift)
                           and len(skipped) < (drift_cap or 10 ** 9))
            elif not just_skipped:
                if skip_mode == "adaptive":
                    do_skip = last_cos is not None and last_cos >= skip_cos
                else:
                    do_skip = (i % max(1, round(1.0 / skip))) == 0

        if do_skip:
            # 关键：跳过的步**仍要推进 latents**，只是复用上一次的预测方向。
            # 直接 continue 会白白丢掉这一步的位移，轨迹严重偏离（实测画质崩）。
            if reused is not None:
                pred = reused                      # 官方 StepCache 只做复制
            elif extrapolate and len(hist) >= 2:
                # 复制会把"上一步的方向"原样搬到这一步，轨迹曲率大时误差被放大。
                # 用最近两个真实 pred 做半额线性外推，抵消一部分曲率。
                pred = hist[-1] + (hist[-1] - hist[-2]) * 0.5
            else:
                pred = pred_prev
            skipped.append(i)
            just_skipped = True
            if skip_mode == "drift" and last_drift is not None:
                acc_drift += last_drift   # 跳过这一步按最近的漂移速度记账
            drift_log.append((round(acc_drift, 4), round(last_drift or 0.0, 4)))
        else:
            t = timesteps[i]
            cond = transformer(latents, float(t), pos).astype(mx.float32)
            if do_cfg:
                uncond = transformer(latents, float(t), neg, skip_layers=skip_set).astype(mx.float32)
                pred = uncond + guidance * (cond - uncond)
            else:
                pred = cond
            if pred_prev is not None:
                last_cos = _cos(pred, pred_prev)
                if skip_mode == "drift":
                    last_drift = _rel_drift(pred, pred_prev)
            acc_drift = 0.0
            cos_log.append(last_cos)
            drift_log.append((0.0, round(last_drift or 0.0, 4)))
            pred_prev = pred
            just_skipped = False
            hist.append(pred)
            if len(hist) > 4:
                hist.pop(0)
            if step_cache is not None:
                step_cache.store(pred)

        latents = config.scheduler.step(noise=pred, timestep=i, latents=latents)
        mx.eval(latents)
        if on_step:
            on_step(i, n, do_skip)

    return latents, {"skipped": skipped, "total": n, "last_cos": last_cos,
                     "cos_log": cos_log, "drift_log": drift_log}


def decode(vae, latents, cache_limit=1 << 30):
    """latents -> uint8 图像数组 (H, W, 3)。

    cache_limit（字节，默认 1 GB）：解码期间把 MLX buffer 池收紧到该值，结束后
    还原。默认不再是无限制——批量跑图的脚本会连续解码几十次，裸调用不设限时
    池子会一路涨到系统压缩内存（2026-11-14 实测：连着跑 6 个 25 步配置没限，
    16 GB 机器被压出 10 GB swap）。
    VAE 解码会在几秒内制造大量尺寸各异的中转 buffer，池子会顺着已释放的
    空间继续扩张（mlx-vlm #2615 实测：MOSS-Image 1024²/50 步，解码几秒内
    进程 12.1 GB → 38.7 GB，48 GB 的机器都被顶出 5 GB+ swap）。收紧池子
    只影响峰值，像素逐比特一致。
    """
    from mflux.models.common.config.model_config import ModelConfig

    prev_limit = None
    if cache_limit is not None:
        prev_limit = mx.set_cache_limit(int(cache_limit))
    try:
        decoded = vae.decode_packed_latents(latents.astype(ModelConfig.precision))
        mx.eval(decoded)
        arr = np.array(decoded)
    finally:
        if prev_limit is not None:
            try:
                mx.set_cache_limit(prev_limit)
            finally:
                mx.clear_cache()
    if arr.ndim == 4:
        arr = arr[0]                      # (1, C, H, W) -> (C, H, W)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    # [-1,1] -> uint8，跟 diffusers 的 postprocess 一致
    arr = np.clip((arr / 2 + 0.5) * 255, 0, 255).round().astype(np.uint8)
    return arr


# ---------------------------------------------------------------- 组合入口

def generate(transformer, vae, pos, neg, *, seed, height, width, steps=4, guidance=1.0,
             shift=5.0, uncond_skip_layers=(22,), decode_cache_limit=None, **skip_kw):
    """采样 + 解码，返回 (uint8 图像, 采样统计)。"""
    latents, stats = sample(transformer, pos, neg, seed=seed, height=height, width=width,
                            steps=steps, guidance=guidance, shift=shift,
                            uncond_skip_layers=uncond_skip_layers, **skip_kw)
    return decode(vae, latents, cache_limit=decode_cache_limit), stats


def run(model_dir, prompt, *, negative=None, seed=42, height=512, width=512, steps=4,
        guidance=1.0, lora=None, lora_scales=None, out=None, **skip_kw):
    """单阶段一站式：装 TE + DiT + VAE，编码、采样、解码、存图。

    简单但内存峰值高（TE 和 DiT 同时在）。16 GB 机器建议用两阶段。
    """
    from PIL import Image

    te = load_text_encoder(model_dir)
    tok = load_tokenizer(model_dir)
    pos, neg_emb = encode_prompts(te, tok, prompt, negative if (negative or guidance != 1.0) else None)
    # 与两阶段对齐：save_embeds 存 fp32，这里也转 fp32，
    # 否则单阶段(bf16)与两阶段(fp32)同 seed 出图会差 19 dB（embedding 精度敏感）
    pos = pos.astype(mx.float32)
    if neg_emb is not None:
        neg_emb = neg_emb.astype(mx.float32)
    del te

    dit = load_transformer(model_dir)
    vae = load_vae(model_dir)
    if lora:
        apply_lora(dit, lora, lora_scales)

    img, stats = generate(dit, vae, pos, neg_emb, seed=seed, height=height, width=width,
                          steps=steps, guidance=guidance, **skip_kw)
    if out:
        Image.fromarray(img).save(out)
    return img, stats