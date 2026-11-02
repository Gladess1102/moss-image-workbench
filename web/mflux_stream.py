"""MOSS-Image DiT 的按层流式加载（mlx 版）。

不拆文件：42 层直接从原 shard 里按 key 取（mx.load 是 mmap，只触碰这一层的页面），
算完 del + clear_cache，页面立刻还给系统。

实测（640x480、单步前向、同 latent/timestep/embeds）：
  非流式 active 5.54 GB / 34.6s
  流式   active 0.48 GB / 35.2s
  带 LoRA 时 1.69 GB，输出与非流式逐比特一致（PSNR 99 dB，max abs diff 0.0）。
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


def lora_specs(paths, scales=None):
    """LoRA 文件 -> {"layers.N.proj.X": (A(in,r), B(r,out), alpha, scale)}。

    兼容 kohya 的 lora_down/lora_up 与 PEFT 的 lora_A/lora_B；alpha 缺省用 rank。
    """
    if isinstance(paths, (str,)):
        paths = [paths]
    scales = list(scales or [1.0] * len(paths))
    out = {}
    for path, scale in zip(paths, scales):
        raw = dict(mx.load(str(path)).items())
        tmp = {}
        for k, v in raw.items():
            kk = k.removeprefix("diffusion_model.").removeprefix("transformer.")
            if ".lora_down." in kk or ".lora_A." in kk:
                tmp.setdefault(kk.rsplit(".lora_", 1)[0], {})["A"] = v
            elif ".lora_up." in kk or ".lora_B." in kk:
                tmp.setdefault(kk.rsplit(".lora_", 1)[0], {})["B"] = v
            elif kk.endswith(".alpha"):
                tmp.setdefault(kk[:-6], {})["alpha"] = float(v)
        for base, d in tmp.items():
            if "A" not in d or "B" not in d:
                continue
            a_t = d["A"].T          # (r, in) -> (in, r)
            b_t = d["B"].T          # (out, r) -> (r, out)
            # 同一个 target 可以有多个 LoRA（蒸馏档 + 风格档叠加），必须都留下
            out.setdefault(base, []).append((a_t, b_t, d.get("alpha", a_t.shape[1]), scale))
    return out


class StreamingTransformer:
    """把只有非层权重的 MOSSImageTransformer 包成逐层加载的版本。

    base 必须已经：layers 置空、按 config 量化、非层权重 update 完成。
    """

    def __init__(self, base, shards, lora=None, dim=3072, ffn_dim=8192,
                 num_heads=24, num_layers=42, bits=8, group_size=64, quantized=True):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "shards", [str(s) for s in shards])
        object.__setattr__(self, "lora", lora or {})
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "ffn_dim", ffn_dim)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "num_layers", num_layers)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "group_size", group_size)
        object.__setattr__(self, "quantized", quantized)

    def __getattr__(self, name):
        # 只转发到 base；自身属性走正常查找，避免递归
        return getattr(object.__getattribute__(self, "base"), name)

    def _proj_specs(self):
        d, f = self.dim, self.ffn_dim
        return {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d),
                "ffn_up_gate": (2 * f, d), "ffn_down": (d, f)}

    def _build_layer(self, i):
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.moss_image.model.moss_image_transformer.moss_image_block import (
            MOSSImageBlock,
        )

        lw = {}
        for shard in self.shards:
            w = mx.load(shard)
            lw.update({k[len(f"layers.{i}."):]: v for k, v in w.items()
                       if k.startswith(f"layers.{i}.")})
            del w
        if not lw:
            raise KeyError(f"layer {i} not found in {self.shards}")

        block = MOSSImageBlock.__new__(MOSSImageBlock)
        block.heads = self.num_heads
        head_dim = self.dim // self.num_heads
        for name in ("norm1", "norm2", "q_norm", "k_norm"):
            m = nn.RMSNorm(self.dim if name.startswith("norm") else head_dim, eps=1e-6)
            m.update({"weight": lw[name + ".weight"]})
            setattr(block, name, m)

        proj = {}
        for name, (out_dim, in_dim) in self._proj_specs().items():
            if self.quantized:
                q = nn.QuantizedLinear(in_dim, out_dim, bias=False,
                                       group_size=self.group_size, bits=self.bits)
                q.update({
                    "weight": lw[f"proj.{name}.weight"],
                    "scales": lw[f"proj.{name}.scales"],
                    "biases": lw[f"proj.{name}.biases"],
                })
            else:
                # bf16/fp16 非量化模型：直接用 nn.Linear，别拿 QuantizedLinear 去接
                q = nn.Linear(in_dim, out_dim, bias=False)
                q.update({"weight": lw[f"proj.{name}.weight"]})
            specs = self.lora.get(f"layers.{i}.proj.{name}")
            if specs:
                from mflux.models.common.lora.layer.fused_linear_lora_layer import (
                    FusedLoRALinear,
                )

                loras = []
                for a_t, b_t, alpha, scale in specs:
                    lin = LoRALinear.from_linear(q, r=a_t.shape[1], scale=scale)
                    lin.lora_A = a_t
                    lin.lora_B = b_t * (alpha / a_t.shape[1])
                    loras.append(lin)
                # 与 mflux 的叠加路径保持一致：单 LoRA 直接用 LoRALinear，
                # 多个走 FusedLoRALinear（base + sum(scale * x@A@B)），这样和非流式逐比特相同
                q = loras[0] if len(loras) == 1 else FusedLoRALinear(base_linear=q, loras=loras)
            proj[name] = q
        block.proj = proj
        return block

    def __call__(self, hidden_states, timestep, encoder_hidden_states, skip_layers=frozenset()):
        base = self.base
        batch, channels, height, width = hidden_states.shape
        modulation, head = base._modulation(timestep, batch)
        video = hidden_states.reshape(batch, channels, height * width).transpose(0, 2, 1)
        video = base.video_embedder(video.astype(modulation.dtype))
        rope = base.rope(height, width, encoder_hidden_states.shape[1], dtype=video.dtype)
        text = base.text_embedder(encoder_hidden_states, rope[:, height * width:])
        x = mx.concatenate([video, text], axis=1)
        for index in range(self.num_layers):
            if index in skip_layers:
                continue
            block = self._build_layer(index)
            x = block(x, rope, modulation)
            del block
            mx.clear_cache()
        x = base.final_norm_video(x[:, : height * width])
        x = x * (1 + head[:, 1:2]) + head[:, 0:1]
        x = base.final_linear_video(x)
        return x.transpose(0, 2, 1).reshape(batch, channels, height, width)
