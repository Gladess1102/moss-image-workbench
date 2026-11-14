# agent/ —— 给本机 agent 的出图接口

**一句话**：任何 agent 用一行命令就能让这台 Mac 出图，拿回磁盘上的绝对路径 + 完整参数快照。

```bash
mossimg capabilities                                  # 先看有什么（已装到 ~/.local/bin）
mossimg generate "提示词" --preset draft --seed 42     # 出图，约 1 分钟
mossimg generate "提示词" --preset quality --seed 42   # 要质量，约 10 分钟
```

默认输出**单行 JSON**：`{"ok":true,"path":"/Users/.../xxx.png","sha256":"...","params":{...},"warnings":[]}`

## 文件

| 文件 | 作用 |
|---|---|
| `moss_client.py` | 核心客户端（零依赖）：发现/拉起后端、参数校验、阻塞提交、幂等、复现、变体、超分 |
| `mossimg.py` / `mossimg` | CLI 入口（默认单行 JSON，`--human` 给人看） |
| `mcp_server.py` | MCP stdio server（`moss_generate` 等 11 个工具） |
| `contract.json` | 机器可读契约：参数、档位、错误码、不变量 |
| `mcp_selftest.py` | MCP 协议自测（不出图） |
| `e2e_test.py` | 验收套件：契约/校验/CLI/MCP/容错，`--gpu` / `--tiers` 加真实出图 |

## 三条铁律（写进任何 agent 的提示词都成立）

1. **先 `capabilities()` 再猜参数**：档位/引擎/角色 LoRA/尺寸规则都在里面。
2. **不确定就先 `--preset draft`**：56 秒一张，确认构图与措辞再升档，别一上来烧 10 分钟。
3. **拿 `path` 用，不要重编码**：产物是原始 PNG，旁边有同名 `.json` 快照；要复现用 `replay`。

## 常见陷阱

- 宽高**必须是 16 的倍数**（`1184x896` 是训练比例）。
- `akk4` 档 CFG=1.0 → **负面词完全不生效**；要负面词用 `quality` / `precise`。
- `turbo` 档忽略 `steps`/`cfg`，也不支持角色 LoRA。
- Layout JSON 顶层只能四个键；写错键名**不报错**，只会变成被模型忽略的废话——先 `validate-prompt`。
- 同 prompt + 同 seed 默认**幂等命中**（0 秒返回原图）；要重跑加 `--no-dedupe`。

完整规范：`docs/agent-interface.md`。提示词速查：`./agent/mossimg prompt-guide`。
