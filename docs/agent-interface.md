# MOSS-Image Agent 接口规范 v1.1

> 面向"本机的 agent"：任何能跑 bash / 能 import Python / 能连 MCP 的智能体，
> 都可以按本文档调用这台 Mac 上的 MOSS-Image 端侧出图，**正确、无损、少走弯路**。
>
> 代码在 `agent/`，契约在 `agent/contract.json`，机器能力清单由 `capabilities()` 实时给出。
> 后端仍是原来的 `web/app.py`（HTTP + JSON），本层不修改它、不重编码任何图像。

---

## 0. 一分钟上手

```bash
# CLI 已装到 ~/.local/bin/mossimg（在 PATH 里），任意目录可用
mossimg capabilities | jq .backend,.presets

# 2) 出一张草稿（512²，约 1 分钟），返回单行 JSON，path 是绝对路径
mossimg generate "老城区深夜面馆，暖黄灯光，店主掀蒸笼盖，白汽升腾，纪实摄影" \
  --preset draft --seed 1234 | jq -r .path

# 3) 满意后升档重跑同一提示词
mossimg generate "..." --preset standard --seed 1234 | jq -r .path
```

Python：

```python
import sys; sys.path.insert(0, "/Users/yangyuxin/moss-image/agent")
import moss_client as mc
r = mc.generate("提示词", preset="draft", seed=1234)
print(r["path"], r["sha256"], r["elapsed"])
```

MCP（Claude Code / Cursor / 任意 MCP 客户端）：

```bash
python3 ~/moss-image/agent/mcp_server.py    # stdio，换行分隔的 JSON-RPC
```

---

## 1. 为什么不让 agent 直接 POST /api/generate

后端 `/api/generate` 只做"收下参数、排队、跑"，它面向的是网页前端。agent 直接调会遇到六个坑：

| 坑 | 后果 | 本层怎么解决 |
|---|---|---|
| 只返回 `job_id`，要自己轮询 `/api/job` | 每个 agent 都要写一套轮询、超时、续等逻辑 | `generate()` 阻塞到出图，超时抛 `E_JOB_TIMEOUT` 并带回可续等的 `job_id` |
| 参数错误只有一句中文（`unknown lora_mode`） | agent 无法分支处理 | 本地先校验，结构化错误码 + 修复建议 |
| 同 prompt 同 seed 重复提交 | 白烧 1–10 分钟 GPU | 幂等台账，命中直接返回原图（`cached: true`） |
| 只回 `filename`，不回绝对路径 | agent 拿不到文件 | 返回 `path` / `url` / `sha256` / 完整 `params` |
| 提示词写错（Layout JSON 键名错）**不报错** | 静默产出废图 | `validate_prompt()` 提交前拦截 |
| 7860 被别的服务占着就"服务不可用" | agent 无法自愈 | 自动发现 / 换端口 / 拉起后端 |

---

## 2. 架构与生命周期

```
agent 进程 ──┐
             ├─ moss_client (发现/校验/幂等/等待) ──HTTP──> web/app.py (agent 拉起用 7861；工作台用 7860)
MCP 客户端 ──┘        │                                        │ GPU_LOCK 串行
                      └─ ~/.moss-image/{agent-endpoint.json, agent-ledger.jsonl, logs/}
                                                               └─ 子进程：mflux024_generate / msmlx_runner / pysafe
```

- **发现顺序**：`MOSS_AGENT_URL` → `~/.moss-image/agent-endpoint.json` → 端口 `7860..7869` 的 `/api/health`
  （只认 `{"app": "moss-image"}`，别的服务占着就跳过）。
- **自动拉起**：都没有就 `python3 web/app.py` 起一个（独立会话、不随调用方退出），
  端点写进 `agent-endpoint.json`，之后所有 agent 复用同一个实例。
  `MOSS_AGENT_AUTOSTART=0` 可禁止。
- **端口策略**：自己拉起时**固定从 7861 起**（`MOSS_AGENT_PORT` 可改），
  **绝不占用 7860**——那是桌面工作台和 IndexTTS 的默认端口。
  发现阶段仍会复用 7860 上已经在跑的**工作台**实例（只认 `app == "moss-image"`）。
- **单实例 + 串行**：后端用 `GPU_LOCK` 一次只跑一个任务，排队上限 `MOSS_MAX_QUEUE`（默认 6）。
  并发提交不会并行烧 GPU，只会排队——这正是 16GB 机器上想要的行为。
- **空闲自退**：agent 拉起的实例默认空闲 1800s 退出（`MOSS_AGENT_IDLE_EXIT` 可调）。
  正在跑任务时不会退（watchdog 会看 busy）。
- **和 GUI 共用**：桌面工作台（`.command` / `MOSS-Image.app`）也读同一个 7860；
  谁先起来谁占，后起来的复用。**不要同时起两个后端**——两个任务并行会把 16GB 压进 swap。

---

## 3. 三种入口

### 3.1 CLI（任何能跑 bash 的 agent 首选）

默认输出**单行 JSON**；`--human` 给人看；错误打到 stderr 并以非 0 退出。

| 命令 | 作用 |
|---|---|
| `capabilities [--full]` | 能力清单（引擎/档位/presets/规则/角色 LoRA） |
| `generate "提示词" [--preset P] [--w --h] [--seed N\|random] [--negative ...]` | 出图（阻塞） |
| `batch "提示词" -n 4 [--seed-base N]` | 同 prompt 多变体（串行） |
| `upscale <图> [--model mewzoom]` | x4 超分（默认 mewzoom，约 7s） |
| `list [-n 20] [--since 分钟] [--prompt-contains X]` | 最近产物 |
| `inspect <图>` | 读参数快照（复现依据） |
| `replay <图> [--no-compare]` | 原样重跑并比对 sha256 |
| `vary <图> -n 3` | 同参数换 seed 出变体 |
| `validate-prompt [-f 文件] [--w --h]` | 提交前校验提示词 |
| `layout --summary "..." --item '{...}'` | 生成合规 Layout JSON |
| `status` / `cancel <job_id>` / `stop` | 后端管理 |
| `prompt-guide` | 打印提示词规范速查 |

`generate` 还支持 `-f prompt.txt`（长 Layout JSON 走文件）、`--timeout`、`--no-dedupe`、
`--idem-key`、`--preview out.jpg`（另存缩略图）、`--enhance`。

### 3.2 Python（在 agent 自己的进程里）

```python
import moss_client as mc

mc.capabilities()                    # 能力清单（60s 缓存）
mc.validate_prompt(text, 1184, 896)  # 结构校验，返回 ok/errors/warnings/stats
mc.generate(prompt, preset="draft", seed=42, on_progress=print)
mc.generate_batch(prompt, n=4, seed_base=100)
mc.upscale("文件名.png", model="mewzoom")
mc.inspect("文件名.png"); mc.replay("文件名.png"); mc.vary("文件名.png", n=3)
mc.list_outputs(limit=10, since=time.time()-3600)
mc.thumbnail(path, 768)              # 返回 JPEG bytes（给多模态 agent 看）
mc.backend_info()                    # 后端状态（不抛异常）
```

全部函数见 `agent/moss_client.py` 的 docstring；错误统一是 `mc.MossError`（`code` / `message` / `hint` / `detail`）。

### 3.3 MCP

```bash
python3 ~/moss-image/agent/mcp_server.py
```

工具：`moss_capabilities` `moss_generate` `moss_generate_batch` `moss_upscale`
`moss_list_outputs` `moss_inspect` `moss_replay` `moss_validate_prompt`
`moss_build_layout` `moss_cancel` `moss_backend_status`。

`moss_generate` 每次返回三样东西：文本 JSON、`resource_link`（`file://` 指向无损原图，
客户端可直接读文件）、以及 `structuredContent`。
传 `include_preview=true` 会**额外**附一张 768px 的 JPEG 预览（image content block），
多模态 agent 可以"看一眼再决定要不要改提示词/升档"——预览默认关闭，避免每次调用都吃上下文。

### 3.4 自检（改完代码跑一遍）

```bash
python3 agent/e2e_test.py            # 静态 + 契约 + 校验矩阵 + CLI + MCP + 容错（约 1 秒）
python3 agent/e2e_test.py --gpu      # 再加真实出图 / 幂等 / 复现 / 超分（约 3 分钟）
python3 agent/e2e_test.py --tiers    # 再加 standard / quality / precise 三档实测（30 分钟+）
```

套件会自己检查内存与 swap（free < 20% 直接 SKIP 而不是硬跑），并等后端空闲后再提交，
所以可以放心在别的任务跑着的时候执行。失败会打印"期望 / 实际"，不中断后续用例。

---

## 4. 参数契约

| 参数 | 类型 | 默认 | 约束 / 陷阱 |
|---|---|---|---|
| `prompt` | string | 必填 | caption 或四键 Layout JSON；建议 1300–2600 字符（编码器 4096 token） |
| `preset` | enum | `draft`（未指定时） | `draft` / `standard` / `quality` / `precise`，见 §7 |
| `width`,`height` | int | 1184×896 | **必须是 16 的倍数**，256..4096；1MP 左右最贴训练分布 |
| `negative` | string | 官方负面词 | **akk4 档 CFG=1.0 → 完全不参与计算**；要生效用 `quality`/`precise` |
| `steps` | int | 档位默认 | 1..200（本层保守上限；mflux 硬边界 218，>218 直接报错）；**turbo 档忽略** |
| `cfg` | float | 档位默认 | 0..20；**turbo 档忽略**；=1.0 时负面词失效 |
| `seed` | int 或 `random` | random | 0..2³¹-1；整数才有幂等语义；实际用值写进快照 |
| `engine` | enum | `mlx8` | `mlx8`（mflux 0.24 两阶段）/ `msmlx`（moss-image-mlx）；turbo 强制 msmlx |
| `lora_mode` | enum | `akk4` | `akk4` `turbo` `native25` `none` `full` |
| `char_lora` | string | null | 文件名必须在 `capabilities().char_loras`；**turbo 不支持**；触发词要自己写进 prompt |
| `char_lora_scale` | float | 1.0 | 0..2 |
| `enhance` | bool | false | 本地 Qwen3.5-9B 强化提示词，+10~90s，失败自动回退原文 |
| `skip` | float 或 `adaptive` | 档位 | 0..0.6，仅 steps≥10 生效；一般别动 |
| `stream` | bool | true | 仅 mlx8：DiT 按层 mmap，省 ~5GB 常驻，输出逐像素一致 |

后端还有几个**故意不接受外部开关**的项：`two_stage`（mlx8 恒两阶段、msmlx 恒单进程）。
历史事故：把 mlx8 切成单进程会让峰值 13GB、swap 14.7GB、所有任务慢 2–3 倍。

---

## 5. 结果与无损保证

```json
{
  "ok": true,
  "path": "/Users/yangyuxin/Pictures/MOSS-Image/一位穿深蓝围裙的店主正弯_20261108-145845.png",
  "filename": "...png",
  "url": "http://127.0.0.1:7860/out/...png",
  "bytes": 335949,
  "sha256": "4e2f97...",
  "width": 512, "height": 512,
  "elapsed": 55.8,
  "job_id": "731f0e5d554a",
  "params": {"prompt": "...", "engine": "mlx8", "lora_mode": "akk4", "steps": 4,
             "cfg": 1.0, "seed": 1234, "width": 512, "height": 512, "negative": ""},
  "meta": {"type": "generate", "seed": 1234, "skip": 0.0, "two_stage": true},
  "warnings": [],
  "cached": false
}
```

无损的三条硬保证：

1. **给路径，不给像素**。产物是磁盘上的原始 PNG，客户端从不重新编码、不二次压缩、
   不 base64 化（预览图另存 `~/.moss-image/cache/`，绝不写进输出目录）。
2. **参数完整回传**。`params` 是本次**实际生效**的参数（含后端当场摇出的 seed），
   `meta` 是后端写的同名 `.json` 快照原文——两者都在，才能既复现又对账。
3. **`sha256` 随结果返回**。要做"这张是不是那张"的判断，比哈希，不要比文件大小。

---

## 6. 幂等 / 复现 / 变体

### 幂等

- 键 = `sha256(prompt, negative, engine, lora_mode, width, height, steps, cfg, seed, char_lora, char_lora_scale)[:32]`
- `seed` 为整数时默认开启；`seed="random"` 自动关闭（随机没有幂等语义）。
- 命中已完成 → 直接返回原图，`cached: true`，亚秒级（实测约 0.1s，不再重算哈希）。
- 命中**正在跑** → 挂到同一个 job 一起等，不会重复占 GPU。
- 要强制重跑：`dedupe=False` / `--no-dedupe` / 传自己的 `idempotency_key`。

台账在 `~/.moss-image/agent-ledger.jsonl`（append-only），删掉不影响出图，只丢幂等与 ETA 样本。

### 复现

```bash
./agent/mossimg replay <图>          # 读快照 → 原样重跑 → 比 sha256
```

结果里 `identical: true` 表示**逐字节一致**；`false` 会附 `W_NOT_IDENTICAL` 警告
（通常是引擎/跳步/量化在这期间改过）。

**identical 的承诺边界**：只在「同引擎 + 同两阶段开关 + 同量化/权重」下成立。
`mlx8`（mflux 0.24 适配层）与 `msmlx`（moss-image-mlx）是**两套独立实现**，
跨引擎只保证构图意图接近，不保证逐字节。`capabilities().version` 里有
`mflux / mlx / quantize / uncond_skip_layers`，对不上就别指望 identical。

### 变体

```bash
./agent/mossimg vary <图> -n 3       # 同 prompt/参数，seed+1,+2,+3
```

### 多个 agent 同时调用

| 场景 | 行为 |
|---|---|
| 两个 agent 同时提交**同参数同 seed** | 只生成一次：一个真跑、另一个挂到同一 job 一起等，两边拿到同一张图（同一 path/sha256） |
| 两个 agent 提交**不同参数** | 后端 `GPU_LOCK` 串行：一个在跑、另一个排队，不会并行烧 GPU |
| 排队超过上限 | 第 7 个起返回 `E_QUEUE_FULL`（`MOSS_MAX_QUEUE` 可调） |

实现上：命中检查与"提交"在同一把跨进程锁（`~/.moss-image/agent-generate.lock`）里完成，
**等待任务在锁外**——所以 A 在等一张 10 分钟的 turbo 图时，B 的 draft 提交不会被锁住。

---

## 7. 档位怎么选

| preset | 构成 | 尺寸 | 实测/估计耗时 | 什么时候用 |
|---|---|---|---|---|
| `draft` | akk4 4 步 CFG1 | 512×512 | **实测 56s** | 先验证构图与措辞；确定后升档 |
| `standard` | akk4 4 步 CFG1 | 1184×896 | **实测 189s** | 日常出片；文字最稳；负面词无效 |
| `quality` | turbo 15 次前向 CFG3 | 1184×896 | **实测 676s（11.3 分钟）** | 要质量、要负面词、能等 |
| `precise` | native25 25 步 CFG3 + 自适应跳步 | 1184×896 | **未测完**（1MP 上跑约 20 分钟、swap 风险高，验收时中止；640×480 实测 335s） | 复杂 Layout JSON、要负面词、要可控 |

> 耗时口径：本机 M1/16GB、单任务独占、`mlx8` 两阶段；并发排队时等待时间不计入 `elapsed`。
> 1024² draft 实测 200.6s（约 512² 的 3.9 倍），可用来给别的尺寸做线性外推。

**决策建议**：不确定就先 `draft` 跑 1–2 张确认提示词，再 `standard`/`quality` 定稿。
别一上来就 `quality`——10 分钟一张，提示词写错就是纯亏。

**默认档位就是 `draft`**：不传 `preset` 时按 512² 草稿跑（会带 `W_DRAFT_DEFAULT` 警告）。
这条是刻意反直觉的——宁可 agent 先花 1 分钟看构图，也别一句话就烧掉 3–10 分钟。

---

## 8. 提示词规范（agent 最容易翻车的地方）

### 8.1 两种格式，二选一

1. **纯文本 caption**
2. **四键 Layout JSON**

```json
{
  "summary": { "caption": "整体：风格/场景/主光/色调/质感/镜头语言", "canvas": "single" },
  "items": [
    { "label": "character", "caption": "该单元的外观", "bbox": [y1, x1, y2, x2] }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

硬规则（写错**不会报错**，只会变成被模型忽略的废话——所以务必先 `validate-prompt`）：

- 顶层只有这四个键；`schema`/`width`/`negative` 这类自定义字段一律不认。
- `bbox_coord_space` 必须 `abs_centralized`；`bbox_format` 必须 `yxyx`。
- `bbox` 相对**画布中心**、单位像素、顺序 `y1,x1,y2,x2`；画布左上角是负 y、负 x。
- 换输出尺寸后 bbox 必须按新画布中心**等比缩放**，否则区域全错位。
- `items` 建议 5–6 个；同一件东西只在一个 item 里详写（重复会出双份）。
- 尺寸不在 JSON 里，走 `width`/`height` 参数。

拿不准就用工具生成，别手写：

```bash
mossimg layout --summary "夜景街道，暖黄路灯" --w 1184 --h 896 \
  --item '{"label":"character","caption":"三十岁上下的店主，深蓝围裙，正弯腰掀蒸笼盖"}' \
  --item '{"label":"object","caption":"青石板上反光的水渍与木凳"}' \
  --item '{"label":"object","caption":"模糊街巷与远处霓虹"}' -o /tmp/p.json
```

### 8.2 caption 的写法

- 顺序：**位置锚点 → 主体外形 → 服装与道具 → 动作与表情 → 材质细节 → 光**。
- 长度：建议 1300–2600 字符（**本地实测口径**，官方只说支持到 4K token）；
  本层用 `est_tokens` 估算（CJK 约 1 字 1 token），超 4096 直接报 `E_PROMPT_TOO_LONG`。
- 年龄写"三十岁上下"；要"大叔感"靠体型、胡茬、油污、疲惫感堆，不靠年龄词。
- **姿态要动**：蹲、扑、跳、够、拧——比"坐着看镜头"出片率高得多。

### 8.3 负面词与 LoRA

- 负面词是**独立参数**，绝不塞进 JSON；只在 `cfg != 1.0` 时参与计算（`akk4` 无效）。
- 角色 LoRA 的**触发词必须自己写进 caption 开头**；挂权重不会替你补触发词。

---

## 9. 错误码

| code | 含义 | agent 该做什么 |
|---|---|---|
| `E_PROMPT_EMPTY` | 提示词为空 | 补提示词 |
| `E_PROMPT_TOO_LONG` | 超上下文 | 砍到 3000 字符内 |
| `E_BAD_LAYOUT_JSON` | JSON 结构不对 | 看 `detail.errors`，或用 `layout` 重新生成 |
| `E_SIZE` | 宽高不是 16 倍数 | 用 `hint` 里的建议值 |
| `E_UNKNOWN_PRESET` / `E_UNKNOWN_LORA_MODE` / `E_UNKNOWN_ENGINE` | 名字写错 | 先 `capabilities()` |
| `E_ENGINE_UNAVAILABLE` | 本机缺 venv/权重 | 换引擎或补环境 |
| `E_UNKNOWN_LORA` / `E_LORA_UNSUPPORTED` | LoRA 文件不存在 / 档位不支持 | 看 `char_loras`，或换非 turbo 档 |
| `E_QUEUE_FULL` | 队列满 | 等一会儿或先 `cancel` |
| `E_BACKEND_DOWN` | 后端不可用 | 去掉 `MOSS_AGENT_AUTOSTART=0`，或手动起 `web/app.py` |
| `E_BACKEND_START_TIMEOUT` | 拉起超时 | 看 `~/.moss-image/logs/agentd-*.log` |
| `E_JOB_TIMEOUT` | 等待超时 | **别重跑**：`wait_job(job_id)` 续等（异常对象带 `job_id`） |
| `E_JOB_FAILED` | 后端任务失败 | `detail` 是子进程日志尾部；多半是内存或参数组合 |
| `E_CANCELLED` | 被取消 | — |
| `E_OUTPUT_MISSING` | 产物路径找不到 | `MOSS_OUT` 与后端不一致；看 `capabilities().backend.out_dir` |
| `E_UPSCALE_SRC` | 超分源图不在输出目录 | 先 `generate` 或把图放进输出目录 |
| `E_BAD_ARGS` | MCP/CLI 层参数类型不对（提交前拦下） | 按 `hint` 改类型；这是**接口层**错误，不是后端拒绝 |
| `E_INTERNAL` | 客户端内部异常 | 属于 bug：把命令与 `code` 一起反馈 |

**警告码**（`warnings[]`，不阻断出图，但值得 agent 读一眼）：
`W_NEG_IGNORED`（CFG=1 负面词无效）、`W_NEG_DEFAULT`（用了官方默认负面词）、
`W_TURBO_IGNORES_STEPS`、`W_PROMPT_LONG` / `W_PROMPT_SHORT`、
`W_LAYOUT_EXTRA_KEYS` / `W_LAYOUT_SUMMARY_KEYS` / `W_LAYOUT_ITEM_KEYS`（键会被忽略）、
`W_LAYOUT_LABEL`（label 不在 object/character/text）、`W_LAYOUT_CANVAS`、`W_LAYOUT_ITEMS`、
`W_LAYOUT_BBOX_ORDER` / `W_LAYOUT_BBOX_RANGE`、`W_LAYOUT_NEGATIVE`、`W_NOT_IDENTICAL`、
`W_STALE_LEDGER`（同参数旧图因版本/过期失效，本次重跑）、`W_DRAFT_DEFAULT`（没指定 preset，按 512² 草稿出）。

---

## 10. 集成配方

### 10.1 任意 shell / agent 工具

```bash
cd ~/moss-image
path=$(./agent/mossimg generate "$PROMPT" --preset draft --seed 42 | jq -r .path)
```

退出码约定：

| 退出码 | 含义 | 输出 |
|---|---|---|
| 0 | 成功 | stdout 单行 JSON |
| 2 | **业务/参数错误**（`MossError`，含校验失败、后端拒绝、任务失败） | stderr 单行 JSON `{"ok":false,"code":...}` |
| 1 | 环境/内部错误（如 `status` 时后端不可用、未预期异常） | stderr 单行 JSON 或异常文本 |
| 2 | argparse 层参数错误（如 `--steps abc`、未知子命令） | stderr 是 **usage 文本**（不是 JSON），这是 argparse 的行为 |

所以脚本里判成功请只看 `exit 0` + stdout 可解析，不要假设失败一定是 JSON。

### 10.2 Python agent

```python
import sys
sys.path.insert(0, "/Users/yangyuxin/moss-image/agent")
import moss_client as mc

caps = mc.capabilities()
try:
    r = mc.generate(prompt, preset="draft", seed=42, timeout=300)
except mc.MossError as e:
    if e.code == "E_JOB_TIMEOUT":
        r = mc.wait_job(e.job_id, timeout=1800)     # 续等，不重跑
    elif e.code == "E_BAD_LAYOUT_JSON":
        prompt = mc.build_layout_json(...)           # 重建合规 JSON
    else:
        raise
```

### 10.3 MCP 客户端

Claude Code：

```bash
claude mcp add moss-image -- python3 /Users/yangyuxin/moss-image/agent/mcp_server.py
```

Cursor / 通用 JSON 配置（`mcp.json`）：

```json
{
  "mcpServers": {
    "moss-image": {
      "command": "python3",
      "args": ["/Users/yangyuxin/moss-image/agent/mcp_server.py"],
      "env": {"MOSS_AGENT_AUTOSTART": "1"}
    }
  }
}
```

DSH：**已注册为技能 `moss-image`**（`~/.dsh/skills/moss-image/SKILL.md`）——所有 DSH 会话的技能目录里都能看到它，
问到出图会自动加载，按里面的三条入口用即可。
DSH 的 MCP 能力由插件提供（`dsh-mcp-client` 一类），不是 `cordis.patch.yml` 里的裸配置；
在 DSH 里最省事的路径是 CLI（`mossimg generate ...`）或 `import moss_client`。

### 10.4 让 agent 自己发现这套工具

- 仓库根的 `AGENTS.md` 写了"要出图先看 `agent/README.md`"；
- `./agent/mossimg prompt-guide` 是给模型读的提示词速查；
- `agent/contract.json` 是机器可读的完整契约。

---

## 11. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `E_BACKEND_DOWN` 且 7860 上不是工作台 | 端口被 IndexTTS 等其它服务占了 | 正常：客户端会自动换到 7861+；或给后端指定 `MOSS_PORT=7862` |
| 任务成功但 `E_OUTPUT_MISSING` | 客户端 `MOSS_OUT` 与后端不一致 | 导出同样的 `MOSS_OUT`，或看 `capabilities().backend.out_dir` |
| 出图很慢 / swap 暴涨 | 同时有两个后端在跑，或并行提交 | `./agent/mossimg status` 看 `running`/`queued`；确认只有一个 `app.py` 进程 |
| `E_JOB_FAILED` 带 `Insufficient Memory` | 1MP+ 大任务 + 其它内存大户 | 先降尺寸或换 `draft`，关掉别的模型 |
| Layout JSON 图全错位 | bbox 没按新画布缩放 | 用 `layout --w --h` 重新生成 |
| 负面词"没用" | 用了 `akk4`（CFG=1.0） | 换 `quality` / `precise` |

---

## 12. 已知限制

1. **单任务串行**：后端 `GPU_LOCK` 一次只跑一个；`batch` 是串行的，`n=4` 就是 4 倍时间。
2. **跨版本不保证逐像素复现**：引擎/跳步/量化改动会让 `replay` 的 `identical` 变 false（会明确告知）。
3. **`turbo` 不支持角色 LoRA**，也不接受 `steps`/`cfg` 覆盖（固定调度）。
4. **超分是后处理**，不改文字；SeedVR2 已从本机移除，当前可用：`nomos2_hq` / `mewzoom` / `thera_air` / `anime6b`。
5. **提示词校验是结构性的**，不判断"画得好不好看"——那要靠 `draft` 先出图看。
6. 后端 `web/app.py` 仍是唯一执行体；本层只包装，不改它的行为（因此在 daily / workbench 两个版本上都可用）。
7. **档位不是官方口径**：官方默认 50 步 / CFG 3 / shift 5，推荐 1M–2M 像素；
   本机的 `draft`/`standard` 是 4 步蒸馏档（CFG=1，负面词无效），`quality` 是 turbo 15 次前向。
   这些是本机 16GB 的取舍，不是模型官方推荐值。
8. `scheduler` / `lora_style` / `step_cache_ratio` / `two_stage` 传了也不生效——
   清单在 `capabilities().unsupported`，别指望调它们改行为。

---

*契约版本 1.1.0 · 2026-11-08 · 代码：`agent/moss_client.py` `agent/mossimg.py` `agent/mcp_server.py` · 机器契约：`agent/contract.json`*
