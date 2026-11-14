# Changelog

## 0.1.5 — 2026-11-14

**agent 接口层进仓库 + npm 一键挂 MCP。**

### Added

- `agent/`：MCP stdio server（11 个 `moss_*` 工具）、零依赖客户端、CLI、机器契约 `contract.json`、自测与验收套件。此前只在开发版里存在。
- `docs/agent-interface.md`：agent 接口规范全文。
- `moss-wb mcp`：以 stdio 启动 MCP server，客户端配置只需一行 `npx -y moss-image-workbench mcp`，不用写 Python 路径。
- `moss-wb --help` 补上 MCP 用法。

### Fixed

- `moss_client._runtime_versions()` 之前只认开发版的 `venv-mflux024`，发布版的 `.venv-mflux` 探测不到，版本信息为空。
- `mcp_server.py` 文档里的示例路径不再写死作者的家目录。

## 0.1.4 — 2026-11-14

**npm 一键部署。**

### Added

- `package.json` + `bin/moss-wb.js`：`npx moss-image-workbench setup | start | doctor | models`。
  包体只有 ~13 KB 启动器：代码、venv 与 8 GB 权重仍落在本地目录（默认 `~/moss-image-workbench`）——
  npx 的缓存目录会被清理，不能把权重放那儿。
- `doctor` 逐项检查 macOS / arm64 / Python 3.11+ / `.venv-mflux` / 模型 / 端口占用。
- `setup --dir <路径>`、`--with-msmlx`、`--no-models`；`start --port <端口>`（等价 `MOSS_PORT`）。

### Notes

- npm 不接管 Python 依赖：`setup` 仍然调用仓库里的 `install.sh`（含 `scripts/download_models.py`）。
- 仅 macOS + Apple Silicon，与安装脚本一致。

## 0.1.3 — 2026-11-14

**文本编码 embeds 内容寻址缓存：同 prompt 重跑不再重新编码。**

### Added

- `~/.moss-image/embeds-cache/`：按 `sha256(prompt / 参与编码的 negative / TE 权重指纹)` 缓存 embeds，命中则跳过整个 encode 阶段。seed、尺寸、步数、档位不参与 key，因此「draft 512 验构图 → standard 1184×896 出片」第二步直接复用。
- `_embeds_ok()` 头校验 + 失败清理：kill -9 留下的半截文件不会被误命中；编码/去噪失败时删除对应条目。
- `_embeds_gc()`：LRU 保留 96 条。

### Measured

| 场景 | 未命中 | 命中 |
|---|---|---|
| draft 512² / 4 步（换 seed） | 48.6 s | 42.0 s（−13.6%） |
| standard 1184×896 / 4 步 | 189 s | 177.0 s |

改 prompt 不命中；收益集中在同 prompt 换 seed / 换尺寸 / 升档 / 复现旧稿。

## 0.1.2 — 2026-11-14

**VAE 解码期收紧内存；跳步判据实验（drift / hybrid）。**

### Fixed

- `decode()` 增加 `cache_limit`（默认 1 GB）：解码期收紧 MLX buffer 池，`finally` 还原。默认不再无限制——批量解码会把空闲池顶到系统压缩内存（实测 16 GB 机器被压出 10 GB swap）。
- mlx8 采样结束后先释放 DiT 再解码（`--keep-dit` 可关）：整模配置 1184×896/4 步 peak footprint 15.30 → 10.02 GB，流式 9.93 → 8.28 GB，**输出逐比特一致**。
- `msmlx_runner.py` 的 `--skip-mode uniform` 此前从未生效（只实现了 adaptive 分支），补上固定比例逻辑。

### Added

- `--decode-cache-gb`：解码期缓存上限（默认 0.25 GB），只影响峰值不影响像素。
- `--skip-mode drift|hybrid` + `--skip-drift`：按累计输出漂移预算跳步（MG-Cache / arXiv 2611.11365 思路）。25 步档实测与 adaptive 选点完全相同（10/25、1.59×、PSNR 36.33 dB），**仅作 CLI 实验入口**，不接入档位。

### Notes

- `--stream` 与非 stream 出图不一致（LoRA 应用路径不同：流式在加载时注入、整模在加载后 apply），此差异早于本版本；流式内部改前改后逐比特一致。

## 0.1.1 — 2026-11-05

**审计修复 18 条 + 从开发版回灌两项体验修复，另补 `DELETE` 的跨站校验。**

### Fixed — 阻断

- `install.sh` 在 `set -u` 下裸写 `$PYTHON` 直接 `unbound variable` 退出，README 第一条命令就跑不通；改为 `PY="${PYTHON:-python3}"`，CI 增加干净 shell 下的检查。

### Fixed — API 与安全

- `/api/generate`、`/api/upscale` 要求同源 `Origin` + `application/json`，并按 `MOSS_MAX_QUEUE` 限流；新增 `POST /api/cancel`。
- `DELETE /api/out/*` 也校验 Origin（同源请求通常不带 Origin，照常放行；跨站带 Origin 一律 403）。
- `_stream` 增加空闲读超时（`MOSS_STREAM_IDLE`）与 `proc.wait` 超时；`Popen` 失败时关闭 pty slave fd。
- `/api/job` 用 `parse_qs` 解析 id；`width/height/steps/cfg/seed` 走统一校验，非法参数返回 400 而不是断连接。
- `/out/`、`/thumb/`、`/api/out/` 的包含检查改用 `realpath`；`MOSS_OUT` 尾斜杠归一，不再让所有图片 404。
- shutdown token 改为 `/tmp/moss-shutdown-<pid>.token`，`O_CREAT|O_EXCL|O_NOFOLLOW` 0600，只在 `__main__` 创建。
- `tools/pysafe` 随仓库发布；`/api/meta` 的可用性同时检查 pysafe 与对应执行器；mewzoom 补进 requirements。

### Fixed — 任务与文件

- 每个任务的提示词 / embeds 放进 `tempfile.mkdtemp`，`finally` 整目录删除，不再往 `~/Pictures/MOSS-Image` 丢 `.prompt-*`。
- `unique_path` 在锁内占位，同一秒同 topic 的任务不再互相覆盖。
- 界面选「均匀跳步」时翻译成 CLI 的 `--skip-mode fixed`，不再静默跑自适应。
- mlx8 显式传 `--model <MOSS_MODELS>/mlx_vi_8bit`，runner 读 `MOSS_MODELS`；`/api/generate` 预检 tier 目录。
- `doctor --json` 恢复输出；`compress_history.py` 复用 `web/pngtools.py`。

### Changed — 从开发版回灌

- `per=0.0` 的 ETA 兜底：解析到 0.0 s/步时保留上一次有效值，2000×1120/50 步任务不再整场只显示「已用 Ns」。
- `--stream` 的生效值回传到任务与参数快照，进度行显示「· 流式」，灯箱显示「流式加载 / 整模加载」。

## 0.1.0 — 2026-11-02

首个公开版本：两阶段分进程、自适应跳步、DiT 流式加载、双引擎（mflux 0.24 / moss-image-mlx）、四路超分、WKWebView 桌面壳。
