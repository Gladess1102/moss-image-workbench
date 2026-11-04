# Changelog

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
