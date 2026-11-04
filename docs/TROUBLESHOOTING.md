# 常见问题

## 出图是纯黑 / 117 字节的图

长 prompt（约 300 token 以上）会让 Qwen3.5 的线性注意力数值爆炸，embedding 出 NaN。
适配层在编码后检查非有限值并直接报错，不再产黑图。
处理：缩短 prompt，或把长描述拆成「主体 + 风格」两段。

## 生成很慢、swap 一直涨

确认没有强制单进程：两阶段是默认行为。也可能是另一个引擎的任务没退干净：

    ps aux | grep -E "mflux024_generate|msmlx_runner"

杀掉残留子进程，或从桌面壳退出（它会调 /api/shutdown 清理）。

## moss-image-mlx 报 safetensors 解析错误

macOS 的 AppleDouble 文件（._xxx）会被当成权重读。
下载脚本用 snapshot_download 一般不会有；从 Finder 拷过就会出现：

    dot_clean -m models/

## 端口被占用 / 打开页面是别的程序

工作台用 /api/health 做应用级握手，不会误复用别人的端口。
换端口：

    MOSS_PORT=7861 ./run.sh

## 桌面壳关窗后服务还在

壳会先请求 /api/shutdown（带本地 token），失败才 kill。
如果服务是终端里手动起的，壳只复用、不接管，关窗不会杀它 —— 这是预期行为。

## 后台窗口一段时间后 UI 卡死

macOS 的 App Nap 会冻结后台 WKWebView 的 JS 定时器。
本工作台改用父进程看护（壳活着服务就活着），不用空闲计时器，已规避。

## 升级 mflux 后出图崩了

适配层绑定 mflux 0.24 的内部签名。升级后先跑 4 步档回归；
候选版本先装进独立 venv 再切 MOSS_MFLUX_PY，不要直接升级主环境。

## 引擎显示不可用

用 doctor 看原因：

    .venv-mflux/bin/python scripts/doctor.py

典型：没装 msmlx 的 venv（install.sh --with-msmlx）、
或 MOSS_MFLUX_PY / MOSS_MSMLX_PY 指错。

## 超分报「权重缺失」

超分权重不随仓库分发。把对应权重放到 web/models/（见 web/upscale_registry.py 的 rel 字段），
或不用超分 —— 生成功能不受影响。

## 超分报「执行器未就绪（缺 xxx 或 pysafe）」

pysafe 是随仓库分发的几行包装脚本（tools/pysafe）：它在 torch 析构前 os._exit，
绕开 macOS 上 MetalShaderLibrary 的退出崩溃。缺它就四路超分全不可用。
确认 tools/pysafe 存在且有执行位；路径可以覆盖：MOSS_PYSAFE=/path/to/pysafe ./run.sh。
MewZoom 还要单独装官方库（它不在 spandrel 里）：pip install --no-deps mewzoom==1.0.0。
