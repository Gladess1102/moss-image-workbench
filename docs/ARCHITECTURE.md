# 架构与设计决策

## 1. 进程模型：为什么一定要两阶段

16 GB 的统一内存要装下 5.6 GB 的 9B 文本编码器和 5.1 GB 的 DiT，
加上 MLX 的 buffer 池和 VAE，单进程峰值 13 GB —— 生成期持续写 swap。

两阶段把一条生成拆成两个进程：

    encode 阶段: 只加载 TE  -> 写出 embeds.safetensors -> 进程退出
    run    阶段: 只加载 DiT + VAE -> 读 embeds -> 出图

代价是多一次进程启动（约 1 s），收益是生成期 swap ≈ 0、任务不再被拖慢 2–3 倍。
两阶段的输出与单阶段逐像素一致 —— 这本身就是回归测试项：embedding 精度没对齐就会差 19 dB。

第二引擎 moss-image-mlx 不需要这个编排：它自己在 TE→DiT→VAE 三阶段释放内存，
所以 app.py 里 two_stage 是引擎固有行为，不接受前端开关。

## 2. mflux 0.24 适配层

上游 mflux 0.24 没有 MOSS-Image 的 MLX 实现，本仓库的 web/mflux024_adapter.py 补齐
（约 700 行）。升级 0.24 时踩到的 4 个接口断裂 + 5 个数值坑，最关键三个：

| 坑 | 现象 | 修法 |
|---|---|---|
| TE 的 RMSNorm 语义相反 | embedding 余弦 0.126，文本完全失效 | 按 mflux 的权重布局反向还原，余弦到 0.9998 |
| 跳步没推进 latents | 出图是模糊色块 | 复用上次预测后继续推进 latents |
| embedding 精度不一致 | 单/两阶段差 19.4 dB | 统一 fp32，两阶段逐像素相同 |

其它修掉的：time_embedding.1/.2 命名、Gated DeltaNet 的 conv1d 布局、VAE 的
OIHW→OHWI 与 bn. 前缀、to_out.0、TokenizerLoader 返回 dict、VAE 输出 NCHW、
mx.save_safetensors 自动追加后缀。

**升级 mflux 前必须先跑一遍 4 步档做回归**：适配层依赖 0.24 的内部类签名。

## 3. 自适应跳步

flow-match 的轨迹不是均匀的：前段（构图）变化剧烈，中段平缓，尾部（细节）又变陡。
所以固定比例跳步会硬跳关键步（25 步跳 7 步只剩 26.0 dB）。

自适应判据：

1. 跳过前 warmup 步（25 步档 8，50 步档 16）——构图区必须全算；
2. 跳过尾部 tail 步（25 步档 2，50 步档 4）——细节区必须全算；
3. 中段看相邻预测的余弦 >= cos 阈值（0.975）才跳，且不连续跳两步；
4. 被跳的那步用最近两个真实预测**线性外推**（不是直接复制）补轨迹曲率。

结果：25 步 334.7 s / 33.68 dB；50 步 760.3 s / 38.01 dB。

## 4. LoRA 键名

社区 LoRA 大多是 kohya / diffusers 格式，直接喂给 MLX runner 命中 0/756 个张量。
转换器把它们映射到 diffusion_model.* 的 756 个键（含 time_embedding、attn 的 qkv 合并）。
加载时校验命中数，0 命中直接报错而不是静默出糊图。

## 5. 流式加载

DiT 5.5 GB 常驻是 16 GB 机器的主要压力之一。流式模式按层 mmap，
常驻降到 0.5 GB，单步代价约 1.7%，输出逐像素一致（同 seed 对比过）。
前端默认开启，可用 MOSS_STREAM=0 关闭。

## 6. 提示词强化

用同一份 Qwen3.5-9B 在 DiT 加载之前改写 prompt，然后进程退出。
因为两者不同时驻留，额外内存为 0；它不是一次独立的 9B 模型加载。

## 7. 超分为什么必须独立进程

macOS 上 torch 的 MPS 后端在进程退出时会崩在 MetalShaderLibrary 析构。
所以主进程绝不 import torch：超分走 upscale_runner.py 子进程，外面再包一层 pysafe
（在 C 层析构前 os._exit）。

## 8. 服务端

单文件 ThreadingHTTPServer（web/app.py），路由：

    GET  /                前端
    GET  /api/health      应用级身份握手（端口被别的程序占用时不会误复用）
    GET  /api/meta        引擎 / 档位 / 超分模型 / 角色 LoRA
    GET  /api/sys         内存 / swap / CPU（2.5 s 轮询）
    GET  /api/history     出图历史
    POST /api/generate    提交生成任务
    POST /api/upscale     提交超分任务
    GET  /api/job?id=     任务进度
    GET  /api/shutdown    桌面壳关窗时调用（需本地 token）

任务在后台线程跑，GPU_LOCK 保证同一时刻只有一个推理；WebSocket 一律不用，
前端 2.5 s 轮询，简单且断线可恢复。
