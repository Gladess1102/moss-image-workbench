#!/usr/bin/env node
"use strict";

/**
 * moss-wb —— MOSS-Image Workbench 的 npm 启动器。
 *
 * 这个仓库是 Python + MLX 项目，npm 包只做「环境自举 + 启动」：
 *   npx moss-image-workbench setup   # git clone + ./install.sh（含 ~8GB 模型）
 *   npx moss-image-workbench start   # 起服务（默认 http://127.0.0.1:7860）
 *   npx moss-image-workbench doctor  # 环境自检
 *
 * 为什么 setup 要 clone 到别处：npx 的包目录在 ~/.npm/_npx 里，会被缓存清理
 * 扫掉；venv 和模型动辄几个 GB，放那儿等于随时准备重装。
 */

const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const REPO = "https://github.com/Gladess1102/moss-image-workbench.git";
const DEFAULT_DIR = path.join(os.homedir(), "moss-image-workbench");
const DEFAULT_PORT = "7860";

const argv = process.argv.slice(2);
const cmd = argv[0] && !argv[0].startsWith("-") ? argv[0] : "start";
const rest = cmd === "start" && (!argv[0] || argv[0].startsWith("-")) ? argv : argv.slice(1);

function opt(name, fallback) {
  const i = rest.indexOf(name);
  if (i >= 0 && rest[i + 1]) return rest[i + 1];
  const eq = rest.find((a) => a.startsWith(name + "="));
  return eq ? eq.slice(name.length + 1) : fallback;
}
function flag(name) {
  return rest.includes(name);
}
const say = (s) => console.log("moss-wb " + s);
const die = (s) => { console.error("moss-wb: " + s); process.exit(1); };

function run(file, args, opts = {}) {
  const r = spawnSync(file, args, { stdio: "inherit", ...opts });
  if (r.error) return 127;
  return r.status === null ? 0 : r.status;
}
function capture(file, args) {
  const r = spawnSync(file, args, { encoding: "utf8" });
  if (r.error || r.status !== 0) return null;
  return (r.stdout || "").trim();
}

// ---------------------------------------------------------------- 探测

function findCheckout(start) {
  let d = path.resolve(start || process.cwd());
  for (;;) {
    if (fs.existsSync(path.join(d, "web", "app.py"))) return d;
    const up = path.dirname(d);
    if (up === d) return null;
    d = up;
  }
}
function checkoutDir() {
  const explicit = opt("--dir");
  if (explicit) {
    const d = path.resolve(explicit);
    if (!fs.existsSync(path.join(d, "web", "app.py"))) {
      die("--dir " + d + " 里没有 web/app.py，不像工作台目录");
    }
    return d;
  }
  return findCheckout() || (fs.existsSync(path.join(DEFAULT_DIR, "web", "app.py")) ? DEFAULT_DIR : null);
}
function venvPython(dir) {
  const p = path.join(dir, ".venv-mflux", "bin", "python");
  return fs.existsSync(p) ? p : null;
}
function pythonVersion() {
  for (const py of ["python3", "python"]) {
    const v = capture(py, ["-c", "import sys,platform;print('%d.%d|%s|%s' % (sys.version_info[0], sys.version_info[1], sys.platform, platform.machine()))"]);
    if (v) {
      const [ver, plat, arch] = v.split("|");
      return { bin: py, ver, plat, arch };
    }
  }
  return null;
}
function modelsPresent(dir) {
  // 主模型（约 8 GB）：transformers 目录在就算装过；缺了由 download_models.py 补
  for (const rel of ["models/mlx_vi_8bit/transformer", "models/mlx_moss_sft/transformer"]) {
    if (fs.existsSync(path.join(dir, rel))) return true;
  }
  return false;
}
function portBusy(port) {
  const r = spawnSync("lsof", ["-nP", "-iTCP:" + port, "-sTCP:LISTEN"], { encoding: "utf8" });
  return r.status === 0 && (r.stdout || "").trim().length > 0;
}

// ---------------------------------------------------------------- 子命令

function doctor() {
  const py = pythonVersion();
  const dir = checkoutDir();
  const port = opt("--port", process.env.MOSS_PORT || DEFAULT_PORT);
  let bad = 0;
  const row = (ok, label, detail) => {
    console.log((ok ? "  ✓ " : "  ✗ ") + label + (detail ? "  — " + detail : ""));
    if (!ok) bad++;
  };

  console.log("moss-wb doctor\n");
  row(process.platform === "darwin", "macOS", process.platform);
  row(os.arch() === "arm64", "Apple Silicon", os.arch());
  row(!!py, "Python 3.11+", py ? py.bin + " " + py.ver : "未找到 python3");
  if (py) {
    const [maj, min] = py.ver.split(".").map(Number);
    row(maj > 3 || (maj === 3 && min >= 11), "版本达标", py.ver);
  }
  row(!!dir, "工作台目录", dir || "未找到（先跑 moss-wb setup）");
  if (dir) {
    row(!!venvPython(dir), "主引擎 venv (.venv-mflux)", venvPython(dir) || "缺（跑 ./install.sh）");
    row(modelsPresent(dir), "模型权重", modelsPresent(dir) ? "已就位" : "缺（scripts/download_models.py）");
  }
  row(true, "端口 " + port, portBusy(port) ? "已被占用（可能是已在跑的工作台）" : "空闲");
  console.log("");
  if (bad) die(bad + " 项不满足");
  say("环境检查通过");
}

function setup() {
  const dir = path.resolve(opt("--dir", DEFAULT_DIR));
  if (!fs.existsSync(path.join(dir, "web", "app.py"))) {
    if (fs.existsSync(dir) && fs.readdirSync(dir).length) {
      die(dir + " 已存在且不是空的，换一个 --dir");
    }
    say("克隆到 " + dir);
    if (run("git", ["clone", "--depth", "1", REPO, dir]) !== 0) die("git clone 失败");
  } else {
    say("已存在工作台目录，跳过克隆：" + dir);
  }
  const args = [];
  if (flag("--with-msmlx")) args.push("--with-msmlx");
  if (flag("--no-models")) args.push("--no-models");
  return run("bash", ["install.sh", ...args], { cwd: dir, stdio: "inherit" });
}

function start() {
  const dir = checkoutDir() || die("没找到工作台目录：先跑 moss-wb setup，或用 --dir 指定");
  const py = venvPython(dir) || die("缺 " + path.join(dir, ".venv-mflux") + "：先跑 ./install.sh 或 moss-wb setup");
  const port = opt("--port", process.env.MOSS_PORT || DEFAULT_PORT);
  const env = { ...process.env, MOSS_PORT: port };
  const child = spawn(py, ["web/app.py"], { cwd: dir, env, stdio: "inherit" });
  const stop = (sig) => { try { child.kill(sig); } catch (_) {} };
  process.on("SIGINT", () => stop("SIGINT"));
  process.on("SIGTERM", () => stop("SIGTERM"));
  child.on("exit", (code) => process.exit(code === null ? 0 : code));
}

function models() {
  const dir = checkoutDir() || DEFAULT_DIR;
  console.log("主模型约 8 GB，走 Hugging Face：");
  console.log("  cd " + dir);
  console.log("  .venv-mflux/bin/python scripts/download_models.py            # 完整");
  console.log("  .venv-mflux/bin/python scripts/download_models.py --tier compact  # 小盘");
  console.log("");
  console.log("也可以用你自己的镜像：HF_ENDPOINT=https://hf-mirror.com ...");
}

function help() {
  console.log(`moss-wb — MOSS-Image Workbench 启动器

  moss-wb setup [--dir <路径>] [--with-msmlx] [--no-models]
      克隆仓库并安装（默认 ~/moss-image-workbench，含约 8 GB 模型）

  moss-wb start [--dir <路径>] [--port <端口>]
      启动工作台（默认 7860）

  moss-wb doctor [--dir <路径>] [--port <端口>]
      环境自检：平台 / Python / venv / 模型 / 端口

  moss-wb models
      打印模型下载命令

只支持 macOS + Apple Silicon（依赖 MLX / Metal）。`);
}

const table = { setup, start, doctor, models, help, "-h": help, "--help": help };
if (!table[cmd]) {
  console.error("未知子命令：" + cmd);
  help();
  process.exit(1);
}
const code = table[cmd]();
if (typeof code === "number" && code !== 0) process.exit(code);
