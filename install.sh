#!/usr/bin/env bash
# MOSS-Image Workbench 安装脚本（Apple Silicon / macOS）
set -euo pipefail
cd "$(dirname "$0")"

# set -u 下裸写 "$PYTHON" 会直接 unbound variable 退出（干净 shell 里第一行就崩，
# 下面那个 if 回落永远到不了）
PY="${PYTHON:-python3}"
WITH_MSMLX=0
NO_MODELS=0
NO_4STEP=0
TIER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --with-msmlx) WITH_MSMLX=1 ;;
    --no-models)  NO_MODELS=1 ;;
    --no-4step)   NO_4STEP=1 ;;
    --tier)
      [ $# -ge 2 ] || { echo "--tier 需要一个值（compact|8bit）"; exit 1; }
      TIER="$2"; shift ;;
    --tier=*)     TIER="${1#--tier=}" ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
  shift
done

echo "==> Python: $("$PY" -V)"
"$PY" - <<'PYEOF'
import platform, sys
if sys.version_info < (3, 11):
    sys.exit("需要 Python 3.11+（推荐 3.13），当前: " + platform.python_version())
if sys.platform != "darwin":
    sys.exit("这个工作台只支持 macOS（依赖 MLX / Metal）")
if platform.machine() != "arm64":
    sys.exit("需要 Apple Silicon（arm64），检测到: " + platform.machine())
print("平台检查通过:", platform.machine(), platform.mac_ver()[0])
PYEOF

echo "==> 创建主引擎 venv (.venv-mflux)"
"$PY" -m venv .venv-mflux
.venv-mflux/bin/python -m pip install -U pip wheel
.venv-mflux/bin/python -m pip install -r requirements.txt

if [ "$WITH_MSMLX" = "1" ]; then
  echo "==> 创建第二引擎 venv (.venv-msmlx)"
  "$PY" -m venv .venv-msmlx
  .venv-msmlx/bin/python -m pip install -U pip wheel
  .venv-msmlx/bin/python -m pip install -r requirements-msmlx.txt
fi

if [ "$NO_MODELS" = "0" ]; then
  echo "==> 下载模型（约 8 GB，可中断重跑）"
  # --tier / --no-4step 透传给下载脚本（README 一直是这么写的，之前这里不认）
  DL_ARGS=""
  if [ -n "$TIER" ]; then DL_ARGS="--tier $TIER"; fi
  if [ "$NO_4STEP" = "1" ]; then DL_ARGS="$DL_ARGS --no-4step"; fi
  # shellcheck disable=SC2086  # 有意按空格拆分：只可能是 "--tier 8bit" / "--no-4step"
  .venv-mflux/bin/python scripts/download_models.py $DL_ARGS
fi

echo
echo "==> 完成。启动工作台："
echo "    ./run.sh"
echo "首次启动前建议先跑：.venv-mflux/bin/python scripts/doctor.py"
echo
echo "可选：桌面壳（WKWebView，关窗即清理推理进程）"
echo "    python3 -m pip install -r requirements-desktop.txt"
echo "    python3 web/desktop.py"
