#!/usr/bin/env bash
# MOSS-Image Workbench 安装脚本（Apple Silicon / macOS）
set -euo pipefail
cd "$(dirname "$0")"

PY="$PYTHON"
if [ -z "$PY" ]; then PY=python3; fi
WITH_MSMLX=0
NO_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --with-msmlx) WITH_MSMLX=1 ;;
    --no-models)  NO_MODELS=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
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
  .venv-mflux/bin/python scripts/download_models.py
fi

echo
echo "==> 完成。启动工作台："
echo "    ./run.sh"
echo "首次启动前建议先跑：.venv-mflux/bin/python scripts/doctor.py"
