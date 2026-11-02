#!/usr/bin/env bash
# 启动工作台（终端模式）
set -euo pipefail
cd "$(dirname "$0")"
exec .venv-mflux/bin/python web/app.py
