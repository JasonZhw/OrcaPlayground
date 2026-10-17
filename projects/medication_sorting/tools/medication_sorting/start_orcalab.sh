#!/usr/bin/env bash
# 本任务启动入口；保留已有资产缓存和菜单，不修改 HOME 或 Conda 环境。
set -euo pipefail

TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PROJECT_ROOT="$(cd -- "$TASK_SCRIPT_DIR/../.." && pwd)"
export SY_ROOT="$TASK_PROJECT_ROOT"
# 将 .venv/bin 加到 PATH 开头，供 OrcaLab 内部 python3/PySide6 检测使用。
source "$TASK_PROJECT_ROOT/tools/orca/environment.sh"

if [[ -f "$ORCA_SYSTEM_LIB_DIR/libxcb-cursor.so.0" ]]; then
  export LD_PRELOAD="$ORCA_SYSTEM_LIB_DIR/libxcb-cursor.so.0${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  echo "未检测到图形会话。请在服务器图形桌面的终端运行此脚本。" >&2
  exit 2
fi

echo "Python3: $(command -v python3)"
python3 -c 'import PySide6; print("PySide6:", PySide6.__file__)'
echo "Layout: $TASK_PROJECT_ROOT/data/medication_sorting/scenes/drag_classfication.json"
echo "运行时先等待资产同步完成，再由 prepare_scene.py 加载 Layout。"
cd "$TASK_PROJECT_ROOT/software/OrcaPlayground"
exec "$TASK_PROJECT_ROOT/.venv/bin/orcalab" \
  --verbose --lang zh_CN "$@"
