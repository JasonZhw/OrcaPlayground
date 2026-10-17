#!/usr/bin/env bash
# ORCA 课程的项目级环境变量；不修改系统全局 Locale 或用户 Shell 配置。
export SY_ROOT="${SY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PATH="$SY_ROOT/.venv/bin:$PATH"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANGUAGE="zh_CN:zh"

# 当前桌面 Shell 全局注入了 ROS Jazzy 的 Python 路径；ORCA 使用独立环境时隔离它，
# 避免系统包进入 .venv 的依赖解析。不会卸载或修改 ROS。
unset PYTHONPATH
unset PYTHONHOME
export PIP_CACHE_DIR="$SY_ROOT/runtime/cache/pip"
export XDG_CACHE_HOME="$SY_ROOT/runtime/cache/xdg"
export XDG_CONFIG_HOME="$SY_ROOT/runtime/config"
export XDG_DATA_HOME="$SY_ROOT/runtime/data"

# 移除当前登录 Shell 注入的 ROS 原生库路径，避免 Qt/MuJoCo 加载不匹配库；
# 其他自定义路径（例如 CUDA）保持不变。
_ORCA_CLEAN_LD_LIBRARY_PATH=""
_orca_ld_remaining="${LD_LIBRARY_PATH:-}"
while [[ -n "$_orca_ld_remaining" ]]; do
  case "$_orca_ld_remaining" in
    *:*)
      _orca_path="${_orca_ld_remaining%%:*}"
      _orca_ld_remaining="${_orca_ld_remaining#*:}"
      ;;
    *)
      _orca_path="$_orca_ld_remaining"
      _orca_ld_remaining=""
      ;;
  esac
  [[ -z "$_orca_path" || "$_orca_path" == /opt/ros/* ]] && continue
  _ORCA_CLEAN_LD_LIBRARY_PATH="${_ORCA_CLEAN_LD_LIBRARY_PATH:+$_ORCA_CLEAN_LD_LIBRARY_PATH:}$_orca_path"
done

# Qt xcb 插件所需库以普通用户方式解压在项目内，不安装到 Ubuntu 系统。
export ORCA_SYSTEM_LIB_DIR="$SY_ROOT/runtime/system-libs/root/usr/lib/x86_64-linux-gnu"
if [[ -d "$ORCA_SYSTEM_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$ORCA_SYSTEM_LIB_DIR${_ORCA_CLEAN_LD_LIBRARY_PATH:+:$_ORCA_CLEAN_LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="$_ORCA_CLEAN_LD_LIBRARY_PATH"
fi
unset _ORCA_CLEAN_LD_LIBRARY_PATH _orca_ld_remaining _orca_path

mkdir -p \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" \
  "$SY_ROOT/runtime/logs" \
  "$SY_ROOT/runtime/reports"
