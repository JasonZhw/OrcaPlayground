#!/usr/bin/env bash
# 只为当前 Python 进程启用 pi05-deps；不 source activate，不修改仿真环境。
set -euo pipefail

pi05_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
pi05_python="${pi05_project_root}/.venv/bin/python"
pi05_flags=()

while (( $# )); do
    case "$1" in
        --dry-run|--cpu)
            pi05_flags+=("$1")
            shift
            ;;
        -h|--help)
            cat <<'USAGE'
用法：pi05_python.sh [--dry-run] [--cpu] <Python 参数>
  --dry-run：只打印解释器、argv 和隔离环境，不安装、不使用 GPU。
  --cpu：设置 CUDA_VISIBLE_DEVICES='' 和 JAX_PLATFORMS=cpu。
示例：
  pi05_python.sh --dry-run -m tools.medication_sorting.pi05_adapter train -- --exp-name=g1_trial
  pi05_python.sh --cpu -m tools.medication_sorting.setup_pi05 verify
首次使用前，显式运行安装入口 .venv/bin/python tools/medication_sorting/setup_pi05.py install --execute。
安装中断后：同一入口 install --execute --retry 会核对受保护快照并修复 partial target。
USAGE
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *) break ;;
    esac
done

exec "${pi05_python}" "${pi05_project_root}/tools/medication_sorting/setup_pi05.py" run "${pi05_flags[@]}" -- "$@"
