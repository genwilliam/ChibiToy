#!/usr/bin/env bash
# 启动 ComfyUI
#
# 用法:
#   ./scripts/start_comfyui.sh              # 前台启动
#   ./scripts/start_comfyui.sh --background # 后台启动（日志写入 logs/comfyui.log）
#
# 配置说明:
#   --extra-model-paths-config  让 ComfyUI 从项目 models/ 读取模型
#   --output-directory          生成结果统一落到项目根目录 results/
#   --input-directory           输入照片统一放项目 inputs/
#   PYTORCH_ENABLE_MPS_FALLBACK=1
#       允许 MPS 未实现的算子回退到 CPU，避免个别算子直接报错中断。
#       Apple Silicon 上这是保证工作流稳定跑通的重要设置。

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
COMFY_DIR="${PROJECT_ROOT}/ComfyUI"

PORT="${CHIBITOY_PORT:-8188}"
HOST="${CHIBITOY_HOST:-127.0.0.1}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results" "${PROJECT_ROOT}/inputs"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[错误] 找不到虚拟环境: ${PYTHON}" >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
# 关闭 HuggingFace 遥测，保持完全本地运行
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

ARGS=(
  main.py
  --listen "${HOST}"
  --port "${PORT}"
  --extra-model-paths-config "${PROJECT_ROOT}/config/extra_model_paths.yaml"
  --output-directory "${PROJECT_ROOT}/results"
  --input-directory "${PROJECT_ROOT}/inputs"
  --disable-auto-launch
)

cd "${COMFY_DIR}"

if [[ "${1:-}" == "--background" ]]; then
  echo "[启动] 后台运行，日志: ${PROJECT_ROOT}/logs/comfyui.log"
  nohup "${PYTHON}" "${ARGS[@]}" > "${PROJECT_ROOT}/logs/comfyui.log" 2>&1 &
  echo $! > "${PROJECT_ROOT}/logs/comfyui.pid"
  echo "[PID] $(cat "${PROJECT_ROOT}/logs/comfyui.pid")"
  echo "[地址] http://${HOST}:${PORT}"
else
  # 前台运行，同时把输出写入日志文件
  LOG_FILE="${PROJECT_ROOT}/logs/comfyui.log"
  echo "[启动] 前台运行 (Ctrl+C 停止)，日志: ${LOG_FILE}"
  echo "[地址] http://${HOST}:${PORT}"
  "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi
