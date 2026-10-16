#!/usr/bin/env bash
# 模型下载脚本 —— 支持断点续传与 SHA256 校验
#
# 用法:
#   ./scripts/download_models.sh              # 下载全部
#   ./scripts/download_models.sh sd_xl_base_1.0   # 只下载指定模型
#   VERIFY=0 ./scripts/download_models.sh     # 跳过校验（快速检查）
#
# 说明: 大文件一旦中断，重新执行即可从断点继续（curl -C -）。

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/checkpoints"
mkdir -p "${MODEL_DIR}"

# 模型清单: 名称|下载地址|字节数|SHA256|许可
MODELS=(
  "sd_xl_base_1.0.safetensors|https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors|6938078334|31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b|CreativeML Open RAIL++-M"
)

# 计算文件 SHA256（macOS 用 shasum，Linux 用 sha256sum）
file_sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

file_size() {
  stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null
}

human() {
  awk -v b="$1" 'BEGIN{ printf "%.2f GB", b/1024/1024/1024 }'
}

download_one() {
  local name="$1" url="$2" want_size="$3" want_sha="$4" license="$5"
  local dest="${MODEL_DIR}/${name}"

  echo "=============================================================="
  echo "模型      : ${name}"
  echo "目标路径  : ${dest}"
  echo "期望大小  : ${want_size} ($(human "${want_size}"))"
  echo "期望SHA256: ${want_sha}"
  echo "使用许可  : ${license}"
  echo "=============================================================="

  # 已存在且完整 -> 跳过
  if [[ -f "${dest}" ]]; then
    local cur_size; cur_size="$(file_size "${dest}")"
    if [[ "${cur_size}" == "${want_size}" ]]; then
      echo "[跳过] 文件已存在且大小一致，直接校验。"
    else
      echo "[续传] 已存在 ${cur_size} 字节，继续下载。"
    fi
  fi

  # 断点续传下载（--retry 应对网络抖动）
  if ! curl -L -C - \
        --retry 5 --retry-delay 5 --retry-all-errors \
        --fail --progress-bar \
        -o "${dest}" "${url}"; then
    echo "[失败] 下载中断: ${name}" >&2
    return 1
  fi

  local got_size; got_size="$(file_size "${dest}")"
  if [[ "${got_size}" != "${want_size}" ]]; then
    echo "[失败] 大小不符: 期望 ${want_size}，实际 ${got_size}" >&2
    return 1
  fi
  echo "[通过] 文件大小校验一致。"

  if [[ "${VERIFY:-1}" == "1" ]]; then
    echo "[校验] 正在计算 SHA256（6.9GB 约需 10-30 秒）..."
    local got_sha; got_sha="$(file_sha256 "${dest}")"
    if [[ "${got_sha}" != "${want_sha}" ]]; then
      echo "[失败] SHA256 不匹配!" >&2
      echo "  期望: ${want_sha}" >&2
      echo "  实际: ${got_sha}" >&2
      return 1
    fi
    echo "[通过] SHA256 校验一致。"
  else
    echo "[跳过] 按 VERIFY=0 跳过 SHA256 校验。"
  fi

  return 0
}

main() {
  local filter="${1:-}"
  local failed=0
  local matched=0

  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name url size sha license <<< "${entry}"
    # 过滤器按子串匹配，便于用 "sd_xl_base" 这样的简写指定模型
    if [[ -n "${filter}" && "${name}" != *"${filter}"* ]]; then
      continue
    fi
    matched=$((matched + 1))
    if ! download_one "${name}" "${url}" "${size}" "${sha}" "${license}"; then
      failed=$((failed + 1))
    fi
    echo
  done

  # 过滤器没匹配到任何模型时报错
  if [[ ${matched} -eq 0 ]]; then
    echo "[失败] 没有模型匹配过滤器: '${filter}'" >&2
    echo "可用模型:" >&2
    for entry in "${MODELS[@]}"; do
      IFS='|' read -r name _ _ _ _ <<< "${entry}"
      echo "  - ${name}" >&2
    done
    return 2
  fi

  if [[ ${failed} -gt 0 ]]; then
    echo "==> 完成，但有 ${failed} 个模型失败。"
    return 1
  fi
  echo "==> 全部模型下载并校验成功（共 ${matched} 个）。"
}

main "$@"
