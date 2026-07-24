#!/usr/bin/env bash
set -Eeuo pipefail

proxy_url="${MSD_DOWNLOAD_PROXY:-http://127.0.0.1:17890}"
hf_home="${MSD_HF_HOME:-/dkucc/home/rw335/.cache/huggingface}"
conda_env="${MSD_CONDA_ENV:-msd-androidcontrol}"

curl \
  -x "${proxy_url}" \
  --fail \
  --head \
  --connect-timeout 10 \
  --max-time 30 \
  https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

export HTTP_PROXY="${proxy_url}"
export HTTPS_PROXY="${proxy_url}"
export http_proxy="${proxy_url}"
export https_proxy="${proxy_url}"
export HF_HOME="${hf_home}"

mkdir -p "${HF_HOME}"

conda run -n "${conda_env}" hf download \
  Qwen/Qwen2-VL-7B-Instruct \
  --cache-dir "${HF_HOME}/hub"

conda run -n "${conda_env}" hf download \
  lucylyn/MSD-Qwen2VL-7B-Instruct \
  --cache-dir "${HF_HOME}/hub"

echo "Models cached under ${HF_HOME}/hub"
