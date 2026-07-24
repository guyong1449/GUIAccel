#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# LearnGUI VisionZip evaluation SLURM script
#
# Dual-mode:
#   - Login node (no SLURM_JOB_ID): sbatch-submit this script as a job
#   - Compute node (SLURM_JOB_ID set): run VisionZip eval for LearnGUI benchmark
#
# Partition: l20-gpu (no h20 fallback).
# --------------------------------------------------------------------------- #

if [[ -n "${VISIONZIP_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${VISIONZIP_REPO_ROOT}"
  SCRIPT_DIR="${REPO_ROOT}/run_scripts"
else
_SKILLREUSE_SEARCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_SKILLREUSE_SEARCH}" != "/" ]]; do
  if [[ -f "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh" ]]; then
    # shellcheck disable=SC1090
    source "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh"
    skillreuse_init_from "${BASH_SOURCE[0]}"
    break
  fi
  _SKILLREUSE_SEARCH="$(dirname "${_SKILLREUSE_SEARCH}")"
done
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib/load_env.sh"
skillreuse_load_env "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/run_scripts/_lib/slurm_helpers.sh"

DEFAULT_CONFIG="${REPO_ROOT}/configs/learngui/visionzip/default.json"
DEFAULT_SMOKE_CONFIG="${REPO_ROOT}/configs/learngui/visionzip/smoke.json"
DEFAULT_JOB_NAME="learngui-visionzip-eval"
DEFAULT_PARTITION="l20-gpu"
DEFAULT_ACCOUNT="faculty"
DEFAULT_TIME="7-00:00:00"
DEFAULT_GPUS="4"
DEFAULT_CPUS="16"
DEFAULT_MEM="128G"
DEFAULT_CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:-${REPO_ROOT}/.conda/envs/skillreuse}"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs/visionzip_learngui_v0_eval"
DEFAULT_MODE="eval"
DEFAULT_RESUME="false"

# Server is US time; +12h gives Beijing date for journal folders.
cst_date() { date -d "+12 hours" "$@"; }

usage() {
  cat <<EOF
用法:
  ${0} [options]

功能:
  - 在登录节点执行时自动调用 sbatch 提交 LearnGUI VisionZip 评测作业
  - 在计算节点执行时激活 conda 环境并运行评测（local transformers，无 vLLM）
  - 将 Slurm stdout/stderr 写入 training_journal/run_* 或 slurm_logs/visionzip/smoke/

常用示例:
  ${0} --smoke
  ${0} --gpus 4
  ${0} --output-dir ${REPO_ROOT}/outputs/visionzip_learngui_v0_eval --comment full-test

选项:
  --job-name NAME          Slurm job 名称，默认: ${DEFAULT_JOB_NAME}
  --config PATH            JSON 配置，默认: ${DEFAULT_CONFIG}
  --smoke                  单卡冒烟测试（使用 ${DEFAULT_SMOKE_CONFIG}）
  --gpus N                 GPU 数量，默认: ${DEFAULT_GPUS}（smoke 为 1）
  --partition NAME         Slurm partition，默认: ${DEFAULT_PARTITION}
  --account NAME           Slurm account，默认: ${DEFAULT_ACCOUNT}
  --time HH:MM:SS          作业时间限制，默认: ${DEFAULT_TIME}
  --cpus N                 每任务 CPU 数量，默认: ${DEFAULT_CPUS}
  --mem SIZE               内存，默认: ${DEFAULT_MEM}
  --conda-prefix PATH      conda 环境路径，默认: ${DEFAULT_CONDA_PREFIX}
  --output-dir PATH        评测输出目录；未显式指定时默认 ${DEFAULT_OUTPUT_DIR}_<timestamp>
  --resume                 从 output-dir 断点继续评测（须显式指定 --output-dir）
  --comment TEXT           追加到作业名的注释标签
  --dry-run                仅打印 sbatch 命令，不提交
  -h, --help               显示帮助

日志布局:
  - smoke:            ${REPO_ROOT}/slurm_logs/visionzip/smoke/run_<YYYYMMDD_HHMMSS>/
  - eval tentative:   ${REPO_ROOT}/training_journal/.tentative/run_<YYYYMMDD_HHMMSS>/
  - eval promoted:    ${REPO_ROOT}/training_journal/<YYYY_MM_DD>/run_<YYYYMMDD_HHMMSS>/
  - Slurm 原生日志:   <run-dir>/slurm.out | <run-dir>/slurm.err
EOF
}

sanitize_name() {
  local value="$1"
  value="${value// /_}"
  value="${value//\//_}"
  value="${value//:/-}"
  value="${value//,/__}"
  printf '%s' "${value}"
}

submit_job() {
  local run_stamp
  run_stamp="$(cst_date '+%Y%m%d_%H%M%S')"
  local run_date
  run_date="$(cst_date '+%Y_%m_%d')"
  if [[ "${OUTPUT_DIR_EXPLICIT}" != "true" ]]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_${run_stamp}"
  fi
  echo "Output dir: ${OUTPUT_DIR}"
  local journal_run_dir
  if [[ "${MODE}" == "smoke" ]]; then
    journal_run_dir="${REPO_ROOT}/slurm_logs/visionzip/smoke/run_${run_stamp}"
  else
    journal_run_dir="${REPO_ROOT}/training_journal/.tentative/run_${run_stamp}"
  fi
  mkdir -p "${journal_run_dir}"

  local output_pattern="${journal_run_dir}/slurm.out"
  local error_pattern="${journal_run_dir}/slurm.err"

  local export_vars=(
    "ALL"
    "VISIONZIP_REPO_ROOT=${REPO_ROOT}"
    "VISIONZIP_CONFIG=${CONFIG_PATH}"
    "VISIONZIP_MODE=${MODE}"
    "VISIONZIP_CONDA_PREFIX=${CONDA_PREFIX_PATH}"
    "VISIONZIP_OUTPUT_DIR=${OUTPUT_DIR}"
    "VISIONZIP_RESUME=${RESUME}"
    "VISIONZIP_RUN_DATE=${run_date}"
    "VISIONZIP_RUN_STAMP=${run_stamp}"
    "VISIONZIP_JOURNAL_RUN_DIR=${journal_run_dir}"
    "VISIONZIP_JOB_SUBMITTED_AT=${run_stamp}"
    "VISIONZIP_JOB_COMMENT=${COMMENT}"
    "VISIONZIP_GPUS=${GPUS}"
  )

  local sbatch_cmd=(sbatch)
  sbatch_cmd+=(--job-name "${JOB_NAME}")
  sbatch_cmd+=(--nodes 1 --ntasks 1)
  sbatch_cmd+=(--cpus-per-task "${CPUS}")
  sbatch_cmd+=(--mem "${MEM}")
  sbatch_cmd+=(--time "${TIME_LIMIT}")
  sbatch_cmd+=(--chdir "${REPO_ROOT}")
  sbatch_cmd+=(--output "${output_pattern}")
  sbatch_cmd+=(--error "${error_pattern}")
  sbatch_cmd+=(--export "$(IFS=,; echo "${export_vars[*]}")")
  if [[ -n "${ACCOUNT}" ]]; then
    sbatch_cmd+=(--account "${ACCOUNT}")
  fi
  sbatch_cmd+=("${0}")

  local is_smoke="false"
  if [[ "${MODE}" == "smoke" ]]; then
    is_smoke="true"
  fi

  local submit_output
  if ! submit_output="$(submit_sbatch_with_gpu_fallback \
    sbatch_cmd "${is_smoke}" "${GPUS}" "${PARTITION}" "${PARTITION_EXPLICIT}" "${DRY_RUN}")"; then
    exit 1
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Journal dir: ${journal_run_dir}"
    return 0
  fi

  local job_id
  job_id="$(awk '{print $NF}' <<<"${submit_output}")"
  if [[ -z "${job_id}" ]]; then
    echo "无法解析 sbatch 输出: ${submit_output}" >&2
    exit 1
  fi

  echo "${submit_output}"
  echo "Journal dir: ${journal_run_dir}"
  if [[ "${MODE}" != "smoke" ]]; then
    echo "Promoted run dir: ${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
  fi
  echo "查看队列: squeue -j ${job_id}"
  echo "看 Slurm 输出: tail -f ${output_pattern}"
}

run_job() {
  local config_path="${VISIONZIP_CONFIG:-${DEFAULT_CONFIG}}"
  local mode="${VISIONZIP_MODE:-${DEFAULT_MODE}}"
  local conda_prefix="${VISIONZIP_CONDA_PREFIX:-${DEFAULT_CONDA_PREFIX}}"
  local output_dir="${VISIONZIP_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
  local resume="${VISIONZIP_RESUME:-false}"
  local run_date="${VISIONZIP_RUN_DATE:-$(cst_date '+%Y_%m_%d')}"
  local run_stamp="${VISIONZIP_RUN_STAMP:-$(cst_date '+%Y%m%d_%H%M%S')}"
  local journal_run_dir="${VISIONZIP_JOURNAL_RUN_DIR:-}"
  local comment="${VISIONZIP_JOB_COMMENT:-}"
  local requested_gpus="${VISIONZIP_GPUS:-${GPUS}}"

  if [[ -z "${journal_run_dir}" ]]; then
    if [[ "${mode}" == "smoke" ]]; then
      journal_run_dir="${REPO_ROOT}/slurm_logs/visionzip/smoke/run_${run_stamp}"
    else
      journal_run_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
    fi
  fi
  mkdir -p "${journal_run_dir}"
  ln -sf slurm.out "${journal_run_dir}/terminal.log"

  cleanup_on_failure() {
    if [[ "${mode}" == "smoke" ]]; then
      return 0
    fi
    if [[ -d "${journal_run_dir}" && "${journal_run_dir}" == *"/.tentative/"* ]]; then
      local aborted_dir="${REPO_ROOT}/training_journal/.aborted/$(basename "${journal_run_dir}")"
      mkdir -p "${aborted_dir}"
      for name in slurm.out slurm.err terminal.log config_snapshot.json pointers.json run_summary.md; do
        if [[ -f "${journal_run_dir}/${name}" ]]; then
          cp -f "${journal_run_dir}/${name}" "${aborted_dir}/${name}"
        fi
      done
      rm -rf "${journal_run_dir}"
    fi
  }
  trap cleanup_on_failure EXIT

  export VISIONZIP_JOURNAL_RUN_DIR="${journal_run_dir}"
  export VISIONZIP_RUN_DATE="${run_date}"
  export VISIONZIP_RUN_STAMP="${run_stamp}"
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1

  echo "[$(cst_date '+%F %T')] Starting LearnGUI VisionZip job ${SLURM_JOB_ID:-local} (mode=${mode})"
  echo "Repo root: ${REPO_ROOT}"
  echo "Config: ${config_path}"
  echo "Mode: ${mode}"
  echo "Requested GPUs: ${requested_gpus}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "Conda prefix: ${conda_prefix}"
  echo "Output dir: ${output_dir}"
  echo "Journal run dir: ${journal_run_dir}"

  cd "${REPO_ROOT}"

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_prefix}"
  else
    export PATH="${conda_prefix}/bin:${PATH}"
  fi

  local python_bin="${conda_prefix}/bin/python"

  "${python_bin}" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from skillreuse.journal import prepare_journal
from pathlib import Path
prepare_journal(
    Path('${journal_run_dir}'),
    config_path='${config_path}',
    output_dir='${output_dir}',
    mode='visionzip_${mode}',
    comment='${comment}',
)
"

  local client_exit=0
  local eval_args=("${python_bin}" "${REPO_ROOT}/scripts/androidcontrol/eval/run_visionzip_eval.py"
    --benchmark LearnGUI
    --config "${config_path}")

  if [[ "${mode}" == "smoke" ]]; then
    eval_args+=(--dry-run --gpu 0)
  else
    eval_args+=(--output-dir "${output_dir}")
    if [[ "${resume}" == "true" ]]; then
      eval_args+=(--resume)
    fi
  fi

  echo "[$(cst_date '+%F %T')] Launching: ${eval_args[*]}"
  "${eval_args[@]}" || client_exit=$?

  if [[ "${mode}" != "smoke" && ${client_exit} -eq 0 ]]; then
    local final_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
    if [[ "${journal_run_dir}" == *"/.tentative/"* ]]; then
      mkdir -p "$(dirname "${final_dir}")"
      mv "${journal_run_dir}" "${final_dir}"
      journal_run_dir="${final_dir}"
      export VISIONZIP_JOURNAL_RUN_DIR="${journal_run_dir}"
      echo "[$(cst_date '+%F %T')] Journal promoted to: ${journal_run_dir}"
      trap - EXIT
    fi
  fi

  local final_status="completed"
  if [[ ${client_exit} -ne 0 ]]; then
    final_status="failed"
  fi

  if [[ "${mode}" != "smoke" ]]; then
    "${python_bin}" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from skillreuse.journal import update_status
from pathlib import Path
update_status(Path('${journal_run_dir}'), '${final_status}')
"
  fi

  echo "[$(cst_date '+%F %T')] Done. Status: ${final_status} (exit code: ${client_exit})"
  exit ${client_exit}
}

JOB_NAME="${DEFAULT_JOB_NAME}"
CONFIG_PATH="${VISIONZIP_CONFIG:-${DEFAULT_CONFIG}}"
MODE="${DEFAULT_MODE}"
GPUS="${DEFAULT_GPUS}"
PARTITION="${DEFAULT_PARTITION}"
ACCOUNT="${DEFAULT_ACCOUNT}"
TIME_LIMIT="${DEFAULT_TIME}"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
CONDA_PREFIX_PATH="${DEFAULT_CONDA_PREFIX}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
RESUME="${DEFAULT_RESUME}"
COMMENT=""
DRY_RUN="false"
PARTITION_EXPLICIT="true"
OUTPUT_DIR_EXPLICIT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)    JOB_NAME="$2"; shift 2 ;;
    --config)      CONFIG_PATH="$2"; shift 2 ;;
    --smoke)       MODE="smoke"; GPUS="1"; CONFIG_PATH="${DEFAULT_SMOKE_CONFIG}"; shift ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --partition)   PARTITION="$2"; PARTITION_EXPLICIT="true"; shift 2 ;;
    --account)     ACCOUNT="$2"; shift 2 ;;
    --time)        TIME_LIMIT="$2"; shift 2 ;;
    --cpus)        CPUS="$2"; shift 2 ;;
    --mem)         MEM="$2"; shift 2 ;;
    --conda-prefix) CONDA_PREFIX_PATH="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; OUTPUT_DIR_EXPLICIT="true"; shift 2 ;;
    --resume)      RESUME="true"; shift ;;
    --comment)     COMMENT="$2"; shift 2 ;;
    --dry-run)     DRY_RUN="true"; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${SLURM_JOB_ID:-}" && ! -f "${CONFIG_PATH}" ]]; then
  echo "配置文件不存在: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${RESUME}" == "true" && "${OUTPUT_DIR_EXPLICIT}" != "true" ]]; then
  echo "--resume requires explicit --output-dir (auto-stamped default dirs cannot be resumed)" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_job
else
  run_job
fi
