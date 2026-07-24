#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# LearnGUI evaluation SLURM script
#
# Dual-mode:
#   - Login node (no SLURM_JOB_ID): sbatch-submit this script as a job
#   - Compute node (SLURM_JOB_ID set): activate conda and run evaluation
#
# Supports V0–V3 variants, --resume, --measure-e2e-latency, --variant.
# --------------------------------------------------------------------------- #

if [[ -n "${LEARNGUI_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${LEARNGUI_REPO_ROOT}"
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

DEFAULT_CONFIG="${REPO_ROOT}/configs/learngui/default.json"
DEFAULT_JOB_NAME="learngui-eval"
DEFAULT_PARTITION=""
DEFAULT_ACCOUNT="faculty"
DEFAULT_TIME="7-00:00:00"
DEFAULT_GPUS="4"
DEFAULT_CPUS="16"
DEFAULT_MEM="128G"
DEFAULT_CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:-${REPO_ROOT}/.conda/envs/skillreuse}"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs/learngui_eval"
DEFAULT_VARIANT="V0"
DEFAULT_SPLIT=""
DEFAULT_MEASURE_E2E="false"
DEFAULT_RESUME="false"

cst_date() { date -d "+12 hours" "$@"; }

usage() {
  cat <<EOF
用法:
  ${0} [options]

功能:
  - 在登录节点执行时自动调用 sbatch 提交 LearnGUI 评测作业
  - 在计算节点执行时激活 conda 环境并运行评测
  - 将 Slurm stdout/stderr 和转发终端日志写入 training_journal/run_* 或 slurm_logs/

常用示例:
  ${0} --variant V0 --gpus 4
  ${0} --variant V1 --skill-library-path outputs/learngui_discovery/skill_library.pkl
  ${0} --partition h20-gpu --account faculty --gpus 4
  ${0} --output-dir ${REPO_ROOT}/outputs/learngui_eval --comment smoke

选项:
  --job-name NAME          Slurm job 名称，默认: ${DEFAULT_JOB_NAME}
  --config PATH            JSON 配置，默认: ${DEFAULT_CONFIG}
  --variant V0|V1|V2|V3   评测变体，默认: ${DEFAULT_VARIANT}
  --skill-library-path P   LoRA/skill library pickle 路径（V1/V2/V3 必须）
  --gpus N                 GPU 数量，默认: ${DEFAULT_GPUS}
  --partition NAME         Slurm partition（4 卡默认先试 l20-gpu，失败再试 h20-gpu）
  --account NAME           Slurm account，默认: ${DEFAULT_ACCOUNT}
  --time HH:MM:SS          作业时间限制，默认: ${DEFAULT_TIME}
  --cpus N                 每任务 CPU 数量，默认: ${DEFAULT_CPUS}
  --mem SIZE               内存，默认: ${DEFAULT_MEM}
  --conda-prefix PATH      conda 环境路径，默认: ${DEFAULT_CONDA_PREFIX}
  --output-dir PATH        评测输出目录，默认: ${DEFAULT_OUTPUT_DIR}
  --split NAME             LearnGUI split 覆盖（如 validation / test）
  --measure-e2e-latency    评测时测量逐步端到端延迟
  --resume                 从 output-dir 断点继续评测
  --comment TEXT           追加到作业名的注释标签
  --dry-run                仅打印 sbatch 命令，不提交
  -h, --help               显示帮助

日志布局:
  - eval tentative:   ${REPO_ROOT}/training_journal/.tentative/run_<YYYYMMDD_HHMMSS>/
  - eval promoted:    ${REPO_ROOT}/training_journal/<YYYY_MM_DD>/run_<YYYYMMDD_HHMMSS>/
  - Slurm 原生日志:   <run-dir>/slurm.out | <run-dir>/slurm.err
  - 转发终端日志:     <run-dir>/terminal.log
EOF
}

submit_job() {
  local run_stamp
  run_stamp="$(cst_date '+%Y%m%d_%H%M%S')"
  local run_date
  run_date="$(cst_date '+%Y_%m_%d')"
  local journal_run_dir="${REPO_ROOT}/training_journal/.tentative/run_${run_stamp}"
  mkdir -p "${journal_run_dir}"

  local output_pattern="${journal_run_dir}/slurm.out"
  local error_pattern="${journal_run_dir}/slurm.err"

  local export_vars=(
    "ALL"
    "LEARNGUI_REPO_ROOT=${REPO_ROOT}"
    "LEARNGUI_CONFIG=${CONFIG_PATH}"
    "LEARNGUI_CONDA_PREFIX=${CONDA_PREFIX_PATH}"
    "LEARNGUI_OUTPUT_DIR=${OUTPUT_DIR}"
    "LEARNGUI_VARIANT=${VARIANT}"
    "LEARNGUI_SKILL_LIBRARY_PATH=${SKILL_LIBRARY_PATH}"
    "LEARNGUI_SPLIT=${SPLIT}"
    "LEARNGUI_MEASURE_E2E=${MEASURE_E2E}"
    "LEARNGUI_RESUME=${RESUME}"
    "LEARNGUI_RUN_DATE=${run_date}"
    "LEARNGUI_RUN_STAMP=${run_stamp}"
    "LEARNGUI_JOURNAL_RUN_DIR=${journal_run_dir}"
    "LEARNGUI_JOB_SUBMITTED_AT=${run_stamp}"
    "LEARNGUI_JOB_COMMENT=${COMMENT}"
    "LEARNGUI_GPUS=${GPUS}"
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

  local submit_output
  if ! submit_output="$(submit_sbatch_with_gpu_fallback \
    sbatch_cmd "false" "${GPUS}" "${PARTITION}" "${PARTITION_EXPLICIT}" "${DRY_RUN}")"; then
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
  echo "Promoted run dir: ${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
  echo "查看队列: squeue -j ${job_id}"
  echo "看 Slurm 输出: tail -f ${output_pattern}"
  echo "看转发日志: tail -f ${journal_run_dir}/terminal.log"
}

run_job() {
  local config_path="${LEARNGUI_CONFIG:-${DEFAULT_CONFIG}}"
  local conda_prefix="${LEARNGUI_CONDA_PREFIX:-${DEFAULT_CONDA_PREFIX}}"
  local output_dir="${LEARNGUI_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
  local variant="${LEARNGUI_VARIANT:-${DEFAULT_VARIANT}}"
  local skill_library_path="${LEARNGUI_SKILL_LIBRARY_PATH:-}"
  local split="${LEARNGUI_SPLIT:-}"
  local measure_e2e="${LEARNGUI_MEASURE_E2E:-false}"
  local resume="${LEARNGUI_RESUME:-false}"
  local run_date="${LEARNGUI_RUN_DATE:-$(cst_date '+%Y_%m_%d')}"
  local run_stamp="${LEARNGUI_RUN_STAMP:-$(cst_date '+%Y%m%d_%H%M%S')}"
  local journal_run_dir="${LEARNGUI_JOURNAL_RUN_DIR:-}"
  local comment="${LEARNGUI_JOB_COMMENT:-}"
  local requested_gpus="${LEARNGUI_GPUS:-${GPUS}}"

  if [[ -z "${journal_run_dir}" ]]; then
    journal_run_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
  fi
  mkdir -p "${journal_run_dir}"

  local forwarded_log="${journal_run_dir}/terminal.log"
  exec > >(tee -a "${forwarded_log}") 2>&1

  cleanup_on_failure() {
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

  export LEARNGUI_JOURNAL_RUN_DIR="${journal_run_dir}"
  export LEARNGUI_RUN_DATE="${run_date}"
  export LEARNGUI_RUN_STAMP="${run_stamp}"
  export PYTHONNOUSERSITE=1

  echo "[$(cst_date '+%F %T')] Starting LearnGUI eval job ${SLURM_JOB_ID:-local}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Config: ${config_path}"
  echo "Variant: ${variant}"
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
    mode='learngui_eval',
    comment='${comment}',
)
"

  local client_exit=0
  local eval_args=("${python_bin}" "${REPO_ROOT}/scripts/core/run_evaluation.py"
    --benchmark LearnGUI
    --config "${config_path}"
    --variant "${variant}"
    --output-dir "${output_dir}")

  if [[ -n "${skill_library_path}" ]]; then
    eval_args+=(--skill-library-path "${skill_library_path}")
  fi
  if [[ -n "${split}" ]]; then
    eval_args+=(--split "${split}")
  fi
  if [[ "${measure_e2e}" == "true" ]]; then
    eval_args+=(--measure-e2e-latency)
  fi
  if [[ "${resume}" == "true" ]]; then
    eval_args+=(--resume)
  fi

  echo "[$(cst_date '+%F %T')] Launching: ${eval_args[*]}"
  "${eval_args[@]}" || client_exit=$?

  if [[ ${client_exit} -eq 0 ]]; then
    local final_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
    if [[ "${journal_run_dir}" == *"/.tentative/"* ]]; then
      mkdir -p "$(dirname "${final_dir}")"
      mv "${journal_run_dir}" "${final_dir}"
      journal_run_dir="${final_dir}"
      export LEARNGUI_JOURNAL_RUN_DIR="${journal_run_dir}"
      echo "[$(cst_date '+%F %T')] Journal promoted to: ${journal_run_dir}"
      trap - EXIT
    fi
  fi

  local final_status="completed"
  if [[ ${client_exit} -ne 0 ]]; then
    final_status="failed"
  fi

  "${python_bin}" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from skillreuse.journal import update_status
from pathlib import Path
update_status(Path('${journal_run_dir}'), '${final_status}')
"

  echo "[$(cst_date '+%F %T')] Done. Status: ${final_status} (exit code: ${client_exit})"
  exit ${client_exit}
}

JOB_NAME="${DEFAULT_JOB_NAME}"
CONFIG_PATH="${LEARNGUI_CONFIG:-${DEFAULT_CONFIG}}"
GPUS="${DEFAULT_GPUS}"
PARTITION="${DEFAULT_PARTITION}"
ACCOUNT="${DEFAULT_ACCOUNT}"
TIME_LIMIT="${DEFAULT_TIME}"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
CONDA_PREFIX_PATH="${DEFAULT_CONDA_PREFIX}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
VARIANT="${DEFAULT_VARIANT}"
SKILL_LIBRARY_PATH=""
SPLIT="${DEFAULT_SPLIT}"
MEASURE_E2E="${DEFAULT_MEASURE_E2E}"
RESUME="${DEFAULT_RESUME}"
COMMENT=""
DRY_RUN="false"
PARTITION_EXPLICIT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)             JOB_NAME="$2"; shift 2 ;;
    --config)               CONFIG_PATH="$2"; shift 2 ;;
    --variant)              VARIANT="$2"; shift 2 ;;
    --skill-library-path)   SKILL_LIBRARY_PATH="$2"; shift 2 ;;
    --gpus)                 GPUS="$2"; shift 2 ;;
    --partition)            PARTITION="$2"; PARTITION_EXPLICIT="true"; shift 2 ;;
    --account)              ACCOUNT="$2"; shift 2 ;;
    --time)                 TIME_LIMIT="$2"; shift 2 ;;
    --cpus)                 CPUS="$2"; shift 2 ;;
    --mem)                  MEM="$2"; shift 2 ;;
    --conda-prefix)         CONDA_PREFIX_PATH="$2"; shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
    --split)                SPLIT="$2"; shift 2 ;;
    --measure-e2e-latency)  MEASURE_E2E="true"; shift ;;
    --resume)               RESUME="true"; shift ;;
    --comment)              COMMENT="$2"; shift 2 ;;
    --dry-run)              DRY_RUN="true"; shift ;;
    -h|--help)              usage; exit 0 ;;
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

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_job
else
  run_job
fi
