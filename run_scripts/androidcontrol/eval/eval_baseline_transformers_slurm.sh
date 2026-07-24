#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# local_transformers baseline evaluation SLURM script (mirrors visionzip pattern)
#
# Dual-mode:
#   - Login node (no SLURM_JOB_ID): sbatch-submit this script as a job
#   - Compute node (SLURM_JOB_ID set): run V0 eval with local HuggingFace transformers
#
# Fair latency baseline for VisionZip/DivPrune speedup comparisons (no pruning).
# --------------------------------------------------------------------------- #

if [[ -n "${BASELINE_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${BASELINE_REPO_ROOT}"
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

DEFAULT_CONFIG="${REPO_ROOT}/configs/androidcontrol/baseline/default.json"
DEFAULT_JOB_NAME="baseline-transformers-eval"
DEFAULT_PARTITION="l20-gpu"
DEFAULT_ACCOUNT="faculty"
DEFAULT_TIME="7-00:00:00"
DEFAULT_GPUS="4"
DEFAULT_CPUS="16"
DEFAULT_MEM="128G"
DEFAULT_CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:-${REPO_ROOT}/.conda/envs/skillreuse}"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs/baseline_transformers"
DEFAULT_SPLIT=""
DEFAULT_MEASURE_E2E="false"
DEFAULT_RESUME="false"

# Server is US time; +12h gives Beijing date for journal folders.
cst_date() { date -d "+12 hours" "$@"; }

usage() {
  cat <<EOF
用法:
  ${0} [options]

功能:
  - 在登录节点执行时自动调用 sbatch 提交 local_transformers baseline 评测作业
  - 在计算节点执行时激活 conda 环境并运行 V0 评测（无剪枝，local transformers）
  - 将 Slurm stdout/stderr 和转发终端日志写入 training_journal/run_* 或 slurm_logs/baseline_transformers/smoke/

常用示例:
  ${0} --gpus 4
  ${0} --partition l20-gpu --account faculty --gpus 4
  ${0} --output-dir ${REPO_ROOT}/outputs/baseline_transformers --comment full-test

选项:
  --job-name NAME          Slurm job 名称，默认: ${DEFAULT_JOB_NAME}
  --config PATH            JSON 配置，默认: ${DEFAULT_CONFIG}
  --gpus N                 GPU 数量，默认: ${DEFAULT_GPUS}
  --partition NAME         Slurm partition，默认: ${DEFAULT_PARTITION}
  --account NAME           Slurm account，默认: ${DEFAULT_ACCOUNT}
  --time HH:MM:SS          作业时间限制，默认: ${DEFAULT_TIME}
  --cpus N                 每任务 CPU 数量，默认: ${DEFAULT_CPUS}
  --mem SIZE               内存，默认: ${DEFAULT_MEM}
  --conda-prefix PATH      conda 环境路径，默认: ${DEFAULT_CONDA_PREFIX}
  --output-dir PATH        评测输出目录；未显式指定时默认 ${DEFAULT_OUTPUT_DIR}_<run_stamp>（防覆盖）
  --split NAME             AndroidControl split 覆盖（如 validation / test）
  --measure-e2e-latency    评测时测量逐步端到端延迟
  --resume                 从 output-dir 断点继续评测（须显式指定 --output-dir，不能与自动时间戳目录联用）
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
  if [[ "${OUTPUT_DIR_EXPLICIT}" != "true" ]]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_${run_stamp}"
  fi
  echo "Output dir: ${OUTPUT_DIR}"
  local journal_run_dir="${REPO_ROOT}/training_journal/.tentative/run_${run_stamp}"
  mkdir -p "${journal_run_dir}"

  local output_pattern="${journal_run_dir}/slurm.out"
  local error_pattern="${journal_run_dir}/slurm.err"

  local export_vars=(
    "ALL"
    "BASELINE_REPO_ROOT=${REPO_ROOT}"
    "BASELINE_CONFIG=${CONFIG_PATH}"
    "BASELINE_CONDA_PREFIX=${CONDA_PREFIX_PATH}"
    "BASELINE_OUTPUT_DIR=${OUTPUT_DIR}"
    "BASELINE_SPLIT=${SPLIT}"
    "BASELINE_MEASURE_E2E=${MEASURE_E2E}"
    "BASELINE_RESUME=${RESUME}"
    "BASELINE_RUN_DATE=${run_date}"
    "BASELINE_RUN_STAMP=${run_stamp}"
    "BASELINE_JOURNAL_RUN_DIR=${journal_run_dir}"
    "BASELINE_JOB_SUBMITTED_AT=${run_stamp}"
    "BASELINE_JOB_COMMENT=${COMMENT}"
    "BASELINE_GPUS=${GPUS}"
  )

  local partition="${PARTITION:-${DEFAULT_PARTITION}}"
  local sbatch_cmd=(sbatch)
  sbatch_cmd+=(--job-name "${JOB_NAME}")
  sbatch_cmd+=(--nodes 1 --ntasks 1)
  sbatch_cmd+=(--partition "${partition}")
  sbatch_cmd+=(--gres "$(slurm_gres_string "${partition}" "${GPUS}")")
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
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run sbatch command:" >&2
    printf '  %q' "${sbatch_cmd[@]}" >&2
    printf '\n' >&2
    echo "Journal dir: ${journal_run_dir}"
    return 0
  fi

  if ! submit_output="$("${sbatch_cmd[@]}" 2>&1)"; then
    echo "sbatch failed: ${submit_output}" >&2
    exit 1
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
  local config_path="${BASELINE_CONFIG:-${DEFAULT_CONFIG}}"
  local conda_prefix="${BASELINE_CONDA_PREFIX:-${DEFAULT_CONDA_PREFIX}}"
  local output_dir="${BASELINE_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
  local split="${BASELINE_SPLIT:-}"
  local measure_e2e="${BASELINE_MEASURE_E2E:-false}"
  local resume="${BASELINE_RESUME:-false}"
  local run_date="${BASELINE_RUN_DATE:-$(cst_date '+%Y_%m_%d')}"
  local run_stamp="${BASELINE_RUN_STAMP:-$(cst_date '+%Y%m%d_%H%M%S')}"
  local journal_run_dir="${BASELINE_JOURNAL_RUN_DIR:-}"
  local comment="${BASELINE_JOB_COMMENT:-}"
  local requested_gpus="${BASELINE_GPUS:-${GPUS}}"

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

  export BASELINE_JOURNAL_RUN_DIR="${journal_run_dir}"
  export BASELINE_RUN_DATE="${run_date}"
  export BASELINE_RUN_STAMP="${run_stamp}"
  export PYTHONNOUSERSITE=1

  echo "[$(cst_date '+%F %T')] Starting baseline transformers job ${SLURM_JOB_ID:-local}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Config: ${config_path}"
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
    mode='baseline_transformers_eval',
    comment='${comment}',
)
"

  local client_exit=0
  local eval_args=("${python_bin}" "${REPO_ROOT}/scripts/core/run_evaluation.py"
    --benchmark AndroidControl
    --config "${config_path}"
    --variant V0
    --output-dir "${output_dir}")

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
      export BASELINE_JOURNAL_RUN_DIR="${journal_run_dir}"
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
CONFIG_PATH="${BASELINE_CONFIG:-${DEFAULT_CONFIG}}"
GPUS="${DEFAULT_GPUS}"
PARTITION="${DEFAULT_PARTITION}"
ACCOUNT="${DEFAULT_ACCOUNT}"
TIME_LIMIT="${DEFAULT_TIME}"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
CONDA_PREFIX_PATH="${DEFAULT_CONDA_PREFIX}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
SPLIT="${DEFAULT_SPLIT}"
MEASURE_E2E="${DEFAULT_MEASURE_E2E}"
RESUME="${DEFAULT_RESUME}"
COMMENT=""
DRY_RUN="false"
PARTITION_EXPLICIT="false"
OUTPUT_DIR_EXPLICIT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)             JOB_NAME="$2"; shift 2 ;;
    --config)               CONFIG_PATH="$2"; shift 2 ;;
    --gpus)                 GPUS="$2"; shift 2 ;;
    --partition)            PARTITION="$2"; PARTITION_EXPLICIT="true"; shift 2 ;;
    --account)              ACCOUNT="$2"; shift 2 ;;
    --time)                 TIME_LIMIT="$2"; shift 2 ;;
    --cpus)                 CPUS="$2"; shift 2 ;;
    --mem)                  MEM="$2"; shift 2 ;;
    --conda-prefix)         CONDA_PREFIX_PATH="$2"; shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2"; OUTPUT_DIR_EXPLICIT="true"; shift 2 ;;
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

if [[ "${RESUME}" == "true" && "${OUTPUT_DIR_EXPLICIT}" != "true" ]]; then
  echo "--resume requires explicit --output-dir (auto-stamped default dirs cannot be resumed)" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_job
else
  run_job
fi
