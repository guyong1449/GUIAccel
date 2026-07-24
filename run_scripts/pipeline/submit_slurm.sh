#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# GUIAccel SLURM Submission Script
#
# Dual-mode:
#   - Login node (no SLURM_JOB_ID): sbatch-submit this script as a job
#   - Compute node (SLURM_JOB_ID set): orchestrate vLLM service + client pipeline
# --------------------------------------------------------------------------- #

if [[ -n "${GUIACCEL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${GUIACCEL_REPO_ROOT}"
  SCRIPT_DIR="${REPO_ROOT}/run_scripts"
elif [[ -n "${SKILLREUSE_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${SKILLREUSE_REPO_ROOT}"
  SCRIPT_DIR="${REPO_ROOT}/run_scripts"
else
_GUIACCEL_SEARCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_GUIACCEL_SEARCH}" != "/" ]]; do
  if [[ -f "${_GUIACCEL_SEARCH}/scripts/_lib/init.sh" ]]; then
    # shellcheck disable=SC1090
    source "${_GUIACCEL_SEARCH}/scripts/_lib/init.sh"
    guiaccel_init_from "${BASH_SOURCE[0]}"
    break
  fi
  _GUIACCEL_SEARCH="$(dirname "${_GUIACCEL_SEARCH}")"
done
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib/load_env.sh"
guiaccel_load_env "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/run_scripts/_lib/slurm_helpers.sh"

DEFAULT_CONFIG="${REPO_ROOT}/configs/androidcontrol/default.json"
DEFAULT_JOB_NAME="guiaccel"
DEFAULT_PARTITION="common-gpu"
DEFAULT_ACCOUNT=""
DEFAULT_TIME="7-00:00:00"
DEFAULT_GPUS="4"
DEFAULT_CPUS="16"
DEFAULT_MEM="160G"
DEFAULT_MODE="discovery"
DEFAULT_CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
DEFAULT_FRONT_PORT="8000"
DEFAULT_BASE_PORT="8100"
DEFAULT_VARIANT=""
DEFAULT_SKILL_LIBRARY=""
DEFAULT_OUTPUT_DIR=""
DEFAULT_RESUME="false"

HEALTHZ_TIMEOUT=600
HEALTHZ_INTERVAL=5

# 服务器是北美时间，+12h 得到北京时间
cst_date() { date -d "+12 hours" "$@"; }

usage() {
  cat <<EOF
用法:
  ${0} [options]

功能:
  - 在登录节点执行时自动调用 sbatch 提交作业
  - 在计算节点执行时启动 vLLM 服务、等待就绪、运行 client pipeline
  - 将 Slurm stdout/stderr 和转发终端日志统一写入 training_journal/run_* 目录

常用示例:
  ${0} --mode discovery --gpus 4
  ${0} --mode evaluation --skill-library-path outputs/discovery/lib.pkl --variant V1
  ${0} --mode end-to-end --gpus 4
  ${0} --mode vllm-only --gpus 4

选项:
  --job-name NAME            Slurm job 名称，默认: ${DEFAULT_JOB_NAME}
  --config PATH              JSON 配置文件，默认: ${DEFAULT_CONFIG}
  --mode MODE                运行模式: discovery|evaluation|end-to-end|vllm-only，默认: ${DEFAULT_MODE}
  --gpus N                   GPU 数量，默认: ${DEFAULT_GPUS}
  --partition NAME           Slurm partition，默认: ${DEFAULT_PARTITION}
  --account NAME             Slurm account
  --time HH:MM:SS            作业时间限制，默认: ${DEFAULT_TIME}
  --cpus N                   每任务 CPU 数量，默认: ${DEFAULT_CPUS}
  --mem SIZE                 内存，默认: ${DEFAULT_MEM}
  --conda-prefix PATH        conda 环境路径，默认: ${DEFAULT_CONDA_PREFIX}
  --front-port PORT          vLLM 前端端口，默认: ${DEFAULT_FRONT_PORT}
  --base-port PORT           vLLM 基础端口，默认: ${DEFAULT_BASE_PORT}
  --skill-library-path PATH  skill library .pkl (evaluation/end-to-end 模式需要)
  --variant NAME             评估变体 (V1, V2, ...)
  --output-dir PATH          输出目录 (不指定则按模式自动生成)
  --comment TEXT              追加到作业名的注释标签
  --resume                   从中断的 output-dir 恢复
  --dry-run                  仅打印 sbatch 命令，不提交
  -h, --help                 显示帮助

日志布局:
  - tentative 目录:  ${REPO_ROOT}/training_journal/.tentative/run_<YYYYMMDD_HHMMSS>/
  - promote 后目录:  ${REPO_ROOT}/training_journal/<YYYY_MM_DD>/run_<YYYYMMDD_HHMMSS>/
  - Slurm 原生日志:  <run-dir>/slurm.out | <run-dir>/slurm.err
  - 转发终端日志:    <run-dir>/terminal.log
  - vLLM 健康检查:   <run-dir>/vllm_health.json
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

# --------------------------------------------------------------------------- #
# submit_job — runs on the login node, creates tentative dir and sbatch
# --------------------------------------------------------------------------- #
submit_job() {
  local run_stamp
  run_stamp="$(cst_date '+%Y%m%d_%H%M%S')"
  local run_date
  run_date="$(cst_date '+%Y_%m_%d')"
  local journal_run_dir="${REPO_ROOT}/training_journal/.tentative/run_${run_stamp}"
  mkdir -p "${journal_run_dir}"

  local output_pattern="${journal_run_dir}/slurm.out"
  local error_pattern="${journal_run_dir}/slurm.err"

  local safe_job_name
  safe_job_name="$(sanitize_name "${JOB_NAME}")"

  local export_vars=(
    "ALL"
    "GUIACCEL_REPO_ROOT=${REPO_ROOT}"
    "GUIACCEL_CONFIG=${CONFIG_PATH}"
    "GUIACCEL_MODE=${MODE}"
    "GUIACCEL_CONDA_PREFIX=${CONDA_PREFIX_PATH}"
    "GUIACCEL_FRONT_PORT=${FRONT_PORT}"
    "GUIACCEL_BASE_PORT=${BASE_PORT}"
    "GUIACCEL_SKILL_LIBRARY=${SKILL_LIBRARY}"
    "GUIACCEL_VARIANT=${VARIANT}"
    "GUIACCEL_OUTPUT_DIR=${OUTPUT_DIR}"
    "GUIACCEL_RUN_DATE=${run_date}"
    "GUIACCEL_RUN_STAMP=${run_stamp}"
    "GUIACCEL_JOURNAL_RUN_DIR=${journal_run_dir}"
    "GUIACCEL_JOB_SUBMITTED_AT=${run_stamp}"
    "GUIACCEL_JOB_COMMENT=${COMMENT}"
    "GUIACCEL_RESUME=${RESUME}"
  )

  local gpu_list
  gpu_list="$(seq -s ',' 0 $(( GPUS - 1 )))"

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
    return 0
  fi

  local job_id
  job_id="$(awk '{print $NF}' <<<"${submit_output}")"
  if [[ -z "${job_id}" ]]; then
    echo "无法解析 sbatch 输出: ${submit_output}" >&2
    exit 1
  fi

  echo "${submit_output}"
  echo "Journal tentative dir: ${journal_run_dir}"
  echo "Promoted run dir: ${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
  echo "查看队列: squeue -j ${job_id}"
  echo "看 Slurm 输出: tail -f ${output_pattern}"
}

# --------------------------------------------------------------------------- #
# run_job — runs on the compute node
# --------------------------------------------------------------------------- #
run_job() {
  local config_path="${GUIACCEL_CONFIG:-${DEFAULT_CONFIG}}"
  local mode="${GUIACCEL_MODE:-${DEFAULT_MODE}}"
  local conda_prefix="${GUIACCEL_CONDA_PREFIX:-${DEFAULT_CONDA_PREFIX}}"
  local front_port="${GUIACCEL_FRONT_PORT:-${DEFAULT_FRONT_PORT}}"
  local base_port="${GUIACCEL_BASE_PORT:-${DEFAULT_BASE_PORT}}"
  local skill_library="${GUIACCEL_SKILL_LIBRARY:-}"
  local variant="${GUIACCEL_VARIANT:-}"
  local output_dir="${GUIACCEL_OUTPUT_DIR:-}"
  local run_date="${GUIACCEL_RUN_DATE:-$(cst_date '+%Y_%m_%d')}"
  local run_stamp="${GUIACCEL_RUN_STAMP:-$(cst_date '+%Y%m%d_%H%M%S')}"
  local journal_run_dir="${GUIACCEL_JOURNAL_RUN_DIR:-${REPO_ROOT}/training_journal/.tentative/run_${run_stamp}}"
  local comment="${GUIACCEL_JOB_COMMENT:-}"

  mkdir -p "${journal_run_dir}"
  local forwarded_log="${journal_run_dir}/terminal.log"

  exec > >(tee -a "${forwarded_log}") 2>&1

  cleanup_on_failure() {
    if [[ -d "${journal_run_dir}" ]]; then
      local aborted_dir="${REPO_ROOT}/training_journal/.aborted/$(basename "${journal_run_dir}")"
      mkdir -p "${aborted_dir}"
      for name in slurm.out slurm.err terminal.log config_snapshot.json pointers.json run_summary.md vllm_health.json; do
        if [[ -f "${journal_run_dir}/${name}" ]]; then
          cp -f "${journal_run_dir}/${name}" "${aborted_dir}/${name}"
        fi
      done
      rm -rf "${journal_run_dir}"
    fi
  }
  trap cleanup_on_failure EXIT

  export GUIACCEL_JOURNAL_RUN_DIR="${journal_run_dir}"
  export GUIACCEL_RUN_DATE="${run_date}"
  export GUIACCEL_RUN_STAMP="${run_stamp}"
  # Backward-compat for journal.py
  export SKILLREUSE_JOURNAL_RUN_DIR="${journal_run_dir}"
  export SKILLREUSE_JOB_SUBMITTED_AT="${GUIACCEL_JOB_SUBMITTED_AT:-${run_stamp}}"

  echo "[$(cst_date '+%F %T')] Starting GUIAccel job ${SLURM_JOB_ID:-local} (mode=${mode})"
  echo "Repo root: ${REPO_ROOT}"
  echo "Config: ${config_path}"
  echo "Mode: ${mode}"
  echo "Conda prefix: ${conda_prefix}"
  echo "Front port: ${front_port}"
  echo "Journal run dir: ${journal_run_dir}"

  cd "${REPO_ROOT}"

  # Activate conda environment
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_prefix}"
  else
    export PATH="${conda_prefix}/bin:${PATH}"
  fi
  export PYTHONNOUSERSITE=1

  local python_bin="${conda_prefix}/bin/python"

  local gpu_count="${SLURM_GPUS_ON_NODE:-${GPUS:-4}}"
  local gpu_list
  gpu_list="$(seq -s ',' 0 $(( gpu_count - 1 )))"

  if [[ -z "${output_dir}" ]]; then
    output_dir="${REPO_ROOT}/outputs/androidcontrol_${mode}"
  fi

  # Write journal artifacts
  "${python_bin}" -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from guiaccel.journal import prepare_journal
from pathlib import Path
prepare_journal(
    Path('${journal_run_dir}'),
    config_path='${config_path}',
    output_dir='${output_dir}',
    mode='${mode}',
    comment='${comment}',
)
"

  # ----- Start vLLM replicas in background ----- #
  echo "[$(cst_date '+%F %T')] Starting vLLM replicas (GPUs: ${gpu_list})..."
  local vllm_pid
  "${python_bin}" "${REPO_ROOT}/scripts/core/start_vllm_replicas.py" \
    --gpus "${gpu_list}" \
    --front-port "${front_port}" \
    --base-port "${base_port}" &
  vllm_pid=$!
  echo "vLLM background PID: ${vllm_pid}"

  # ----- Poll /healthz until ready ----- #
  echo "[$(cst_date '+%F %T')] Waiting for vLLM healthz (timeout=${HEALTHZ_TIMEOUT}s)..."
  local elapsed=0
  local healthy="false"
  while (( elapsed < HEALTHZ_TIMEOUT )); do
    if curl -sf "http://127.0.0.1:${front_port}/health" >/dev/null 2>&1 || \
       curl -sf "http://127.0.0.1:${front_port}/healthz" >/dev/null 2>&1; then
      healthy="true"
      break
    fi
    if ! kill -0 "${vllm_pid}" 2>/dev/null; then
      echo "ERROR: vLLM process died before becoming healthy" >&2
      wait "${vllm_pid}" || true
      exit 1
    fi
    sleep "${HEALTHZ_INTERVAL}"
    elapsed=$(( elapsed + HEALTHZ_INTERVAL ))
  done

  if [[ "${healthy}" != "true" ]]; then
    echo "ERROR: vLLM health check timed out after ${HEALTHZ_TIMEOUT}s" >&2
    kill "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
    exit 1
  fi

  echo "[$(cst_date '+%F %T')] vLLM healthy!"
  curl -sf "http://127.0.0.1:${front_port}/health" > "${journal_run_dir}/vllm_health.json" 2>/dev/null || \
    echo '{"status":"ok","checked_at":"'"$(cst_date -Iseconds)"'"}' > "${journal_run_dir}/vllm_health.json"

  # ----- Promote journal dir ----- #
  local final_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
  mkdir -p "$(dirname "${final_dir}")"
  mv "${journal_run_dir}" "${final_dir}"
  journal_run_dir="${final_dir}"
  export GUIACCEL_JOURNAL_RUN_DIR="${journal_run_dir}"
  export SKILLREUSE_JOURNAL_RUN_DIR="${journal_run_dir}"
  echo "[$(cst_date '+%F %T')] Journal promoted to: ${journal_run_dir}"

  trap - EXIT

  # ----- Run client pipeline ----- #
  local client_exit=0
  case "${mode}" in
    discovery)
      echo "[$(cst_date '+%F %T')] Running discovery pipeline..."
      local disc_args=("${python_bin}" "${REPO_ROOT}/scripts/core/run_discovery.py"
        --benchmark AndroidControl
        --config "${config_path}"
        --output-dir "${output_dir}")
      if [[ "${GUIACCEL_RESUME:-false}" == "true" ]]; then
        disc_args+=(--resume)
      fi
      "${disc_args[@]}" || client_exit=$?
      ;;
    evaluation)
      if [[ -z "${skill_library}" ]]; then
        echo "ERROR: --skill-library-path required for evaluation mode" >&2
        client_exit=1
      else
        echo "[$(cst_date '+%F %T')] Running evaluation pipeline..."
        local eval_args=("${python_bin}" "${REPO_ROOT}/scripts/core/run_evaluation.py"
          --benchmark AndroidControl
          --config "${config_path}"
          --skill-library-path "${skill_library}"
          --output-dir "${output_dir}")
        if [[ -n "${variant}" ]]; then
          eval_args+=(--variant "${variant}")
        fi
        "${eval_args[@]}" || client_exit=$?
      fi
      ;;
    end-to-end)
      echo "[$(cst_date '+%F %T')] Running end-to-end pipeline..."
      local e2e_args=("${python_bin}" "${REPO_ROOT}/scripts/core/run_end_to_end.py"
        --benchmark AndroidControl
        --config "${config_path}"
        --output-dir "${output_dir}")
      if [[ -n "${skill_library}" ]]; then
        e2e_args+=(--skill-library-path "${skill_library}")
      fi
      if [[ -n "${variant}" ]]; then
        e2e_args+=(--variant "${variant}")
      fi
      "${e2e_args[@]}" || client_exit=$?
      ;;
    vllm-only)
      echo "[$(cst_date '+%F %T')] vLLM-only mode: server running on port ${front_port}. Waiting for job timeout or signal..."
      wait "${vllm_pid}" || true
      vllm_pid=""
      ;;
    *)
      echo "ERROR: unknown mode '${mode}'" >&2
      client_exit=1
      ;;
  esac

  # ----- Tear down vLLM ----- #
  if [[ -n "${vllm_pid:-}" ]]; then
    echo "[$(cst_date '+%F %T')] Shutting down vLLM (PID=${vllm_pid})..."
    kill "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
  fi

  # ----- Update journal status ----- #
  local final_status="completed"
  if [[ ${client_exit} -ne 0 ]]; then
    final_status="failed"
  fi
  "${python_bin}" -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from guiaccel.journal import update_status
from pathlib import Path
update_status(Path('${journal_run_dir}'), '${final_status}')
"

  echo "[$(cst_date '+%F %T')] Done. Status: ${final_status} (exit code: ${client_exit})"
  exit ${client_exit}
}

# --------------------------------------------------------------------------- #
# CLI argument parsing
# --------------------------------------------------------------------------- #
JOB_NAME="${DEFAULT_JOB_NAME}"
CONFIG_PATH="${GUIACCEL_CONFIG:-${DEFAULT_CONFIG}}"
MODE="${DEFAULT_MODE}"
GPUS="${DEFAULT_GPUS}"
PARTITION="${DEFAULT_PARTITION}"
ACCOUNT="${DEFAULT_ACCOUNT}"
TIME_LIMIT="${DEFAULT_TIME}"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
CONDA_PREFIX_PATH="${DEFAULT_CONDA_PREFIX}"
FRONT_PORT="${DEFAULT_FRONT_PORT}"
BASE_PORT="${DEFAULT_BASE_PORT}"
SKILL_LIBRARY="${DEFAULT_SKILL_LIBRARY}"
VARIANT="${DEFAULT_VARIANT}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
COMMENT=""
DRY_RUN="false"
RESUME="${DEFAULT_RESUME}"
PARTITION_EXPLICIT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)        JOB_NAME="$2"; shift 2 ;;
    --config)          CONFIG_PATH="$2"; shift 2 ;;
    --mode)            MODE="$2"; shift 2 ;;
    --gpus)            GPUS="$2"; shift 2 ;;
    --partition)       PARTITION="$2"; PARTITION_EXPLICIT="true"; shift 2 ;;
    --account)         ACCOUNT="$2"; shift 2 ;;
    --time)            TIME_LIMIT="$2"; shift 2 ;;
    --cpus)            CPUS="$2"; shift 2 ;;
    --mem)             MEM="$2"; shift 2 ;;
    --conda-prefix)    CONDA_PREFIX_PATH="$2"; shift 2 ;;
    --front-port)      FRONT_PORT="$2"; shift 2 ;;
    --base-port)       BASE_PORT="$2"; shift 2 ;;
    --skill-library-path) SKILL_LIBRARY="$2"; shift 2 ;;
    --variant)         VARIANT="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --comment)         COMMENT="$2"; shift 2 ;;
    --resume)          RESUME="true"; shift ;;
    --dry-run)         DRY_RUN="true"; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# Validate config exists (login node only)
if [[ -z "${SLURM_JOB_ID:-}" && ! -f "${CONFIG_PATH}" ]]; then
  echo "配置文件不存在: ${CONFIG_PATH}" >&2
  exit 1
fi

# Dispatch
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_job
else
  run_job
fi
