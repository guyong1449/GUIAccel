#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# vLLM baseline evaluation SLURM script (AndroidControl V0)
#
# Dual-mode:
#   - Login node (no SLURM_JOB_ID): sbatch-submit this script
#   - Compute node (SLURM_JOB_ID set): start vLLM replicas + run evaluation
#
# Stability knobs (vs previous ad-hoc wrap):
#   - --mem default 256G (host RAM cgroup; not GPU VRAM)
#   - --enforce-eager + VLLM_DISABLE_COMPILE_CACHE + per-job VLLM_CACHE_ROOT
#   - --max-model-len 32768 (avoid KV OOM on A40)
#   - setsid + per-replica --log-dir
#   - PYTHONFAULTHANDLER=1
#   - on failure: dump diagnostics (health, dmesg OOM/segfault hints) WITHOUT
#     swallowing the original non-zero exit code
#
# Default config: configs/androidcontrol/baseline/vllm_stable.json
#   (client_max_parallel_requests=1, more frequent progress flush)
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

DEFAULT_CONFIG="${REPO_ROOT}/configs/androidcontrol/baseline/vllm_stable.json"
DEFAULT_JOB_NAME="baseline-vllm-fa2-full"
DEFAULT_PARTITION="common-gpu"
DEFAULT_ACCOUNT="faculty"
DEFAULT_TIME="16:00:00"
DEFAULT_GPUS="4"
DEFAULT_CPUS="16"
DEFAULT_MEM="256G"
DEFAULT_CONDA_PREFIX="${REPO_ROOT}/.conda/maiui-vllm"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs/baseline_vllm_fa2_full"
DEFAULT_SPLIT=""
DEFAULT_RESUME="false"
DEFAULT_FRONT_PORT="8000"
DEFAULT_BASE_PORT="8100"
DEFAULT_MAX_MODEL_LEN="32768"
DEFAULT_MAX_NUM_BATCHED_TOKENS="12288"
DEFAULT_MAX_NUM_SEQS="8"
DEFAULT_GPU_MEM_UTIL="0.90"
DEFAULT_TASK_LIMIT=""

# Server is US time; +12h gives Beijing date for journal folders.
cst_date() { date -d "+12 hours" "$@"; }

usage() {
  cat <<EOF
用法:
  ${0} [options]

功能:
  - 登录节点: sbatch 提交 vLLM baseline 评测
  - 计算节点: 启动 1-GPU-per-replica vLLM + proxy，再跑 V0 eval
  - 失败时写 diagnostics，但保留原始非零退出码（不吞错）

常用示例:
  ${0} --gpus 4 --partition common-gpu --account faculty
  ${0} --output-dir outputs/baseline_vllm_fa2_full --resume
  ${0} --task-limit 32 --time 02:00:00 --job-name baseline-vllm-smoke
  ${0} --dry-run

选项:
  --job-name NAME          默认: ${DEFAULT_JOB_NAME}
  --config PATH            默认: ${DEFAULT_CONFIG}
  --gpus N                 默认: ${DEFAULT_GPUS}
  --partition NAME         默认: ${DEFAULT_PARTITION}
  --account NAME           默认: ${DEFAULT_ACCOUNT}
  --time LIMIT             默认: ${DEFAULT_TIME}
  --cpus N                 默认: ${DEFAULT_CPUS}
  --mem SIZE               主机 RAM（cgroup），默认: ${DEFAULT_MEM}
  --conda-prefix PATH      默认: ${DEFAULT_CONDA_PREFIX}
  --output-dir PATH        未显式指定时自动加时间戳后缀
  --split NAME             AndroidControl split 覆盖
  --resume                 断点续跑（须显式 --output-dir）
  --task-limit N           传给 run_evaluation.py（冒烟用）
  --front-port N           proxy 端口，默认: ${DEFAULT_FRONT_PORT}
  --base-port N            首个 replica 端口，默认: ${DEFAULT_BASE_PORT}
  --max-model-len N        vLLM KV 上限，默认: ${DEFAULT_MAX_MODEL_LEN}
  --comment TEXT           作业注释
  --dry-run                只打印 sbatch，不提交
  -h, --help
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
    "BASELINE_RESUME=${RESUME}"
    "BASELINE_TASK_LIMIT=${TASK_LIMIT}"
    "BASELINE_RUN_DATE=${run_date}"
    "BASELINE_RUN_STAMP=${run_stamp}"
    "BASELINE_JOURNAL_RUN_DIR=${journal_run_dir}"
    "BASELINE_JOB_COMMENT=${COMMENT}"
    "BASELINE_GPUS=${GPUS}"
    "BASELINE_FRONT_PORT=${FRONT_PORT}"
    "BASELINE_BASE_PORT=${BASE_PORT}"
    "BASELINE_MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    "BASELINE_MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
    "BASELINE_MAX_NUM_SEQS=${MAX_NUM_SEQS}"
    "BASELINE_GPU_MEM_UTIL=${GPU_MEM_UTIL}"
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

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run sbatch command:" >&2
    printf '  %q' "${sbatch_cmd[@]}" >&2
    printf '\n' >&2
    echo "Journal dir: ${journal_run_dir}"
    echo "Config: ${CONFIG_PATH}"
    echo "Mem(cgroup host RAM): ${MEM}  GPUs: ${GPUS}  max-model-len: ${MAX_MODEL_LEN}"
    return 0
  fi

  local submit_output
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

gpu_id_list() {
  local n="${1}"
  local ids=()
  local i
  for ((i = 0; i < n; i++)); do
    ids+=("${i}")
  done
  (IFS=,; echo "${ids[*]}")
}

capture_vllm_diagnostics() {
  local diag_dir="$1"
  local front_port="$2"
  local base_port="$3"
  local n_gpus="$4"
  local reason="${5:-unknown}"

  mkdir -p "${diag_dir}"
  local diag_file="${diag_dir}/vllm_failure_diagnostics.txt"
  {
    echo "=== vLLM failure diagnostics ==="
    echo "time: $(cst_date '+%F %T')"
    echo "reason: ${reason}"
    echo "host: $(hostname)"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
    echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-<unset>}  SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
    echo "ReqMem/cgroup context: host RAM limit from sbatch --mem (default ${DEFAULT_MEM}; not GPU VRAM)"
    echo
    echo "--- proxy / replica health ---"
    curl -sf "http://127.0.0.1:${front_port}/healthz" && echo " proxy :${front_port} OK" || echo " proxy :${front_port} FAIL"
    local i port
    for ((i = 0; i < n_gpus; i++)); do
      port=$((base_port + i))
      if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo " replica :${port} OK"
      else
        echo " replica :${port} FAIL"
      fi
    done
    echo
    echo "--- nvidia-smi ---"
    nvidia-smi 2>&1 || true
    echo
    echo "--- dmesg (oom/kill/segfault/cuda, last 80 matching) ---"
    if command -v dmesg >/dev/null 2>&1; then
      dmesg -T 2>/dev/null | rg -i 'oom|killed process|out of memory|segfault|general protection|cuda|xid' | tail -n 80 \
        || dmesg 2>/dev/null | rg -i 'oom|killed process|out of memory|segfault|general protection|cuda|xid' | tail -n 80 \
        || echo "(dmesg unavailable or no matches)"
    else
      echo "(dmesg not found)"
    fi
    echo
    echo "--- replica log tails ---"
    if [[ -d "${diag_dir}/vllm_replicas" ]]; then
      local f
      for f in "${diag_dir}/vllm_replicas"/replica_*.err; do
        [[ -f "${f}" ]] || continue
        echo ">> $(basename "${f}") (last 40 lines)"
        tail -n 40 "${f}" || true
        echo
      done
    fi
  } >"${diag_file}" 2>&1 || true

  echo "[diagnostics] wrote ${diag_file}"
}

run_job() {
  local config_path="${BASELINE_CONFIG:-${DEFAULT_CONFIG}}"
  local conda_prefix="${BASELINE_CONDA_PREFIX:-${DEFAULT_CONDA_PREFIX}}"
  local output_dir="${BASELINE_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
  local split="${BASELINE_SPLIT:-}"
  local resume="${BASELINE_RESUME:-false}"
  local task_limit="${BASELINE_TASK_LIMIT:-}"
  local run_date="${BASELINE_RUN_DATE:-$(cst_date '+%Y_%m_%d')}"
  local run_stamp="${BASELINE_RUN_STAMP:-$(cst_date '+%Y%m%d_%H%M%S')}"
  local journal_run_dir="${BASELINE_JOURNAL_RUN_DIR:-}"
  local comment="${BASELINE_JOB_COMMENT:-}"
  local requested_gpus="${BASELINE_GPUS:-${GPUS}}"
  local front_port="${BASELINE_FRONT_PORT:-${DEFAULT_FRONT_PORT}}"
  local base_port="${BASELINE_BASE_PORT:-${DEFAULT_BASE_PORT}}"
  local max_model_len="${BASELINE_MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
  local max_num_batched_tokens="${BASELINE_MAX_NUM_BATCHED_TOKENS:-${DEFAULT_MAX_NUM_BATCHED_TOKENS}}"
  local max_num_seqs="${BASELINE_MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
  local gpu_mem_util="${BASELINE_GPU_MEM_UTIL:-${DEFAULT_GPU_MEM_UTIL}}"

  if [[ -z "${journal_run_dir}" ]]; then
    journal_run_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
  fi
  mkdir -p "${journal_run_dir}"

  local forwarded_log="${journal_run_dir}/terminal.log"
  exec > >(tee -a "${forwarded_log}") 2>&1

  local client_exit=0
  local vllm_pid=""
  local monitor_pid=""
  local replica_log_dir="${journal_run_dir}/vllm_replicas"
  mkdir -p "${replica_log_dir}"

  cleanup_on_failure() {
    if [[ -d "${journal_run_dir}" && "${journal_run_dir}" == *"/.tentative/"* ]]; then
      local aborted_dir="${REPO_ROOT}/training_journal/.aborted/$(basename "${journal_run_dir}")"
      mkdir -p "${aborted_dir}"
      for name in slurm.out slurm.err terminal.log config_snapshot.json pointers.json run_summary.md vllm_failure_diagnostics.txt; do
        if [[ -f "${journal_run_dir}/${name}" ]]; then
          cp -f "${journal_run_dir}/${name}" "${aborted_dir}/${name}"
        fi
      done
      if [[ -d "${replica_log_dir}" ]]; then
        rm -rf "${aborted_dir}/vllm_replicas"
        cp -a "${replica_log_dir}" "${aborted_dir}/vllm_replicas" 2>/dev/null || true
      fi
      rm -rf "${journal_run_dir}"
    fi
  }
  trap cleanup_on_failure EXIT

  stop_vllm() {
    if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
      kill "${monitor_pid}" 2>/dev/null || true
      wait "${monitor_pid}" 2>/dev/null || true
    fi
    if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" 2>/dev/null; then
      kill -TERM "${vllm_pid}" 2>/dev/null || true
      # Give replicas time to flush logs; then hard-kill process group if needed.
      local waited=0
      while kill -0 "${vllm_pid}" 2>/dev/null && [[ ${waited} -lt 20 ]]; do
        sleep 1
        waited=$((waited + 1))
      done
      if kill -0 "${vllm_pid}" 2>/dev/null; then
        kill -KILL "-${vllm_pid}" 2>/dev/null || kill -KILL "${vllm_pid}" 2>/dev/null || true
      fi
      wait "${vllm_pid}" 2>/dev/null || true
    fi
  }

  export BASELINE_JOURNAL_RUN_DIR="${journal_run_dir}"
  export BASELINE_RUN_DATE="${run_date}"
  export BASELINE_RUN_STAMP="${run_stamp}"
  export PYTHONNOUSERSITE=1
  export PYTHONFAULTHANDLER=1
  export VLLM_DISABLE_COMPILE_CACHE=1
  export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm_job_${SLURM_JOB_ID:-local_${run_stamp}}"
  mkdir -p "${VLLM_CACHE_ROOT}"

  echo "[$(cst_date '+%F %T')] Starting baseline vLLM job ${SLURM_JOB_ID:-local}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Config: ${config_path}"
  echo "Requested GPUs: ${requested_gpus}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "Conda prefix: ${conda_prefix}"
  echo "Output dir: ${output_dir}"
  echo "Journal run dir: ${journal_run_dir}"
  echo "Host mem request (cgroup): check sbatch --mem (default ${DEFAULT_MEM})"
  echo "vLLM: max_model_len=${max_model_len} enforce_eager=1 cache=${VLLM_CACHE_ROOT}"

  cd "${REPO_ROOT}"

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_prefix}"
  else
    export PATH="${conda_prefix}/bin:${PATH}"
  fi

  local python_bin="${conda_prefix}/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    echo "ERROR: python not found at ${python_bin}" >&2
    exit 1
  fi

  "${python_bin}" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from skillreuse.journal import prepare_journal
from pathlib import Path
prepare_journal(
    Path('${journal_run_dir}'),
    config_path='${config_path}',
    output_dir='${output_dir}',
    mode='baseline_vllm_eval',
    comment='${comment}',
)
"

  local gpu_list
  gpu_list="$(gpu_id_list "${requested_gpus}")"
  local model_path
  model_path="$("${python_bin}" -c "
import json
from pathlib import Path
cfg = json.loads(Path('${config_path}').read_text())
src = cfg.get('model', {}).get('source') or cfg.get('paths', {}).get('base_model_path')
p = Path(src)
if not p.is_absolute():
    p = Path('${REPO_ROOT}') / p
print(p)
")"

  echo "[$(cst_date '+%F %T')] Launching vLLM replicas GPUs=${gpu_list} model=${model_path}"
  # setsid: own process group so SIGINT to the shell does not silently tear proxy only.
  setsid "${python_bin}" "${REPO_ROOT}/scripts/core/start_vllm_replicas.py" \
    --gpus "${gpu_list}" \
    --model "${model_path}" \
    --served-model-name Qwen3-VL-8B \
    --front-port "${front_port}" \
    --base-port "${base_port}" \
    --gpu-memory-utilization "${gpu_mem_util}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-model-len "${max_model_len}" \
    --trust-remote-code \
    --enforce-eager \
    --log-dir "${replica_log_dir}" \
    >>"${journal_run_dir}/vllm_launcher.log" 2>&1 &
  vllm_pid=$!
  echo "vLLM launcher PID (setsid): ${vllm_pid}"

  local ready=0
  local i
  for i in $(seq 1 120); do
    if ! kill -0 "${vllm_pid}" 2>/dev/null; then
      echo "ERROR: vLLM launcher exited before healthy" >&2
      capture_vllm_diagnostics "${journal_run_dir}" "${front_port}" "${base_port}" "${requested_gpus}" "launcher_exited_early"
      wait "${vllm_pid}" 2>/dev/null || client_exit=$?
      client_exit=${client_exit:-1}
      break
    fi
    if curl -sf "http://127.0.0.1:${front_port}/healthz" >/dev/null 2>&1; then
      ready=1
      echo "[$(cst_date '+%F %T')] vLLM proxy healthy on :${front_port}"
      break
    fi
    sleep 5
  done

  if [[ ${ready} -ne 1 ]]; then
    echo "ERROR: vLLM proxy not healthy within timeout" >&2
    capture_vllm_diagnostics "${journal_run_dir}" "${front_port}" "${base_port}" "${requested_gpus}" "proxy_not_healthy"
    stop_vllm
    exit 1
  fi

  # Background health monitor: record diagnostics on first unhealthy replica, do not kill eval.
  (
    while true; do
      sleep 30
      if ! curl -sf "http://127.0.0.1:${front_port}/healthz" >/dev/null 2>&1; then
        capture_vllm_diagnostics "${journal_run_dir}" "${front_port}" "${base_port}" "${requested_gpus}" "proxy_became_unhealthy"
        exit 0
      fi
      _p=0
      while [[ ${_p} -lt ${requested_gpus} ]]; do
        if ! curl -sf "http://127.0.0.1:$((base_port + _p))/health" >/dev/null 2>&1; then
          capture_vllm_diagnostics "${journal_run_dir}" "${front_port}" "${base_port}" "${requested_gpus}" "replica_port_$((base_port + _p))_unhealthy"
          exit 0
        fi
        _p=$((_p + 1))
      done
    done
  ) &
  monitor_pid=$!

  local eval_args=("${python_bin}" "${REPO_ROOT}/scripts/core/run_evaluation.py"
    --benchmark AndroidControl
    --config "${config_path}"
    --variant V0
    --output-dir "${output_dir}")

  if [[ -n "${split}" ]]; then
    eval_args+=(--split "${split}")
  fi
  if [[ "${resume}" == "true" ]]; then
    eval_args+=(--resume)
  fi
  if [[ -n "${task_limit}" ]]; then
    eval_args+=(--task-limit "${task_limit}")
  fi

  echo "[$(cst_date '+%F %T')] Launching: ${eval_args[*]}"
  set +e
  "${eval_args[@]}"
  client_exit=$?
  set -e

  if [[ ${client_exit} -ne 0 ]]; then
    echo "[$(cst_date '+%F %T')] eval failed with exit=${client_exit}; capturing diagnostics"
    capture_vllm_diagnostics "${journal_run_dir}" "${front_port}" "${base_port}" "${requested_gpus}" "eval_exit_${client_exit}"
  fi

  stop_vllm

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
" || true

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
RESUME="${DEFAULT_RESUME}"
TASK_LIMIT="${DEFAULT_TASK_LIMIT}"
FRONT_PORT="${DEFAULT_FRONT_PORT}"
BASE_PORT="${DEFAULT_BASE_PORT}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
MAX_NUM_BATCHED_TOKENS="${DEFAULT_MAX_NUM_BATCHED_TOKENS}"
MAX_NUM_SEQS="${DEFAULT_MAX_NUM_SEQS}"
GPU_MEM_UTIL="${DEFAULT_GPU_MEM_UTIL}"
COMMENT=""
DRY_RUN="false"
OUTPUT_DIR_EXPLICIT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)             JOB_NAME="$2"; shift 2 ;;
    --config)               CONFIG_PATH="$2"; shift 2 ;;
    --gpus)                 GPUS="$2"; shift 2 ;;
    --partition)            PARTITION="$2"; shift 2 ;;
    --account)              ACCOUNT="$2"; shift 2 ;;
    --time)                 TIME_LIMIT="$2"; shift 2 ;;
    --cpus)                 CPUS="$2"; shift 2 ;;
    --mem)                  MEM="$2"; shift 2 ;;
    --conda-prefix)         CONDA_PREFIX_PATH="$2"; shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2"; OUTPUT_DIR_EXPLICIT="true"; shift 2 ;;
    --split)                SPLIT="$2"; shift 2 ;;
    --resume)               RESUME="true"; shift ;;
    --task-limit)           TASK_LIMIT="$2"; shift 2 ;;
    --front-port)           FRONT_PORT="$2"; shift 2 ;;
    --base-port)            BASE_PORT="$2"; shift 2 ;;
    --max-model-len)        MAX_MODEL_LEN="$2"; shift 2 ;;
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
