#!/usr/bin/env bash
#SBATCH --job-name=visionzip-ac
#SBATCH --partition=l20-gpu
#SBATCH --account=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/dkucc/home/rw335/SkillReuse
#SBATCH --output=/dkucc/home/rw335/SkillReuse/logs/slurm_visionzip_androidcontrol_%j.out
#SBATCH --error=/dkucc/home/rw335/SkillReuse/logs/slurm_visionzip_androidcontrol_%j.err
#
# Submit: sbatch scripts/androidcontrol/slurm/submit_visionzip_androidcontrol.sh
# GPU/partition params from SpectralMAE train_slurm_hsi.sh (l20-gpu, faculty, 4×GPU).
# Eval logic: VisionZip V0, configs/androidcontrol/visionzip/default.json, no LoRA/vLLM.

set -euo pipefail

PROJ=/dkucc/home/rw335/SkillReuse
cd "${PROJ}"

echo "[$(date '+%F %T')] SLURM job ${SLURM_JOB_ID:-local} starting on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

exec bash "${PROJ}/scripts/androidcontrol/eval/run_visionzip_eval.sh" --measure-e2e-latency "$@"
