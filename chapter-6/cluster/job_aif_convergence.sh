#!/usr/bin/env bash
#SBATCH --job-name=lp-aif-convergence
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/aif_convergence_%x_%A_%a.out
#SBATCH --error=logs/aif_convergence_%x_%A_%a.err
# Required env: ITERS, DAMP, N_SHARDS
# Array bounds are passed by the submit script: --array=0-(N_SHARDS-1)
#
# Convergence-sensitivity arm of job_door_code.sh. AIF only, so this is a
# GPU (gpu_a100) job like the other JAX planners. Each shard runs
# ceil(n_seeds/N_SHARDS) seeds at ITERS fixed-point iterations and DAMP
# damping. --time is sized for the 140-iteration cell: 2x the canonical
# 70-iteration door_code arm (00:15:00) plus margin.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source cluster/setup_env.sh
export JAX_PLATFORMS=cuda

ITERS="${ITERS:?ITERS not set}"
DAMP="${DAMP:?DAMP not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

CELL="iter${ITERS}_damp${DAMP}"
mkdir -p logs "output/aif_convergence/${CELL}/shards"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['aif_convergence']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=aif cell=${CELL} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_door_code.py \
    --planner aif \
    --K "$(read_param "['env']['K']")" \
    --lock-length "$(read_param "['env']['lock_length']")" \
    --probe-noise "$(read_param "['env']['probe_noise']")" \
    --n-episodes "$(read_param "['runner']['n_episodes']")" \
    --max-steps "$(read_param "['runner']['max_steps']")" \
    --horizon "$(read_param "['runner']['horizon']")" \
    --seed "$(read_param "['runner']['seed']")" \
    --learn-on-failure "$(read_param "['runner']['learn_on_failure']")" \
    --clip-horizon "$(read_param "['runner']['clip_horizon']")" \
    --planner-iterations "$ITERS" \
    --planner-damping "$DAMP" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/aif_convergence/${CELL}/shards/aif.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=aif cell=${CELL} shard=${SHARD_INDEX} at $(date -Is)"
