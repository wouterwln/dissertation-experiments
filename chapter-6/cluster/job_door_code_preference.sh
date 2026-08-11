#!/usr/bin/env bash
#SBATCH --job-name=lp-dc-preference
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/door_code_preference_%x_%A_%a.out
#SBATCH --error=logs/door_code_preference_%x_%A_%a.err
# Required env: SUCC, FAIL, N_SHARDS
# AIF only — BAMCP/POMCP/BOSS/IDS/Belief-Q/Random ignore the goal prior
# C, so there is nothing to sweep for them. PSRL and Oracle do read C
# (both wrap VBP) but are excluded to bound GPU cost. AIF is
# GPU-eligible, hence gpu_a100.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source cluster/setup_env.sh
export JAX_PLATFORMS=cuda

SUCC_VAL="${SUCC:?SUCC not set}"
FAIL_VAL="${FAIL:?FAIL not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
CELL="succ${SUCC_VAL}_fail${FAIL_VAL}"

mkdir -p logs "output/door_code_preference/${CELL}/shards"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['door_code_preference']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) cell=${CELL} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_door_code.py \
    --planner aif \
    --K "$(read_param "['env']['K']")" \
    --lock-length "$(read_param "['env']['lock_length']")" \
    --probe-noise "$(read_param "['env']['probe_noise']")" \
    --success-logit "$SUCC_VAL" \
    --failure-logit "$FAIL_VAL" \
    --n-episodes "$(read_param "['runner']['n_episodes']")" \
    --max-steps "$(read_param "['runner']['max_steps']")" \
    --horizon "$(read_param "['runner']['horizon']")" \
    --seed "$(read_param "['runner']['seed']")" \
    --learn-on-failure "$(read_param "['runner']['learn_on_failure']")" \
    --clip-horizon "$(read_param "['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['planner']['iterations']")" \
    --planner-damping "$(read_param "['planner']['damping']")" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/door_code_preference/${CELL}/shards/aif.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done cell=${CELL} shard=${SHARD_INDEX} at $(date -Is)"
