#!/usr/bin/env bash
#SBATCH --job-name=lp-rs-preference
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=logs/rock_sample_preference_%x_%A_%a.out
#SBATCH --error=logs/rock_sample_preference_%x_%A_%a.err
# Required env: GOOD, BAD, N_SHARDS — one (GOOD, BAD) pair per cell of
# params.yaml's rock_sample_preference.cells (the 6-cell cross), exported
# by cluster/submit_rock_sample_preference.sh.
# AIF only — BAMCP/Factored-BAMCP/BOSS/IDS/Random plan on the env reward
# table and never read the goal prior C, so there is nothing to sweep for
# them. PSRL and Oracle do read C (both wrap VBP) but are excluded to
# bound GPU cost. AIF is GPU-eligible, hence gpu_a100.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source cluster/setup_env.sh
export JAX_PLATFORMS=cuda

GOOD_VAL="${GOOD:?GOOD not set}"
BAD_VAL="${BAD:?BAD not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
CELL="good${GOOD_VAL}_bad${BAD_VAL}"

mkdir -p logs "output/rock_sample_preference/${CELL}/shards"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['rock_sample_preference']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) cell=${CELL} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_rock_sample.py \
    --planner aif \
    --K "$(read_param "['rock_sample']['env']['K']")" \
    --grid-size "$(read_param "['rock_sample']['env']['grid_size']")" \
    --sensor-noise-eta "$(read_param "['rock_sample']['env']['sensor_noise_eta']")" \
    --sample-reward "$(read_param "['rock_sample']['env']['sample_reward']")" \
    --good-logit "$GOOD_VAL" \
    --bad-logit "$BAD_VAL" \
    --layout-seed "$(read_param "['rock_sample']['env']['layout_seed']")" \
    --n-episodes "$(read_param "['rock_sample']['runner']['n_episodes']")" \
    --max-steps "$(read_param "['rock_sample']['runner']['max_steps']")" \
    --horizon "$(read_param "['rock_sample']['runner']['horizon']")" \
    --seed "$(read_param "['rock_sample']['runner']['seed']")" \
    --clip-horizon "$(read_param "['rock_sample']['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['rock_sample']['planner']['iterations']")" \
    --planner-damping "$(read_param "['rock_sample']['planner']['damping']")" \
    --workers 1 \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/rock_sample_preference/${CELL}/shards/aif.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done cell=${CELL} shard=${SHARD_INDEX} at $(date -Is)"
