#!/usr/bin/env bash
#SBATCH --job-name=lp-ft-preference
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/factored_tiger_preference_%x_%A_%a.out
#SBATCH --error=logs/factored_tiger_preference_%x_%A_%a.err
# Required env: GOOD, BAD, N_SHARDS
# Optional env: LISTEN_COST, OUT_DIR — set together by the Katt-2019
# listen_cost arm (DVC stage factored_tiger_listen_cost), which is the same
# runner call at the canonical goal prior with the environment charging per
# LISTEN. Default to params' listen_cost and the sweep's cell directory.
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

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

LISTEN_COST_VAL="${LISTEN_COST:-$(read_param "['factored_tiger']['env']['listen_cost']")}"
OUT_DIR="${OUT_DIR:-output/factored_tiger_preference/${CELL}}"
mkdir -p logs "${OUT_DIR}/shards"

TOTAL_SEEDS=$(read_param "['factored_tiger_preference']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) cell=${CELL} listen_cost=${LISTEN_COST_VAL} out=${OUT_DIR} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_factored_tiger.py \
    --planner aif \
    --K "$(read_param "['factored_tiger']['env']['K']")" \
    --listen-accuracy "$(read_param "['factored_tiger']['env']['listen_accuracy']")" \
    --open-safe-reward "$(read_param "['factored_tiger']['env']['open_safe_reward']")" \
    --open-tiger-penalty "$(read_param "['factored_tiger']['env']['open_tiger_penalty']")" \
    --listen-cost "$LISTEN_COST_VAL" \
    --good-logit "$GOOD_VAL" \
    --bad-logit "$BAD_VAL" \
    --n-episodes "$(read_param "['factored_tiger']['runner']['n_episodes']")" \
    --max-steps "$(read_param "['factored_tiger']['runner']['max_steps']")" \
    --horizon "$(read_param "['factored_tiger']['runner']['horizon']")" \
    --seed "$(read_param "['factored_tiger']['runner']['seed']")" \
    --clip-horizon "$(read_param "['factored_tiger']['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['factored_tiger']['planner']['iterations']")" \
    --planner-damping "$(read_param "['factored_tiger']['planner']['damping']")" \
    --workers 1 \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "${OUT_DIR}/shards/aif.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done cell=${CELL} out=${OUT_DIR} shard=${SHARD_INDEX} at $(date -Is)"
