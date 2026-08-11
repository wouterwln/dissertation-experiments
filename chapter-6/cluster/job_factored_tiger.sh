#!/usr/bin/env bash
#SBATCH --job-name=lp-factored-tiger
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/factored_tiger_%x_%A_%a.out
#SBATCH --error=logs/factored_tiger_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS
# Optional env: OUT_DIR, SEEDS_OVERRIDE — shard directory and seed count,
# defaulted below; cluster/submit_pilot.sh sets both to run this same job at
# reduced scale outside output/<stage>/. See that script for why.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-output/factored_tiger}"
mkdir -p logs "${OUT_DIR}/shards"

source cluster/setup_env.sh
export JAX_PLATFORMS=cuda

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS="${SEEDS_OVERRIDE:-$(read_param "['factored_tiger']['runner']['n_seeds']")}"

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} out=${OUT_DIR} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_factored_tiger.py \
    --planner "$PLANNER" \
    --K "$(read_param "['factored_tiger']['env']['K']")" \
    --listen-accuracy "$(read_param "['factored_tiger']['env']['listen_accuracy']")" \
    --open-safe-reward "$(read_param "['factored_tiger']['env']['open_safe_reward']")" \
    --open-tiger-penalty "$(read_param "['factored_tiger']['env']['open_tiger_penalty']")" \
    --listen-cost "$(read_param "['factored_tiger']['env']['listen_cost']")" \
    --good-logit "$(read_param "['factored_tiger']['env']['good_logit']")" \
    --bad-logit "$(read_param "['factored_tiger']['env']['bad_logit']")" \
    --n-episodes "$(read_param "['factored_tiger']['runner']['n_episodes']")" \
    --max-steps "$(read_param "['factored_tiger']['runner']['max_steps']")" \
    --horizon "$(read_param "['factored_tiger']['runner']['horizon']")" \
    --seed "$(read_param "['factored_tiger']['runner']['seed']")" \
    --clip-horizon "$(read_param "['factored_tiger']['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['factored_tiger']['planner']['iterations']")" \
    --planner-damping "$(read_param "['factored_tiger']['planner']['damping']")" \
    --bamcp-n-simulations "$(read_param "['factored_tiger']['planner']['bamcp_n_simulations']")" \
    --bamcp-depth "$(read_param "['factored_tiger']['planner']['bamcp_depth']")" \
    --bamcp-ucb-c "$(read_param "['factored_tiger']['planner']['bamcp_ucb_c']")" \
    --bamcp-gamma "$(read_param "['factored_tiger']['planner']['bamcp_gamma']")" \
    --boss-n-samples "$(read_param "['factored_tiger']['planner']['boss_n_samples']")" \
    --ids-bamcp-n-simulations "$(read_param "['factored_tiger']['planner']['ids_bamcp_n_simulations']")" \
    --ids-epsilon-info "$(read_param "['factored_tiger']['planner']['ids_epsilon_info']")" \
    --workers 1 \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "${OUT_DIR}/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
