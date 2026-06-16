#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-factored-tiger
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=7
#SBATCH --mem=12G
#SBATCH --time=16:00:00
#SBATCH --output=logs/cpu_factored_tiger_%x_%A_%a.out
#SBATCH --error=logs/cpu_factored_tiger_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS, WORKERS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output/factored_tiger/shards

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
WORKERS="${WORKERS:?WORKERS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['factored_tiger']['runner']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} shard=${SHARD_INDEX}/${N_SHARDS} workers=${WORKERS}"
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
    --workers "$WORKERS" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/factored_tiger/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
