#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-factored-tiger
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=7
#SBATCH --mem=12G
# 1h. The 16h this replaces predated the info-gain prune (b4b8985/d731c92)
# and was the worst offender in the slate at ~84x the real cost — see the
# reservation note in job_cpu_rock_sample.sh. scripts/price_slate.py
# projects the worst shard (IDS, 5 seeds/shard over workers=2) at 0.19h,
# so 1h is ~5x margin. Planning time only, so a lower bound.
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_factored_tiger_%x_%A_%a.out
#SBATCH --error=logs/cpu_factored_tiger_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS, WORKERS
# Optional env: OUT_DIR, SEEDS_OVERRIDE — shard directory and seed count,
# defaulted below; cluster/submit_pilot.sh sets both to run this same job at
# reduced scale outside output/<stage>/. See that script for why.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-output/factored_tiger}"
mkdir -p logs "${OUT_DIR}/shards"

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
WORKERS="${WORKERS:?WORKERS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS="${SEEDS_OVERRIDE:-$(read_param "['factored_tiger']['runner']['n_seeds']")}"

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} out=${OUT_DIR} shard=${SHARD_INDEX}/${N_SHARDS} workers=${WORKERS}"
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
    --output "${OUT_DIR}/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
