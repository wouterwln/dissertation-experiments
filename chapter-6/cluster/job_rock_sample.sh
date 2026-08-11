#!/usr/bin/env bash
#SBATCH --job-name=lp-rocksample
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
# Left at 09:00:00. AIF's 25-seed shard measures 6.05h (33% headroom),
# and the persistent-oracle arms are separate array jobs, so their ~3.7h
# never stacks onto AIF's. Unaffected by the bamcp_depth change — these
# planners use runner.horizon, which did not move. Long --time requests
# cost queue priority on Snellius, so this is not raised speculatively.
#SBATCH --time=09:00:00
#SBATCH --output=logs/rock_sample_%x_%A_%a.out
#SBATCH --error=logs/rock_sample_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS
# Optional env: OUT_DIR, SEEDS_OVERRIDE — shard directory and seed count,
# defaulted below; cluster/submit_pilot.sh sets both to run this same job at
# reduced scale outside output/<stage>/. See that script for why.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-output/rock_sample}"
mkdir -p logs "${OUT_DIR}/shards"

source cluster/setup_env.sh
export JAX_PLATFORMS=cuda

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS="${SEEDS_OVERRIDE:-$(read_param "['rock_sample']['runner']['n_seeds']")}"

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} out=${OUT_DIR} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_rock_sample.py \
    --planner "$PLANNER" \
    --K "$(read_param "['rock_sample']['env']['K']")" \
    --grid-size "$(read_param "['rock_sample']['env']['grid_size']")" \
    --sensor-noise-eta "$(read_param "['rock_sample']['env']['sensor_noise_eta']")" \
    --sample-reward "$(read_param "['rock_sample']['env']['sample_reward']")" \
    --good-logit "$(read_param "['rock_sample']['env']['good_logit']")" \
    --bad-logit "$(read_param "['rock_sample']['env']['bad_logit']")" \
    --layout-seed "$(read_param "['rock_sample']['env']['layout_seed']")" \
    --n-episodes "$(read_param "['rock_sample']['runner']['n_episodes']")" \
    --max-steps "$(read_param "['rock_sample']['runner']['max_steps']")" \
    --horizon "$(read_param "['rock_sample']['runner']['horizon']")" \
    --seed "$(read_param "['rock_sample']['runner']['seed']")" \
    --clip-horizon "$(read_param "['rock_sample']['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['rock_sample']['planner']['iterations']")" \
    --planner-damping "$(read_param "['rock_sample']['planner']['damping']")" \
    --bamcp-n-simulations "$(read_param "['rock_sample']['planner']['bamcp_n_simulations']")" \
    --bamcp-depth "$(read_param "['rock_sample']['planner']['bamcp_depth']")" \
    --bamcp-ucb-c "$(read_param "['rock_sample']['planner']['bamcp_ucb_c']")" \
    --bamcp-gamma "$(read_param "['rock_sample']['planner']['bamcp_gamma']")" \
    --boss-n-samples "$(read_param "['rock_sample']['planner']['boss_n_samples']")" \
    --ids-bamcp-n-simulations "$(read_param "['rock_sample']['planner']['ids_bamcp_n_simulations']")" \
    --ids-epsilon-info "$(read_param "['rock_sample']['planner']['ids_epsilon_info']")" \
    --workers 1 \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "${OUT_DIR}/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
