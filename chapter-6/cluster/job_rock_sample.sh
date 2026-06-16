#!/usr/bin/env bash
#SBATCH --job-name=lp-rocksample
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=logs/rock_sample_%x_%A_%a.out
#SBATCH --error=logs/rock_sample_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output/rock_sample/shards

source cluster/setup_env.sh
export JAX_PLATFORMS=cuda

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['rock_sample']['runner']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} shard=${SHARD_INDEX}/${N_SHARDS}"
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
    --output "output/rock_sample/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
