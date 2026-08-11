#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-rs-kscaling
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=21
#SBATCH --mem=36G
# 2h, mirroring job_cpu_rock_sample.sh — see the reservation note there
# for why the old 14h was root-cancelled at start. Worst cell is still
# K=4 IDS, but the info-gain prune (b4b8985/d731c92) takes it from the
# ~8.2h that sized the 14h to 0.40h projected by scripts/price_slate.py;
# K=2 and K=3 come out at 0.10h and 0.14h. Planning time only, so a
# lower bound.
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpu_rs_kscaling_%x_%A_%a.out
#SBATCH --error=logs/cpu_rs_kscaling_%x_%A_%a.err
# Required env: PLANNER, K, N_SHARDS, WORKERS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
K_VAL="${K:?K not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
WORKERS="${WORKERS:?WORKERS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p logs "output/rock_sample_kscaling/K${K_VAL}/shards"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['rock_sample']['runner']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} K=${K_VAL} shard=${SHARD_INDEX}/${N_SHARDS} workers=${WORKERS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_rock_sample.py \
    --planner "$PLANNER" \
    --K "$K_VAL" \
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
    --workers "$WORKERS" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/rock_sample_kscaling/K${K_VAL}/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} K=${K_VAL} shard=${SHARD_INDEX} at $(date -Is)"
