#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-rs-depth
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
# 1h — see the reservation note in job_cpu_rock_sample.sh for why the 12h
# this replaces was root-cancelled at start. Only bamcp runs here (IDS is
# not swept over depth), and rock_sample.planner.bamcp_depth is now 15, the
# TOP of depth_sweep, so scripts/price_slate.py's 0.18h prices the most
# expensive cell in the sweep, not the canonical one. ~5x margin.
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_rs_depth_%x_%A_%a.out
#SBATCH --error=logs/cpu_rs_depth_%x_%A_%a.err
# Required env: PLANNER, DEPTH, N_SHARDS, WORKERS
#
# MCTS depth-sensitivity arm of job_cpu_rock_sample.sh. Each shard runs
# ceil(n_seeds/N_SHARDS) seeds of PLANNER at DEPTH total simulated
# lookahead (tree descent + leaf rollout). DEPTH is fed to BOTH
# --bamcp-depth and --ids-bamcp-depth so the cell name means the same
# thing for both planners in the slate.
#
# The cell list lives in submit_cpu_rock_sample_depth.sh: bamcp at every
# rock_sample_depth.depth_sweep entry, and nothing else — IDS is not swept
# over depth, at ~110x more per seed for a curve profiling already shows is
# flat (see the params.yaml block). This job script itself is cell-agnostic,
# so it still passes both depth flags.
#
# --time is therefore sized for the deepest BAMCP cell: ~2.1 min/seed, and
# MCTS cost grows ~linearly in depth but the sweep tops out at the canonical
# depth=15 (min(depth, horizon) saturates above it — see params.yaml), so no
# cell exceeds the canonical per-seed wall-clock.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
DEPTH="${DEPTH:?DEPTH not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
WORKERS="${WORKERS:?WORKERS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p logs "output/rock_sample_depth/depth${DEPTH}/shards"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['rock_sample_depth']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} depth=${DEPTH} shard=${SHARD_INDEX}/${N_SHARDS} workers=${WORKERS}"
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
    --bamcp-depth "$DEPTH" \
    --bamcp-ucb-c "$(read_param "['rock_sample']['planner']['bamcp_ucb_c']")" \
    --bamcp-gamma "$(read_param "['rock_sample']['planner']['bamcp_gamma']")" \
    --ids-bamcp-n-simulations "$(read_param "['rock_sample']['planner']['ids_bamcp_n_simulations']")" \
    --ids-bamcp-depth "$DEPTH" \
    --ids-rollout-seed "$(read_param "['rock_sample']['planner']['ids_rollout_seed']")" \
    --ids-epsilon-info "$(read_param "['rock_sample']['planner']['ids_epsilon_info']")" \
    --workers "$WORKERS" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/rock_sample_depth/depth${DEPTH}/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} depth=${DEPTH} shard=${SHARD_INDEX} at $(date -Is)"
