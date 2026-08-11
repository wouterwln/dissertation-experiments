#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-rocksample
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=21
#SBATCH --mem=36G
# 2h. The 14h this replaces sized the array on IDS at ~8.2h, measured
# BEFORE b4b8985/d731c92 pruned info-gain enumeration to states carrying
# predictive mass (12.947 -> 0.262 s/decision, 49.4x). IDS was "the one
# arm with no slack", so that one stale number set the whole reservation.
# scripts/price_slate.py projects the post-prune worst shard — still IDS,
# 5 seeds/shard over workers=4 — at 0.41h, so 2h is ~5x margin.
# The reservation, not the spend, is the binding constraint: genoa bills
# 24 cores per job whatever it asks for, so 14h reserved 10*24*14 = 3360
# SBU per array against a 5612 SBU balance, and `excess usage allowed: no`
# root-cancels at start anything the budget cannot cover. That is what
# killed the 2026-08-06 wave one second into each job.
# Planning time only, so a lower bound: JAX compile and `uv run` startup
# are on top of the 0.41h.
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpu_rock_sample_%x_%A_%a.out
#SBATCH --error=logs/cpu_rock_sample_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS, WORKERS
# Optional env: OUT_DIR, SEEDS_OVERRIDE — shard directory and seed count,
# defaulted below; cluster/submit_pilot.sh sets both to run this same job at
# reduced scale outside output/<stage>/. See that script for why.
#
# CPU sibling of job_rock_sample.sh. Uses multiprocessing.ProcessPoolExecutor
# (spawn) inside one SLURM task to parallelize across seeds.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-output/rock_sample}"
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

TOTAL_SEEDS="${SEEDS_OVERRIDE:-$(read_param "['rock_sample']['runner']['n_seeds']")}"

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} out=${OUT_DIR} shard=${SHARD_INDEX}/${N_SHARDS} workers=${WORKERS}"
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
    --workers "$WORKERS" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "${OUT_DIR}/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
