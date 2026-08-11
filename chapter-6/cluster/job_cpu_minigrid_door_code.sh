#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-minigrid
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
# 02:30:00 for a packed block, up from 01:30:00 for a single shard (which
# was itself raised from 00:45:00 for bamcp_depth = runner.horizon,
# 15 -> 25, x1.67; projected worst shard 0.50h -> 0.84h from
# output/minigrid_door_code/*.json per-seed planning_time_s at 1 seed/shard,
# planning time only, so a lower bound — JAX compile and `uv run` startup
# are on top). A block's shards run concurrently, so its elapsed is the
# worst shard's, not the sum; the extra margin covers contention. genoa
# bills ELAPSED, so this costs queue priority and no SBU.
#SBATCH --time=02:30:00
#SBATCH --output=logs/cpu_minigrid_%x_%A_%a.out
#SBATCH --error=logs/cpu_minigrid_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS, SHARD_LIST, PACK
# Optional env: OUT_DIR, SEEDS_OVERRIDE — shard directory and seed count,
# defaulted below; cluster/submit_pilot.sh sets both to run this same job at
# reduced scale outside output/<stage>/. See that script for why.
# Array bounds are passed by the submit script: --array=0-(N_BLOCKS-1),
# where an array task is a BLOCK of PACK shards, not a single shard.
#
# CPU sibling of job_minigrid_door_code.sh. Single-seed shards, PACK of
# them per allocation — see job_cpu_door_code.sh for why the billing floor
# makes a block the unit and why this is packing rather than --workers.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-output/minigrid_door_code}"
mkdir -p logs "${OUT_DIR}/shards"

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_LIST="${SHARD_LIST:?SHARD_LIST not set}"
PACK="${PACK:?PACK not set}"
BLOCK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

source cluster/_read_params.sh

TOTAL_SEEDS="${SEEDS_OVERRIDE:-$(read_param "['minigrid']['runner']['n_seeds']")}"

# This block's slice of the submit script's missing-shard list. Colon-
# separated, not comma: sbatch --export takes commas as its own separator.
IFS=':' read -ra ALL_SHARDS <<< "$SHARD_LIST"
SHARDS=("${ALL_SHARDS[@]:$((BLOCK_INDEX * PACK)):PACK}")
if [ "${#SHARDS[@]}" -eq 0 ]; then
    echo "block ${BLOCK_INDEX} is empty (${#ALL_SHARDS[@]} shards, pack=${PACK}) —" \
         "array bounds and SHARD_LIST disagree" >&2
    exit 1
fi

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} out=${OUT_DIR}" \
     "block=${BLOCK_INDEX} pack=${PACK} shards=${SHARDS[*]}"
# No `import jax` banner here; see job_cpu_door_code.sh for why.

# Resolved once and shared by every shard in the block; see
# job_cpu_door_code.sh for why (read_param is now a lookup into the single
# params.yaml read cluster/_read_params.sh does, and --frozen --no-sync
# skips project resolution).
COMMON_ARGS=(
    --planner "$PLANNER"
    --K "$(read_param "['minigrid']['env']['K']")"
    --lock-length "$(read_param "['minigrid']['env']['lock_length']")"
    --probe-noise "$(read_param "['minigrid']['env']['probe_noise']")"
    --grid-size "$(read_param "['minigrid']['env']['grid_size']")"
    --slip-prob "$(read_param "['minigrid']['env']['slip_prob']")"
    --agent-view-size "$(read_param "['minigrid']['env']['agent_view_size']")"
    --n-episodes "$(read_param "['minigrid']['runner']['n_episodes']")"
    --max-steps "$(read_param "['minigrid']['runner']['max_steps']")"
    --horizon "$(read_param "['minigrid']['runner']['horizon']")"
    --seed "$(read_param "['minigrid']['runner']['seed']")"
    --learn-on-failure "$(read_param "['minigrid']['runner']['learn_on_failure']")"
    --clip-horizon "$(read_param "['minigrid']['runner']['clip_horizon']")"
    --planner-iterations "$(read_param "['minigrid']['planner']['iterations']")"
    --planner-damping "$(read_param "['minigrid']['planner']['damping']")"
    --q-lr "$(read_param "['minigrid']['planner']['q_lr']")"
    --q-epsilon "$(read_param "['minigrid']['planner']['q_epsilon']")"
    --q-gamma "$(read_param "['minigrid']['planner']['q_gamma']")"
    --boss-n-samples "$(read_param "['minigrid']['planner']['boss_n_samples']")"
    --bamcp-n-simulations "$(read_param "['minigrid']['planner']['bamcp_n_simulations']")"
    --bamcp-depth "$(read_param "['minigrid']['planner']['bamcp_depth']")"
    --bamcp-ucb-c "$(read_param "['minigrid']['planner']['bamcp_ucb_c']")"
    --bamcp-gamma "$(read_param "['minigrid']['planner']['bamcp_gamma']")"
    --ids-bamcp-n-simulations "$(read_param "['minigrid']['planner']['ids_bamcp_n_simulations']")"
    --ids-epsilon-info "$(read_param "['minigrid']['planner']['ids_epsilon_info']")"
    --video-dir "output/minigrid_door_code/${PLANNER}/videos"
    --video-fps "$(read_param "['minigrid']['runner']['video_fps']")"
    --total-seeds "$TOTAL_SEEDS"
    --n-shards "$N_SHARDS"
)
require_all_set "${COMMON_ARGS[@]}"

# One process per shard, all launched at once — PACK is sized in
# params.yaml to fit the cores and memory this job is billed for.
#
# NOTE: --video-dir is per-planner and the runner names frames by the
# SHARD-LOCAL seed index, so every single-seed shard writes
# videos/0/<ep>.mp4. Those paths already collided across the 20
# simultaneous array tasks max_concurrent allowed, so packing does not
# change the situation; the videos are a debugging artifact, not a metric,
# and no JSON field depends on them.
pids=()
for idx in "${SHARDS[@]}"; do
    padded="$(printf '%02d' "$idx")"
    uv run --frozen --no-sync python scripts/run_minigrid_door_code.py \
        "${COMMON_ARGS[@]}" \
        --shard-index "$idx" \
        --output "${OUT_DIR}/shards/${PLANNER}.shard${padded}.json" \
        > "logs/cpu_minigrid_${PLANNER}_shard${padded}.log" 2>&1 &
    pids+=("$!")
done

# Every status is collected before exiting, rather than failing on the
# first: an early exit would orphan the block's other shards. A block with
# any failed shard must exit non-zero, or the afterok aggregate job would
# fire against a shard directory with holes in it.
status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "shard ${SHARDS[$i]} FAILED — see" \
             "logs/cpu_minigrid_${PLANNER}_shard$(printf '%02d' "${SHARDS[$i]}").log" >&2
        status=1
    fi
done

echo "done planner=${PLANNER} block=${BLOCK_INDEX} shards=${SHARDS[*]}" \
     "status=${status} at $(date -Is)"
exit "$status"
