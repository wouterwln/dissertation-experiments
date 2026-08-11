#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-dc-simscaling
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=27G
# Was 02:30:00 for a single shard: worst cell (40000 sims) projects
# 0.99h -> 1.32h under depth 6 -> 8, inside that limit; planning time only,
# so a lower bound. A block's shards run concurrently, so its elapsed is
# the worst shard's rather than the sum; 03:30:00 adds contention margin,
# and genoa bills ELAPSED so the headroom is free in SBU.
#SBATCH --time=03:30:00
#SBATCH --output=logs/cpu_dc_simscaling_%x_%A_%a.out
#SBATCH --error=logs/cpu_dc_simscaling_%x_%A_%a.err
# Required env: SIMS, N_SHARDS, SHARD_LIST, PACK
# Array bounds are passed by the submit script: --array=0-(N_BLOCKS-1),
# where an array task is a BLOCK of PACK shards, not a single shard.
#
# BAMCP sims-budget ablation arm of job_cpu_door_code.sh. Each shard
# runs ceil(n_seeds/N_SHARDS) seeds of BAMCP at SIMS rollouts per
# plan() call; PACK shards share one allocation — see
# job_cpu_door_code.sh for why the billing floor makes a block the unit
# and why this is packing rather than --workers.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

SIMS="${SIMS:?SIMS not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_LIST="${SHARD_LIST:?SHARD_LIST not set}"
PACK="${PACK:?PACK not set}"
BLOCK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p logs "output/door_code_simscaling/sims${SIMS}/shards"

source cluster/_read_params.sh

TOTAL_SEEDS=$(read_param "['door_code_simscaling']['n_seeds']")

# This block's slice of the submit script's missing-shard list. Colon-
# separated, not comma: sbatch --export takes commas as its own separator.
IFS=':' read -ra ALL_SHARDS <<< "$SHARD_LIST"
SHARDS=("${ALL_SHARDS[@]:$((BLOCK_INDEX * PACK)):PACK}")
if [ "${#SHARDS[@]}" -eq 0 ]; then
    echo "block ${BLOCK_INDEX} is empty (${#ALL_SHARDS[@]} shards, pack=${PACK}) —" \
         "array bounds and SHARD_LIST disagree" >&2
    exit 1
fi

echo "host=$(hostname) date=$(date -Is) planner=bamcp sims=${SIMS}" \
     "block=${BLOCK_INDEX} pack=${PACK} shards=${SHARDS[*]}"
# No `import jax` banner here; see job_cpu_door_code.sh for why.

# Resolved once and shared by every shard in the block; see
# job_cpu_door_code.sh for why.
COMMON_ARGS=(
    --planner bamcp
    --K "$(read_param "['env']['K']")"
    --lock-length "$(read_param "['env']['lock_length']")"
    --probe-noise "$(read_param "['env']['probe_noise']")"
    --n-episodes "$(read_param "['runner']['n_episodes']")"
    --max-steps "$(read_param "['runner']['max_steps']")"
    --horizon "$(read_param "['runner']['horizon']")"
    --seed "$(read_param "['runner']['seed']")"
    --learn-on-failure "$(read_param "['runner']['learn_on_failure']")"
    --clip-horizon "$(read_param "['runner']['clip_horizon']")"
    --planner-iterations "$(read_param "['planner']['iterations']")"
    --planner-damping "$(read_param "['planner']['damping']")"
    --q-lr "$(read_param "['planner']['q_lr']")"
    --q-epsilon "$(read_param "['planner']['q_epsilon']")"
    --q-gamma "$(read_param "['planner']['q_gamma']")"
    --boss-n-samples "$(read_param "['planner']['boss_n_samples']")"
    --bamcp-n-simulations "$SIMS"
    --bamcp-depth "$(read_param "['planner']['bamcp_depth']")"
    --bamcp-ucb-c "$(read_param "['planner']['bamcp_ucb_c']")"
    --bamcp-gamma "$(read_param "['planner']['bamcp_gamma']")"
    --ids-bamcp-n-simulations "$(read_param "['planner']['ids_bamcp_n_simulations']")"
    --ids-epsilon-info "$(read_param "['planner']['ids_epsilon_info']")"
    --total-seeds "$TOTAL_SEEDS"
    --n-shards "$N_SHARDS"
)
require_all_set "${COMMON_ARGS[@]}"

# One process per shard, all launched at once — PACK is sized in
# params.yaml to fit the cores and memory this job is billed for.
pids=()
for idx in "${SHARDS[@]}"; do
    padded="$(printf '%02d' "$idx")"
    uv run --frozen --no-sync python scripts/run_door_code.py \
        "${COMMON_ARGS[@]}" \
        --shard-index "$idx" \
        --output "output/door_code_simscaling/sims${SIMS}/shards/bamcp.shard${padded}.json" \
        > "logs/cpu_dc_simscaling_s${SIMS}_shard${padded}.log" 2>&1 &
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
             "logs/cpu_dc_simscaling_s${SIMS}_shard$(printf '%02d' "${SHARDS[$i]}").log" >&2
        status=1
    fi
done

echo "done planner=bamcp sims=${SIMS} block=${BLOCK_INDEX}" \
     "shards=${SHARDS[*]} status=${status} at $(date -Is)"
exit "$status"
