#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-dc-rollout
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=27G
# 1h. A task is a BLOCK of PACK shards run concurrently, so elapsed is the
# worst shard's, not their sum: scripts/price_slate.py puts that at 0.35h
# (boss). ~3x margin, tighter than the sibling stages because the packed
# shards contend for the same 24 cores.
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_dc_rollout_%x_%A_%a.out
#SBATCH --error=logs/cpu_dc_rollout_%x_%A_%a.err
# Required env: PLANNER, POLICY, N_SHARDS, SHARD_LIST, PACK
# Array bounds are passed by the submit script: --array=0-(N_BLOCKS-1),
# where an array task is a BLOCK of PACK shards, not a single shard.
#
# Rollout-policy ablation arm of job_cpu_door_code.sh: uniform (the
# shipped default) vs eps_greedy_q (Guez et al. 2012's epsilon-greedy
# rollout against an off-policy-learned Q-hat). Each shard runs
# ceil(n_seeds/N_SHARDS) seeds of PLANNER; PACK shards share one
# allocation — see job_cpu_door_code.sh for why the billing floor makes a
# block the unit and why this is packing rather than --workers.
#
# Packing is safe for eps_greedy_q specifically because Q-hat lives on the
# planner instance and each shard is its own process, exactly as before:
# BAMCPPlanner.prepare_block() clears _q_rollout at every theta_true draw,
# so nothing is shared between shards in a block.
#
# --time matches job_cpu_door_code_depth.sh: eps_greedy_q adds one RNG
# draw and one Q-hat update per rollout step, which is negligible next to
# the categorical successor sampling already in the loop. A block's shards
# run concurrently, so its elapsed is the worst shard's rather than the
# sum; 03:00:00 -> 04:00:00 only adds contention margin, and genoa bills
# ELAPSED so the headroom is free in SBU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
POLICY="${POLICY:?POLICY not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_LIST="${SHARD_LIST:?SHARD_LIST not set}"
PACK="${PACK:?PACK not set}"
BLOCK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p logs "output/door_code_rollout/${POLICY}/shards"

source cluster/_read_params.sh

TOTAL_SEEDS=$(read_param "['door_code_rollout']['n_seeds']")

# This block's slice of the submit script's missing-shard list. Colon-
# separated, not comma: sbatch --export takes commas as its own separator.
IFS=':' read -ra ALL_SHARDS <<< "$SHARD_LIST"
SHARDS=("${ALL_SHARDS[@]:$((BLOCK_INDEX * PACK)):PACK}")
if [ "${#SHARDS[@]}" -eq 0 ]; then
    echo "block ${BLOCK_INDEX} is empty (${#ALL_SHARDS[@]} shards, pack=${PACK}) —" \
         "array bounds and SHARD_LIST disagree" >&2
    exit 1
fi

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} policy=${POLICY}" \
     "block=${BLOCK_INDEX} pack=${PACK} shards=${SHARDS[*]}"
# No `import jax` banner here; see job_cpu_door_code.sh for why.

# Resolved once and shared by every shard in the block; see
# job_cpu_door_code.sh for why.
COMMON_ARGS=(
    --planner "$PLANNER"
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
    --boss-n-samples "$(read_param "['planner']['boss_n_samples']")"
    --bamcp-n-simulations "$(read_param "['planner']['bamcp_n_simulations']")"
    --bamcp-depth "$(read_param "['planner']['bamcp_depth']")"
    --bamcp-ucb-c "$(read_param "['planner']['bamcp_ucb_c']")"
    --bamcp-gamma "$(read_param "['planner']['bamcp_gamma']")"
    --rollout-policy "$POLICY"
    --rollout-epsilon "$(read_param "['door_code_rollout']['epsilon']")"
    --rollout-q-lr "$(read_param "['door_code_rollout']['q_lr']")"
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
        --output "output/door_code_rollout/${POLICY}/shards/${PLANNER}.shard${padded}.json" \
        > "logs/cpu_dc_rollout_${PLANNER}_${POLICY}_shard${padded}.log" 2>&1 &
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
             "logs/cpu_dc_rollout_${PLANNER}_${POLICY}_shard$(printf '%02d' "${SHARDS[$i]}").log" >&2
        status=1
    fi
done

echo "done planner=${PLANNER} policy=${POLICY} block=${BLOCK_INDEX}" \
     "shards=${SHARDS[*]} status=${status} at $(date -Is)"
exit "$status"
