#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-door-code
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=30G
# 01:30:00 for a packed block, up from 00:45:00 for a single shard. The
# block's shards run concurrently, so its elapsed is one shard's (worst
# shard, BOSS, 0.36h) plus whatever contention costs — not the sum. The
# margin is deliberately loose: genoa bills ELAPSED, so a high --time
# costs queue priority and nothing in SBU, while a block that times out
# wastes every shard in it.
#SBATCH --time=01:30:00
#SBATCH --output=logs/cpu_door_code_%x_%A_%a.out
#SBATCH --error=logs/cpu_door_code_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS, SHARD_LIST, PACK
# Optional env: OUT_DIR, SEEDS_OVERRIDE — shard directory and seed count,
# defaulted below; cluster/submit_pilot.sh sets both to run this same job at
# reduced scale outside output/<stage>/. See that script for why.
# Array bounds are passed by the submit script: --array=0-(N_BLOCKS-1),
# where an array task is a BLOCK of PACK shards, not a single shard.
#
# CPU sibling of job_door_code.sh. Each shard runs ceil(n_seeds/N_SHARDS)
# seeds of one CPU planner (bamcp/boss/pomcp/ids/belief_q).
#
# Why blocks: genoa bills max(cpus-per-task, ceil(mem_MiB/1792), 24) per
# job, the 24 being a 1/8-node floor (192 cores, at most 8 jobs per node).
# A 1-seed shard on 2 cores was therefore billed for 24 it never touched.
# This runs PACK shards per allocation instead, cutting the job count and
# the bill by ~PACK. The runner here has no --workers, and could not use
# one: every per-seed stream is f(shard_seed, s) with shard_seed = seed +
# shard_index * 1_000_003, so fewer shards would change every result.
# Packing changes neither n_shards nor shard_index nor any argument — the
# shards are the same processes they were, so the JSONs are byte-identical
# to the unpacked run. See params.yaml cluster.cpu for how PACK is sized.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-output/door_code}"
mkdir -p logs "${OUT_DIR}/shards"

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu
# No thread-count overrides here, and none in job_cpu_aif_timing.sh
# either: the two must declare an identical environment or the AIF timing
# anchor is not comparable with these baselines. Adding a cap here would
# only take effect on a rerun (DVC does not track env vars and the submit
# script skips planners whose JSON already exists), which would leave the
# anchor pinned against unpinned baselines.

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_LIST="${SHARD_LIST:?SHARD_LIST not set}"
PACK="${PACK:?PACK not set}"
BLOCK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

source cluster/_read_params.sh

TOTAL_SEEDS="${SEEDS_OVERRIDE:-$(read_param "['runner']['n_seeds']")}"

# This block's slice of the submit script's missing-shard list. Colon-
# separated, not comma: sbatch --export takes commas as its own separator,
# so a comma list would arrive split across bogus variables.
IFS=':' read -ra ALL_SHARDS <<< "$SHARD_LIST"
SHARDS=("${ALL_SHARDS[@]:$((BLOCK_INDEX * PACK)):PACK}")
if [ "${#SHARDS[@]}" -eq 0 ]; then
    echo "block ${BLOCK_INDEX} is empty (${#ALL_SHARDS[@]} shards, pack=${PACK}) —" \
         "array bounds and SHARD_LIST disagree" >&2
    exit 1
fi

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} out=${OUT_DIR}" \
     "block=${BLOCK_INDEX} pack=${PACK} shards=${SHARDS[*]}"
# No `import jax` banner here. It cost a 631-module import of the whole JAX
# stack per block to print one line, and every shard already records its
# backend in the JSON's config.runtime, so the information was redundant
# with something the artifact keeps.

# Resolved once and shared by every shard in the block, from the single
# params.yaml read that cluster/_read_params.sh does — read_param is now a
# lookup in that text, not its own interpreter. Reading them per shard would
# multiply startup metadata I/O by PACK against the project dir
# setup_env_cpu.sh warns about. --frozen --no-sync for the same reason: it
# skips project resolution, which PACK concurrent `uv run`s would all do at
# once.
COMMON_ARGS=(
    --planner "$PLANNER"
    --K "$(read_param "['env']['K']")"
    --lock-length "$(read_param "['env']['lock_length']")"
    --probe-noise "$(read_param "['env']['probe_noise']")"
    --success-logit "$(read_param "['env']['success_logit']")"
    --failure-logit "$(read_param "['env']['failure_logit']")"
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
    --bamcp-n-simulations "$(read_param "['planner']['bamcp_n_simulations']")"
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
# params.yaml to fit the cores and memory this job is billed for. Per-shard
# log files because PACK interleaved stdouts in one SLURM .out would be
# unreadable.
pids=()
for idx in "${SHARDS[@]}"; do
    padded="$(printf '%02d' "$idx")"
    uv run --frozen --no-sync python scripts/run_door_code.py \
        "${COMMON_ARGS[@]}" \
        --shard-index "$idx" \
        --output "${OUT_DIR}/shards/${PLANNER}.shard${padded}.json" \
        > "logs/cpu_door_code_${PLANNER}_shard${padded}.log" 2>&1 &
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
             "logs/cpu_door_code_${PLANNER}_shard$(printf '%02d' "${SHARDS[$i]}").log" >&2
        status=1
    fi
done

echo "done planner=${PLANNER} block=${BLOCK_INDEX} shards=${SHARDS[*]}" \
     "status=${status} at $(date -Is)"
exit "$status"
