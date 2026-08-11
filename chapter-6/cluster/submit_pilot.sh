#!/usr/bin/env bash
# Calibration pilot — measure what the full slate will cost in SBU before
# committing the budget to it. Submits ONE reduced-scale shard per
# (benchmark, planner) cell through the real job templates, writing to
# output/_pilot/ instead of output/<stage>/, then
# `uv run python scripts/price_slate.py --results-dir output/_pilot` turns
# the measurements into a slate total.
#
# Projected cost of this script: ~140 SBU, about 0.6% of the slate it
# prices. The dominant cells are RockSample on the A100.
#
# WHY THIS SHAPE
#
# Reduce SEEDS, never EPISODES. Per-decision cost falls steeply as the
# posterior sharpens — door_code AIF runs 1.33 s/decision in episode 0 and
# 0.126 by episode 6 — so a truncated-episode pilot would overprice the
# slate by ~3x. Seeds are nearly free to cut where the money is: RockSample
# AIF spends 86.8 s per episode with a 0.010 coefficient of variation across
# 50 seeds, because every episode runs the full max_steps. The tree-search
# planners are under 0.09 everywhere. The two cells that DO vary seed to
# seed (door_code AIF at 0.81, minigrid AIF at 0.47) are ~1300 SBU of the
# slate between them, so even a 50% miss there is small against the total.
#
# Seeds are set to WORKERS, not to 1, on the two stages that fan out across
# workers: one full wave measures the memory-bandwidth contention the real
# shards will actually run under, which a solo seed would not see.
#
# The per-shard fixed cost (uv startup, XLA compiles, Numba JIT) comes out
# of the same run for free, because it is confined to the first seed: seed
# 0 episode 0 of the Jun-2026 RockSample run took 129.4 s against a steady
# 86.8, and seed 25 — the next shard's first seed — took 129.2. So one
# reduced run yields both the steady per-seed cost and the fixed cost, and
# `--warmup-planner` (which only run_door_code.py exposes) is not needed.
#
# NO AGGREGATE JOBS ARE QUEUED, deliberately. An aggregated JSON under
# output/<stage>/ is what every submit_*.sh checks to decide a planner is
# done, so aggregating a pilot would make the real run skip the work. The
# OUT_DIR override keeps pilot shards out of that path as well.
#
# Cells are NOT skipped when their JSON already exists, which is the one
# place this script deviates from its siblings: re-running the pilot means
# re-measuring, and a stale timing is the one thing it must not return.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p$1)"
}

RS_WORKERS=$(read_param "['cluster']['cpu']['rock_sample']['workers']")
FT_WORKERS=$(read_param "['cluster']['cpu']['factored_tiger']['workers']")

# The real templates request up to 16 h so their full shards fit. Every
# pilot cell is projected under 1 h, and a long --time request costs queue
# priority on Snellius, so each cell is capped at 2 h — twice the worst
# projected cell (factored_tiger IDS at 0.93 h, if its information term
# turns out not to prune).
PILOT_TIME="02:00:00"

# stage | job template | planner | seeds | workers
#
# The slate's cost splits into three groups, and each is represented:
#   - the message-passing planners, ~91% of the budget and untouched by
#     both recent PRs. AIF on all four benchmarks; VBP once, since
#     VBP/PSRL/Oracle measured 1.766/1.769/1.770 s/decision on RockSample
#     and are one machine. Both persistent-oracle arms are here because
#     they are new and ~1900 SBU is currently priced on the assumption
#     that they cost what `oracle` costs.
#   - the tree-search family, whose cost moved with the matched depth:
#     BAMCP everywhere, plus BOSS and Factored-BAMCP once each.
#   - IDS, whose information term the 1e-6 state-mass pruning made ~49x
#     cheaper on RockSample. The factor is MEASURED there and assumed
#     everywhere else, so IDS runs on all four benchmarks.
# Skipped as immaterial: belief_q and random, both under 10 SBU per stage
# and unchanged by either PR.
CELLS=(
  "door_code|job_door_code.sh|aif|1|"
  "minigrid_door_code|job_minigrid_door_code.sh|aif|1|"
  "factored_tiger|job_factored_tiger.sh|aif|1|"
  "rock_sample|job_rock_sample.sh|aif|1|"
  "rock_sample|job_rock_sample.sh|vbp|1|"
  "rock_sample|job_rock_sample.sh|persistent_oracle|1|"
  "rock_sample|job_rock_sample.sh|persistent_oracle_vbp|1|"
  "door_code|job_cpu_door_code.sh|bamcp|1|"
  "door_code|job_cpu_door_code.sh|boss|1|"
  "door_code|job_cpu_door_code.sh|ids|1|"
  "minigrid_door_code|job_cpu_minigrid_door_code.sh|bamcp|1|"
  "minigrid_door_code|job_cpu_minigrid_door_code.sh|ids|1|"
  "rock_sample|job_cpu_rock_sample.sh|bamcp|${RS_WORKERS}|${RS_WORKERS}"
  "rock_sample|job_cpu_rock_sample.sh|factored_bamcp|${RS_WORKERS}|${RS_WORKERS}"
  "rock_sample|job_cpu_rock_sample.sh|ids|${RS_WORKERS}|${RS_WORKERS}"
  "factored_tiger|job_cpu_factored_tiger.sh|bamcp|${FT_WORKERS}|${FT_WORKERS}"
  "factored_tiger|job_cpu_factored_tiger.sh|ids|${FT_WORKERS}|${FT_WORKERS}"
)

echo "=== calibration pilot: ${#CELLS[@]} cells, 1 shard each, into output/_pilot/"
JIDS=()
for cell in "${CELLS[@]}"; do
    IFS='|' read -r STAGE TEMPLATE PLANNER SEEDS WORKERS <<< "$cell"
    EXPORTS="ALL,PLANNER=${PLANNER},N_SHARDS=1"
    EXPORTS+=",OUT_DIR=output/_pilot/${STAGE},SEEDS_OVERRIDE=${SEEDS}"
    # Only the templates that declare WORKERS as required env accept it;
    # passing it to the others is harmless but says something false about
    # the run, so it is set only where the template reads it.
    if [ -n "$WORKERS" ]; then
        EXPORTS+=",WORKERS=${WORKERS}"
    fi
    # Same rule for the packed CPU templates: they take a BLOCK of shards
    # rather than one, so they need the shard list and the pack factor. A
    # pilot cell is a single shard, which is block 0 of a 1-shard list.
    case "$TEMPLATE" in
        job_cpu_door_code*.sh|job_cpu_minigrid_door_code.sh)
            EXPORTS+=",SHARD_LIST=0,PACK=1" ;;
    esac
    JID=$(sbatch --parsable \
        --job-name="pilot-${STAGE}-${PLANNER}" \
        --time="${PILOT_TIME}" \
        --array=0 \
        --export="${EXPORTS}" \
        "cluster/${TEMPLATE}")
    JIDS+=("$JID")
    printf '  %-19s %-22s seeds=%-2s %s\n' \
        "$STAGE" "$PLANNER" "$SEEDS" "job ${JID}"
done

echo ""
echo "Monitor:  squeue -u \$USER | grep pilot-"
echo "Billed elapsed, once they finish:"
echo "  sacct -j $(IFS=,; echo "${JIDS[*]}") --format=JobName%30,Elapsed,AllocCPUS,AllocTRES%40,State"
echo "Then price the slate from the measurements:"
echo "  uv run python scripts/price_slate.py --results-dir output/_pilot"
