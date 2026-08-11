#!/usr/bin/env bash
# Submit one SLURM array job per depth cell for the rock_sample MCTS
# depth-sensitivity ablation on CPU. The slate is BAMCP only: BOSS and
# Factored-BAMCP take their action values from the same MCTS machinery so
# they would trace the same curve, and IDS is not swept over depth at all —
# it costs ~110x more per seed here and its cost is depth-invariant, so
# every cell would buy a point on a line profiling already shows is flat
# (see the params.yaml rock_sample_depth block). Mirrors dvc.yaml, which
# likewise defines depth cells for BAMCP only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

DEPTH_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(d) for d in yaml.safe_load(open('params.yaml'))['rock_sample_depth']['depth_sweep']))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['rock_sample_depth']['n_shards'])")
WORKERS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['rock_sample_depth']['workers'])")
MAX_CONCURRENT=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['rock_sample_depth']['max_concurrent'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample_depth']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

# BAMCP only. IDS cells are not submitted here and not defined in dvc.yaml:
# even the two endpoint depths would cost ~3043 SBU, 46% of the genoa budget
# remaining for this run, to buy markers on a line already known to be flat.
# IDS's depth-invariance is reported from profiling instead (8.07 h -> 8.18 h
# across this sweep); see the params.yaml rock_sample_depth block.
CELLS=()
for DEPTH in $DEPTH_LIST; do
    CELLS+=("${DEPTH}:bamcp")
done

echo "=== rock_sample_depth (CPU): ${#CELLS[@]} cells (bamcp at {${DEPTH_LIST}}; ids not swept over depth) x ${N_SHARDS} shards x ${WORKERS} workers ==="
for cell in "${CELLS[@]}"; do
    DEPTH="${cell%%:*}"
    planner="${cell##*:}"
    OUTPUT="output/rock_sample_depth/depth${DEPTH}/${planner}.json"
    if [ -f "$OUTPUT" ]; then
        echo "  depth=${DEPTH} ${planner}: skipped (${OUTPUT} exists)"
        continue
    fi
    MISSING=$(uv run python cluster/_missing_shards.py \
        --shard-dir "output/rock_sample_depth/depth${DEPTH}/shards" \
        --planner "${planner}" \
        --runner rock_sample \
        --n-shards "${N_SHARDS}" \
        --total-seeds "${TOTAL_SEEDS}" \
        --n-episodes "${N_EPISODES}")
    if [ -z "$MISSING" ]; then
        AJID=$(sbatch --parsable \
            --job-name="agg-cpu-rsd-d${DEPTH}-${planner}" \
            --export="ALL,STAGE=rock_sample_depth,PLANNER=${planner},DEPTH=${DEPTH}" \
            cluster/job_aggregate.sh)
        echo "  depth=${DEPTH} ${planner}: all shards present, aggregate ${AJID}"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="rsd-cpu-d${DEPTH}-${planner}" \
        --export="ALL,PLANNER=${planner},DEPTH=${DEPTH},N_SHARDS=${N_SHARDS},WORKERS=${WORKERS}" \
        --array="${MISSING}%${MAX_CONCURRENT}" \
        cluster/job_cpu_rock_sample_depth.sh)
    AJID=$(sbatch --parsable \
        --job-name="agg-cpu-rsd-d${DEPTH}-${planner}" \
        --dependency=afterok:"$JID" \
        --kill-on-invalid-dep=yes \
        --export="ALL,STAGE=rock_sample_depth,PLANNER=${planner},DEPTH=${DEPTH}" \
        cluster/job_aggregate.sh)
    echo "  depth=${DEPTH} ${planner}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
done

echo ""
echo "Monitor:  squeue -u \$USER"
