#!/usr/bin/env bash
# Submit one SLURM array job per (depth, planner) cell for the door_code
# MCTS depth-sensitivity ablation on CPU. Depths come from params.yaml's
# door_code_depth.depth_sweep list; the planner slate is BAMCP + IDS only
# (BOSS and POMCP take their action values from the same MCTS machinery,
# so they would trace the same curve). Aggregation is a dependent job per
# cell (see cluster/job_aggregate.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

CPU_PLANNERS=(bamcp ids)
DEPTH_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(d) for d in yaml.safe_load(open('params.yaml'))['door_code_depth']['depth_sweep']))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['door_code_depth']['n_shards'])")
PACK=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['door_code_depth']['pack'])")
MAX_CONCURRENT=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['door_code_depth']['max_concurrent'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['door_code_depth']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== door_code_depth (CPU): depth in {${DEPTH_LIST}} x ${#CPU_PLANNERS[@]} planners x ${N_SHARDS} shards ==="
for DEPTH in $DEPTH_LIST; do
    for planner in "${CPU_PLANNERS[@]}"; do
        OUTPUT="output/door_code_depth/depth${DEPTH}/${planner}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  depth=${DEPTH} ${planner}: skipped (${OUTPUT} exists)"
            continue
        fi
        MISSING=$(uv run python cluster/_missing_shards.py \
            --shard-dir "output/door_code_depth/depth${DEPTH}/shards" \
            --planner "${planner}" \
            --runner door_code \
            --n-shards "${N_SHARDS}" \
            --total-seeds "${TOTAL_SEEDS}" \
            --n-episodes "${N_EPISODES}")
        if [ -z "$MISSING" ]; then
            AJID=$(sbatch --parsable \
                --job-name="agg-cpu-dcd-d${DEPTH}-${planner}" \
                --export="ALL,STAGE=door_code_depth,PLANNER=${planner},DEPTH=${DEPTH}" \
                cluster/job_aggregate.sh)
            echo "  depth=${DEPTH} ${planner}: all shards present, aggregate ${AJID}"
            continue
        fi
        # An array task is a BLOCK of PACK shards, so the array is indexed
        # by block and the missing-shard list is passed through for the job
        # to slice. Colon-separated: sbatch --export splits on commas.
        IFS=',' read -ra MISSING_ARR <<< "$MISSING"
        N_BLOCKS=$(( (${#MISSING_ARR[@]} + PACK - 1) / PACK ))
        SHARD_LIST=$(IFS=':'; echo "${MISSING_ARR[*]}")
        JID=$(sbatch --parsable \
            --job-name="dcd-cpu-d${DEPTH}-${planner}" \
            --export="ALL,PLANNER=${planner},DEPTH=${DEPTH},N_SHARDS=${N_SHARDS},SHARD_LIST=${SHARD_LIST},PACK=${PACK}" \
            --array="0-$((N_BLOCKS - 1))%${MAX_CONCURRENT}" \
            cluster/job_cpu_door_code_depth.sh)
        AJID=$(sbatch --parsable \
            --job-name="agg-cpu-dcd-d${DEPTH}-${planner}" \
            --dependency=afterok:"$JID" \
            --kill-on-invalid-dep=yes \
            --export="ALL,STAGE=door_code_depth,PLANNER=${planner},DEPTH=${DEPTH}" \
            cluster/job_aggregate.sh)
        echo "  depth=${DEPTH} ${planner}: array ${JID} (${#MISSING_ARR[@]} shards" \
             "in ${N_BLOCKS} blocks of ${PACK}), aggregate ${AJID}"
    done
done

echo ""
echo "Monitor:  squeue -u \$USER"
