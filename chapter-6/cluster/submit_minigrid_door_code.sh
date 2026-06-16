#!/usr/bin/env bash
# Submit one SLURM array job per GPU-eligible planner for the
# minigrid_door_code stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

GPU_PLANNERS=(aif vbp psrl oracle)
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['minigrid_door_code']['n_shards'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['minigrid']['runner']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['minigrid']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== minigrid_door_code: ${#GPU_PLANNERS[@]} planners x ${N_SHARDS} shards ==="
for planner in "${GPU_PLANNERS[@]}"; do
    OUTPUT="output/minigrid_door_code/${planner}.json"
    if [ -f "$OUTPUT" ]; then
        echo "  ${planner}: skipped (${OUTPUT} exists)"
        continue
    fi
    MISSING=$(uv run python cluster/_missing_shards.py \
        --shard-dir output/minigrid_door_code/shards \
        --planner "${planner}" \
        --runner door_code \
        --n-shards "${N_SHARDS}" \
        --total-seeds "${TOTAL_SEEDS}" \
        --n-episodes "${N_EPISODES}")
    if [ -z "$MISSING" ]; then
        AJID=$(sbatch --parsable \
            --job-name="agg-minigrid_door_code-${planner}" \
            --export="ALL,STAGE=minigrid_door_code,PLANNER=${planner}" \
            cluster/job_aggregate.sh)
        echo "  ${planner}: all shards present, aggregate ${AJID}"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="mg-${planner}" \
        --export="ALL,PLANNER=${planner},N_SHARDS=${N_SHARDS}" \
        --array="${MISSING}" \
        cluster/job_minigrid_door_code.sh)
    AJID=$(sbatch --parsable \
        --job-name="agg-minigrid_door_code-${planner}" \
        --dependency=afterok:"$JID" \
        --kill-on-invalid-dep=yes \
        --export="ALL,STAGE=minigrid_door_code,PLANNER=${planner}" \
        cluster/job_aggregate.sh)
    echo "  ${planner}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
done

echo ""
echo "Monitor:  squeue -u \$USER"
