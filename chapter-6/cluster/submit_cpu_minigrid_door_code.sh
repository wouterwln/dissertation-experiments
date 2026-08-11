#!/usr/bin/env bash
# Submit one SLURM array job per CPU-only planner for the
# minigrid_door_code stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

CPU_PLANNERS=(belief_q boss bamcp pomcp ids random)
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['minigrid_door_code']['n_shards'])")
PACK=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['minigrid_door_code']['pack'])")
MAX_CONCURRENT=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['minigrid_door_code']['max_concurrent'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['minigrid']['runner']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['minigrid']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== minigrid_door_code (CPU): ${#CPU_PLANNERS[@]} planners x ${N_SHARDS} shards ==="
for planner in "${CPU_PLANNERS[@]}"; do
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
            --job-name="agg-cpu-minigrid_door_code-${planner}" \
            --export="ALL,STAGE=minigrid_door_code,PLANNER=${planner}" \
            cluster/job_aggregate.sh)
        echo "  ${planner}: all shards present, aggregate ${AJID}"
        continue
    fi
    # An array task is a BLOCK of PACK shards, so the array is indexed by
    # block and the missing-shard list is passed through for the job to
    # slice. Colon-separated because sbatch --export splits on commas.
    IFS=',' read -ra MISSING_ARR <<< "$MISSING"
    N_BLOCKS=$(( (${#MISSING_ARR[@]} + PACK - 1) / PACK ))
    SHARD_LIST=$(IFS=':'; echo "${MISSING_ARR[*]}")
    JID=$(sbatch --parsable \
        --job-name="mg-cpu-${planner}" \
        --export="ALL,PLANNER=${planner},N_SHARDS=${N_SHARDS},SHARD_LIST=${SHARD_LIST},PACK=${PACK}" \
        --array="0-$((N_BLOCKS - 1))%${MAX_CONCURRENT}" \
        cluster/job_cpu_minigrid_door_code.sh)
    AJID=$(sbatch --parsable \
        --job-name="agg-cpu-minigrid_door_code-${planner}" \
        --dependency=afterok:"$JID" \
        --kill-on-invalid-dep=yes \
        --export="ALL,STAGE=minigrid_door_code,PLANNER=${planner}" \
        cluster/job_aggregate.sh)
    echo "  ${planner}: array ${JID} (${#MISSING_ARR[@]} shards in ${N_BLOCKS}" \
         "blocks of ${PACK}), aggregate ${AJID}"
done

echo ""
echo "Monitor:  squeue -u \$USER"
