#!/usr/bin/env bash
# Submit one SLURM array job per CPU-only planner for the
# factored_tiger stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

CPU_PLANNERS=(bamcp boss factored_bamcp ids random)
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['factored_tiger']['n_shards'])")
WORKERS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['factored_tiger']['workers'])")
MAX_CONCURRENT=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['factored_tiger']['max_concurrent'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['factored_tiger']['runner']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['factored_tiger']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== factored_tiger (CPU): ${#CPU_PLANNERS[@]} planners x ${N_SHARDS} shards x ${WORKERS} workers ==="
for planner in "${CPU_PLANNERS[@]}"; do
    OUTPUT="output/factored_tiger/${planner}.json"
    if [ -f "$OUTPUT" ]; then
        echo "  ${planner}: skipped (${OUTPUT} exists)"
        continue
    fi
    MISSING=$(uv run python cluster/_missing_shards.py \
        --shard-dir output/factored_tiger/shards \
        --planner "${planner}" \
        --runner factored_tiger \
        --n-shards "${N_SHARDS}" \
        --total-seeds "${TOTAL_SEEDS}" \
        --n-episodes "${N_EPISODES}")
    if [ -z "$MISSING" ]; then
        AJID=$(sbatch --parsable \
            --job-name="agg-cpu-factored_tiger-${planner}" \
            --export="ALL,STAGE=factored_tiger,PLANNER=${planner}" \
            cluster/job_aggregate.sh)
        echo "  ${planner}: all shards present, aggregate ${AJID}"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="ft-cpu-${planner}" \
        --export="ALL,PLANNER=${planner},N_SHARDS=${N_SHARDS},WORKERS=${WORKERS}" \
        --array="${MISSING}%${MAX_CONCURRENT}" \
        cluster/job_cpu_factored_tiger.sh)
    AJID=$(sbatch --parsable \
        --job-name="agg-cpu-factored_tiger-${planner}" \
        --dependency=afterok:"$JID" \
        --kill-on-invalid-dep=yes \
        --export="ALL,STAGE=factored_tiger,PLANNER=${planner}" \
        cluster/job_aggregate.sh)
    echo "  ${planner}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
done

echo ""
echo "Monitor:  squeue -u \$USER"
