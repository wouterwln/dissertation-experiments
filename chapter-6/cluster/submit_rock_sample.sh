#!/usr/bin/env bash
# Submit one SLURM array job per GPU-eligible planner for the rock_sample
# stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

# persistent_oracle{,_vbp} are rock_sample-only (they need both persistent
# and episodic θ components to differ from `oracle`).
GPU_PLANNERS=(aif vbp psrl oracle persistent_oracle persistent_oracle_vbp)
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['rock_sample']['n_shards'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample']['runner']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== rock_sample: ${#GPU_PLANNERS[@]} planners x ${N_SHARDS} shards ==="
for planner in "${GPU_PLANNERS[@]}"; do
    OUTPUT="output/rock_sample/${planner}.json"
    if [ -f "$OUTPUT" ]; then
        echo "  ${planner}: skipped (${OUTPUT} exists)"
        continue
    fi
    MISSING=$(uv run python cluster/_missing_shards.py \
        --shard-dir output/rock_sample/shards \
        --planner "${planner}" \
        --runner rock_sample \
        --n-shards "${N_SHARDS}" \
        --total-seeds "${TOTAL_SEEDS}" \
        --n-episodes "${N_EPISODES}")
    if [ -z "$MISSING" ]; then
        AJID=$(sbatch --parsable \
            --job-name="agg-rock_sample-${planner}" \
            --export="ALL,STAGE=rock_sample,PLANNER=${planner}" \
            cluster/job_aggregate.sh)
        echo "  ${planner}: all shards present, aggregate ${AJID}"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="rs-${planner}" \
        --export="ALL,PLANNER=${planner},N_SHARDS=${N_SHARDS}" \
        --array="${MISSING}" \
        cluster/job_rock_sample.sh)
    AJID=$(sbatch --parsable \
        --job-name="agg-rock_sample-${planner}" \
        --dependency=afterok:"$JID" \
        --kill-on-invalid-dep=yes \
        --export="ALL,STAGE=rock_sample,PLANNER=${planner}" \
        cluster/job_aggregate.sh)
    echo "  ${planner}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
done

echo ""
echo "Monitor:  squeue -u \$USER"
