#!/usr/bin/env bash
# Submit (K, planner) × shards SLURM array jobs for the
# rock_sample_kscaling sweep. K values come from params.yaml's
# rock_sample.k_sweep list.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

GPU_PLANNERS=(aif vbp psrl oracle)
K_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(k) for k in yaml.safe_load(open('params.yaml'))['rock_sample']['k_sweep']))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['rock_sample_kscaling']['n_shards'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample']['runner']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== rock_sample_kscaling: K in {${K_LIST}} x ${#GPU_PLANNERS[@]} planners x ${N_SHARDS} shards ==="
for K in $K_LIST; do
    for planner in "${GPU_PLANNERS[@]}"; do
        OUTPUT="output/rock_sample_kscaling/K${K}/${planner}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  K=${K} ${planner}: skipped (${OUTPUT} exists)"
            continue
        fi
        MISSING=$(uv run python cluster/_missing_shards.py \
            --shard-dir "output/rock_sample_kscaling/K${K}/shards" \
            --planner "${planner}" \
            --runner rock_sample \
            --n-shards "${N_SHARDS}" \
            --total-seeds "${TOTAL_SEEDS}" \
            --n-episodes "${N_EPISODES}")
        if [ -z "$MISSING" ]; then
            AJID=$(sbatch --parsable \
                --job-name="agg-rsk-K${K}-${planner}" \
                --export="ALL,STAGE=rock_sample_kscaling,PLANNER=${planner},K=${K}" \
                cluster/job_aggregate.sh)
            echo "  K=${K} ${planner}: all shards present, aggregate ${AJID}"
            continue
        fi
        JID=$(sbatch --parsable \
            --job-name="rsk${K}-${planner}" \
            --export="ALL,PLANNER=${planner},K=${K},N_SHARDS=${N_SHARDS}" \
            --array="${MISSING}" \
            cluster/job_rock_sample_kscaling.sh)
        AJID=$(sbatch --parsable \
            --job-name="agg-rsk-K${K}-${planner}" \
            --dependency=afterok:"$JID" \
            --kill-on-invalid-dep=yes \
            --export="ALL,STAGE=rock_sample_kscaling,PLANNER=${planner},K=${K}" \
            cluster/job_aggregate.sh)
        echo "  K=${K} ${planner}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
    done
done

echo ""
echo "Monitor:  squeue -u \$USER"
