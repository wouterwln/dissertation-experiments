#!/usr/bin/env bash
# Submit one SLURM array job per (iterations, damping) cell of the AIF-MP
# convergence sensitivity report. The grid comes from params.yaml's
# aif_convergence.{iterations_sweep, damping_sweep}; planner is AIF only
# (the sample-based baselines have no message-passing fixed point to
# sweep). Aggregation is a dependent job per cell (cluster/job_aggregate.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

ITERS_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(v) for v in yaml.safe_load(open('params.yaml'))['aif_convergence']['iterations_sweep']))")
DAMP_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(v) for v in yaml.safe_load(open('params.yaml'))['aif_convergence']['damping_sweep']))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['aif_convergence']['n_shards'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['aif_convergence']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== aif_convergence: iters in {${ITERS_LIST}} x damping in {${DAMP_LIST}} x ${N_SHARDS} shards ==="
for ITERS in $ITERS_LIST; do
    for DAMP in $DAMP_LIST; do
        CELL="iter${ITERS}_damp${DAMP}"
        OUTPUT="output/aif_convergence/${CELL}/aif.json"
        if [ -f "$OUTPUT" ]; then
            echo "  ${CELL}: skipped (${OUTPUT} exists)"
            continue
        fi
        MISSING=$(uv run python cluster/_missing_shards.py \
            --shard-dir "output/aif_convergence/${CELL}/shards" \
            --planner aif \
            --runner door_code \
            --n-shards "${N_SHARDS}" \
            --total-seeds "${TOTAL_SEEDS}" \
            --n-episodes "${N_EPISODES}")
        if [ -z "$MISSING" ]; then
            AJID=$(sbatch --parsable \
                --job-name="agg-aifconv-${CELL}" \
                --export="ALL,STAGE=aif_convergence,PLANNER=aif,ITERS=${ITERS},DAMP=${DAMP}" \
                cluster/job_aggregate.sh)
            echo "  ${CELL}: all shards present, aggregate ${AJID}"
            continue
        fi
        JID=$(sbatch --parsable \
            --job-name="aifconv-${CELL}" \
            --export="ALL,ITERS=${ITERS},DAMP=${DAMP},N_SHARDS=${N_SHARDS}" \
            --array="${MISSING}" \
            cluster/job_aif_convergence.sh)
        AJID=$(sbatch --parsable \
            --job-name="agg-aifconv-${CELL}" \
            --dependency=afterok:"$JID" \
            --kill-on-invalid-dep=yes \
            --export="ALL,STAGE=aif_convergence,PLANNER=aif,ITERS=${ITERS},DAMP=${DAMP}" \
            cluster/job_aggregate.sh)
        echo "  ${CELL}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
    done
done

echo ""
echo "Monitor:  squeue -u \$USER"
