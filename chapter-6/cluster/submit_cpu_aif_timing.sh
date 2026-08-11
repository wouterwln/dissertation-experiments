#!/usr/bin/env bash
# Submit the matched-hardware AIF timing anchor: one SLURM array over
# seed shards, AIF at the canonical config on the genoa CPU partition.
# Gives the compute table an AIF seconds-per-decision column measured on
# the same hardware family as the CPU-run tree-search baselines, instead
# of an A100 number compared against EPYC cores.
#
# Self-chains its aggregator behind the array, same as the other CPU
# submit scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['aif_timing']['n_shards'])")
MAX_CONCURRENT=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['aif_timing']['max_concurrent'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['aif_cpu_timing']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['aif_cpu_timing']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== aif_cpu_timing (CPU): aif x ${N_SHARDS} shards ==="
OUTPUT="output/aif_cpu_timing/aif.json"
if [ -f "$OUTPUT" ]; then
    echo "  aif: skipped (${OUTPUT} exists)"
    exit 0
fi
MISSING=$(uv run python cluster/_missing_shards.py \
    --shard-dir output/aif_cpu_timing/shards \
    --planner aif \
    --runner door_code \
    --n-shards "${N_SHARDS}" \
    --total-seeds "${TOTAL_SEEDS}" \
    --n-episodes "${N_EPISODES}")
if [ -z "$MISSING" ]; then
    AJID=$(sbatch --parsable \
        --job-name="agg-cpu-aif_timing" \
        --export="ALL,STAGE=aif_cpu_timing,PLANNER=aif" \
        cluster/job_aggregate.sh)
    echo "  aif: all shards present, aggregate ${AJID}"
    exit 0
fi
JID=$(sbatch --parsable \
    --job-name="aiftime-cpu" \
    --export="ALL,N_SHARDS=${N_SHARDS}" \
    --array="${MISSING}%${MAX_CONCURRENT}" \
    cluster/job_cpu_aif_timing.sh)
AJID=$(sbatch --parsable \
    --job-name="agg-cpu-aif_timing" \
    --dependency=afterok:"$JID" \
    --kill-on-invalid-dep=yes \
    --export="ALL,STAGE=aif_cpu_timing,PLANNER=aif" \
    cluster/job_aggregate.sh)
echo "  aif: array ${JID} (missing=${MISSING}), aggregate ${AJID}"

echo ""
echo "Monitor:  squeue -u \$USER"
