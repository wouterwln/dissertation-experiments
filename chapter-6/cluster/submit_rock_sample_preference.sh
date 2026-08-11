#!/usr/bin/env bash
# Submit one SLURM array job per (good_logit, bad_logit) cell for the
# RockSample goal-prior preference ablation. Cells come from params.yaml's
# rock_sample_preference.cells — a 6-cell cross through the canonical
# (2, 4) cell, enumerated rather than crossed, so this loops the list
# once instead of double-looping two axes; planner is AIF only
# (BAMCP/Factored-BAMCP/BOSS/IDS/Random plan on the env reward table and
# never read the goal prior C; PSRL and Oracle do read it via VBP but are
# excluded to bound GPU cost). Aggregation is a dependent job per cell
# (see cluster/job_aggregate.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

# One "good:bad" token per cell.
CELL_LIST=$(uv run python -c \
    "import yaml; print(' '.join(f\"{c['good']}:{c['bad']}\" for c in yaml.safe_load(open('params.yaml'))['rock_sample_preference']['cells'].values()))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['rock_sample_preference']['n_shards'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample_preference']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['rock_sample']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== rock_sample_preference: cells {${CELL_LIST}} x ${N_SHARDS} shards (aif only) ==="
for PAIR in $CELL_LIST; do
    GOOD="${PAIR%%:*}"
    BAD="${PAIR##*:}"
    CELL="good${GOOD}_bad${BAD}"
    OUTPUT="output/rock_sample_preference/${CELL}/aif.json"
    if [ -f "$OUTPUT" ]; then
        echo "  ${CELL}: skipped (${OUTPUT} exists)"
        continue
    fi
    MISSING=$(uv run python cluster/_missing_shards.py \
        --shard-dir "output/rock_sample_preference/${CELL}/shards" \
        --planner aif \
        --runner rock_sample \
        --n-shards "${N_SHARDS}" \
        --total-seeds "${TOTAL_SEEDS}" \
        --n-episodes "${N_EPISODES}")
    if [ -z "$MISSING" ]; then
        AJID=$(sbatch --parsable \
            --job-name="agg-rspref-${CELL}" \
            --export="ALL,STAGE=rock_sample_preference,PLANNER=aif,CELL=${CELL}" \
            cluster/job_aggregate.sh)
        echo "  ${CELL}: all shards present, aggregate ${AJID}"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="rspref-${CELL}" \
        --export="ALL,GOOD=${GOOD},BAD=${BAD},N_SHARDS=${N_SHARDS}" \
        --array="${MISSING}" \
        cluster/job_rock_sample_preference.sh)
    AJID=$(sbatch --parsable \
        --job-name="agg-rspref-${CELL}" \
        --dependency=afterok:"$JID" \
        --kill-on-invalid-dep=yes \
        --export="ALL,STAGE=rock_sample_preference,PLANNER=aif,CELL=${CELL}" \
        cluster/job_aggregate.sh)
    echo "  ${CELL}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
done

echo ""
echo "Monitor:  squeue -u \$USER"
