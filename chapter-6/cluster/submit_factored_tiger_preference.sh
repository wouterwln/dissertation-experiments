#!/usr/bin/env bash
# Submit one SLURM array job per (good_logit, bad_logit) cell for the
# Factored Tiger goal-prior preference ablation. Cells come from
# params.yaml's factored_tiger_preference.{good_logits, bad_logits};
# planner is AIF only (BAMCP/Factored-BAMCP/BOSS/IDS/Random plan on the
# env reward table and never read the goal prior C; PSRL and Oracle do
# read it via VBP but are excluded to bound GPU cost). Aggregation is a
# dependent job per cell (see cluster/job_aggregate.sh). The Katt-2019
# listen_cost arm (one extra cell, own output directory) is submitted at the
# end, reusing the same job script and shard count.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

GOOD_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(g) for g in yaml.safe_load(open('params.yaml'))['factored_tiger_preference']['good_logits']))")
BAD_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(b) for b in yaml.safe_load(open('params.yaml'))['factored_tiger_preference']['bad_logits']))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['factored_tiger_preference']['n_shards'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['factored_tiger_preference']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['factored_tiger']['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== factored_tiger_preference: good in {${GOOD_LIST}} x bad in {${BAD_LIST}} x ${N_SHARDS} shards (aif only) ==="
for GOOD in $GOOD_LIST; do
    for BAD in $BAD_LIST; do
        CELL="good${GOOD}_bad${BAD}"
        OUTPUT="output/factored_tiger_preference/${CELL}/aif.json"
        if [ -f "$OUTPUT" ]; then
            echo "  ${CELL}: skipped (${OUTPUT} exists)"
            continue
        fi
        MISSING=$(uv run python cluster/_missing_shards.py \
            --shard-dir "output/factored_tiger_preference/${CELL}/shards" \
            --planner aif \
            --runner factored_tiger \
            --n-shards "${N_SHARDS}" \
            --total-seeds "${TOTAL_SEEDS}" \
            --n-episodes "${N_EPISODES}")
        if [ -z "$MISSING" ]; then
            AJID=$(sbatch --parsable \
                --job-name="agg-ftpref-${CELL}" \
                --export="ALL,STAGE=factored_tiger_preference,PLANNER=aif,CELL=${CELL}" \
                cluster/job_aggregate.sh)
            echo "  ${CELL}: all shards present, aggregate ${AJID}"
            continue
        fi
        JID=$(sbatch --parsable \
            --job-name="ftpref-${CELL}" \
            --export="ALL,GOOD=${GOOD},BAD=${BAD},N_SHARDS=${N_SHARDS}" \
            --array="${MISSING}" \
            cluster/job_factored_tiger_preference.sh)
        AJID=$(sbatch --parsable \
            --job-name="agg-ftpref-${CELL}" \
            --dependency=afterok:"$JID" \
            --kill-on-invalid-dep=yes \
            --export="ALL,STAGE=factored_tiger_preference,PLANNER=aif,CELL=${CELL}" \
            cluster/job_aggregate.sh)
        echo "  ${CELL}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
    done
done

# Katt-2019 listen_cost arm (DVC stage factored_tiger_listen_cost): the same
# job script at the canonical goal prior with the environment charging per
# LISTEN, into its own output directory. One extra cell — listen_cost is the
# one reward term a state-only goal prior cannot encode, so it is measured
# rather than swept.
LISTEN_COST=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['factored_tiger_preference']['listen_cost_arm'])")
CANON_GOOD=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['factored_tiger']['env']['good_logit'])")
CANON_BAD=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['factored_tiger']['env']['bad_logit'])")
ARM_DIR="output/factored_tiger_listen_cost"

echo ""
echo "=== factored_tiger_listen_cost: listen_cost=${LISTEN_COST} at (${CANON_GOOD}, ${CANON_BAD}) x ${N_SHARDS} shards (aif only) ==="
if [ -f "${ARM_DIR}/aif.json" ]; then
    echo "  skipped (${ARM_DIR}/aif.json exists)"
else
    MISSING=$(uv run python cluster/_missing_shards.py \
        --shard-dir "${ARM_DIR}/shards" \
        --planner aif \
        --runner factored_tiger \
        --n-shards "${N_SHARDS}" \
        --total-seeds "${TOTAL_SEEDS}" \
        --n-episodes "${N_EPISODES}")
    if [ -z "$MISSING" ]; then
        AJID=$(sbatch --parsable \
            --job-name="agg-ftlisten" \
            --export="ALL,STAGE=factored_tiger_listen_cost,PLANNER=aif" \
            cluster/job_aggregate.sh)
        echo "  all shards present, aggregate ${AJID}"
    else
        JID=$(sbatch --parsable \
            --job-name="ftlisten" \
            --export="ALL,GOOD=${CANON_GOOD},BAD=${CANON_BAD},LISTEN_COST=${LISTEN_COST},OUT_DIR=${ARM_DIR},N_SHARDS=${N_SHARDS}" \
            --array="${MISSING}" \
            cluster/job_factored_tiger_preference.sh)
        AJID=$(sbatch --parsable \
            --job-name="agg-ftlisten" \
            --dependency=afterok:"$JID" \
            --kill-on-invalid-dep=yes \
            --export="ALL,STAGE=factored_tiger_listen_cost,PLANNER=aif" \
            cluster/job_aggregate.sh)
        echo "  array ${JID} (missing=${MISSING}), aggregate ${AJID}"
    fi
fi

echo ""
echo "Monitor:  squeue -u \$USER"
