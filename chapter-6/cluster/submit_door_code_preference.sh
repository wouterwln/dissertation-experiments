#!/usr/bin/env bash
# Submit one SLURM array job per (success_logit, failure_logit) cell for
# the door_code goal-prior preference ablation. Cells come from
# params.yaml's door_code_preference.{success_logits, failure_logits};
# planner is AIF only (BAMCP/POMCP/BOSS/IDS/Belief-Q/Random ignore the
# goal prior C; PSRL and Oracle do read it via VBP but are excluded to
# bound GPU cost).
# Aggregation is a dependent job per cell (see cluster/job_aggregate.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

SUCC_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(s) for s in yaml.safe_load(open('params.yaml'))['door_code_preference']['success_logits']))")
FAIL_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(f) for f in yaml.safe_load(open('params.yaml'))['door_code_preference']['failure_logits']))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['door_code_preference']['n_shards'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['door_code_preference']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== door_code_preference: succ in {${SUCC_LIST}} x fail in {${FAIL_LIST}} x ${N_SHARDS} shards (aif only) ==="
for SUCC in $SUCC_LIST; do
    for FAIL in $FAIL_LIST; do
        CELL="succ${SUCC}_fail${FAIL}"
        OUTPUT="output/door_code_preference/${CELL}/aif.json"
        if [ -f "$OUTPUT" ]; then
            echo "  ${CELL}: skipped (${OUTPUT} exists)"
            continue
        fi
        MISSING=$(uv run python cluster/_missing_shards.py \
            --shard-dir "output/door_code_preference/${CELL}/shards" \
            --planner aif \
            --runner door_code \
            --n-shards "${N_SHARDS}" \
            --total-seeds "${TOTAL_SEEDS}" \
            --n-episodes "${N_EPISODES}")
        if [ -z "$MISSING" ]; then
            AJID=$(sbatch --parsable \
                --job-name="agg-dcpref-${CELL}" \
                --export="ALL,STAGE=door_code_preference,PLANNER=aif,CELL=${CELL}" \
                cluster/job_aggregate.sh)
            echo "  ${CELL}: all shards present, aggregate ${AJID}"
            continue
        fi
        JID=$(sbatch --parsable \
            --job-name="dcpref-${CELL}" \
            --export="ALL,SUCC=${SUCC},FAIL=${FAIL},N_SHARDS=${N_SHARDS}" \
            --array="${MISSING}" \
            cluster/job_door_code_preference.sh)
        AJID=$(sbatch --parsable \
            --job-name="agg-dcpref-${CELL}" \
            --dependency=afterok:"$JID" \
            --kill-on-invalid-dep=yes \
            --export="ALL,STAGE=door_code_preference,PLANNER=aif,CELL=${CELL}" \
            cluster/job_aggregate.sh)
        echo "  ${CELL}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
    done
done

echo ""
echo "Monitor:  squeue -u \$USER"
