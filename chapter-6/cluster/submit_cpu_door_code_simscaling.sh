#!/usr/bin/env bash
# Submit one SLURM array job per simulation budget for the door_code
# BAMCP sims-budget ablation on CPU. Budgets come from params.yaml's
# door_code_simscaling.sims_sweep list; planner is BAMCP only (BOSS,
# IDS and POMCP inherit the same MCTS machinery). Aggregation is a
# dependent job per budget (see cluster/job_aggregate.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

SIMS_LIST=$(uv run python -c \
    "import yaml; print(' '.join(str(s) for s in yaml.safe_load(open('params.yaml'))['door_code_simscaling']['sims_sweep']))")
N_SHARDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['door_code_simscaling']['n_shards'])")
MAX_CONCURRENT=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['cluster']['cpu']['door_code_simscaling']['max_concurrent'])")
TOTAL_SEEDS=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['door_code_simscaling']['n_seeds'])")
N_EPISODES=$(uv run python -c \
    "import yaml; print(yaml.safe_load(open('params.yaml'))['runner']['n_episodes'])")
if [ "$N_SHARDS" -gt "$TOTAL_SEEDS" ]; then
    echo "  clamping n_shards ${N_SHARDS} -> ${TOTAL_SEEDS} (only ${TOTAL_SEEDS} seeds)"
    N_SHARDS="$TOTAL_SEEDS"
fi

echo "=== door_code_simscaling (CPU): sims in {${SIMS_LIST}} x ${N_SHARDS} shards ==="
for SIMS in $SIMS_LIST; do
    OUTPUT="output/door_code_simscaling/sims${SIMS}/bamcp.json"
    if [ -f "$OUTPUT" ]; then
        echo "  sims=${SIMS}: skipped (${OUTPUT} exists)"
        continue
    fi
    MISSING=$(uv run python cluster/_missing_shards.py \
        --shard-dir "output/door_code_simscaling/sims${SIMS}/shards" \
        --planner bamcp \
        --runner door_code \
        --n-shards "${N_SHARDS}" \
        --total-seeds "${TOTAL_SEEDS}" \
        --n-episodes "${N_EPISODES}")
    if [ -z "$MISSING" ]; then
        AJID=$(sbatch --parsable \
            --job-name="agg-cpu-dcss-sims${SIMS}" \
            --export="ALL,STAGE=door_code_simscaling,PLANNER=bamcp,SIMS=${SIMS}" \
            cluster/job_aggregate.sh)
        echo "  sims=${SIMS}: all shards present, aggregate ${AJID}"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="dcss-cpu-sims${SIMS}" \
        --export="ALL,SIMS=${SIMS},N_SHARDS=${N_SHARDS}" \
        --array="${MISSING}%${MAX_CONCURRENT}" \
        cluster/job_cpu_door_code_simscaling.sh)
    AJID=$(sbatch --parsable \
        --job-name="agg-cpu-dcss-sims${SIMS}" \
        --dependency=afterok:"$JID" \
        --kill-on-invalid-dep=yes \
        --export="ALL,STAGE=door_code_simscaling,PLANNER=bamcp,SIMS=${SIMS}" \
        cluster/job_aggregate.sh)
    echo "  sims=${SIMS}: array ${JID} (missing=${MISSING}), aggregate ${AJID}"
done

echo ""
echo "Monitor:  squeue -u \$USER"
