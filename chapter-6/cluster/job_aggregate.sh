#!/usr/bin/env bash
#SBATCH --job-name=lp-aggregate
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/aggregate_%x_%A.out
#SBATCH --error=logs/aggregate_%x_%A.err
# Required env: STAGE, PLANNER. For rock_sample_kscaling, also K.
# For door_code_simscaling, also SIMS. For aif_convergence, also ITERS
# and DAMP. For door_code_depth / rock_sample_depth, also DEPTH. For
# door_code_rollout, also POLICY. For the *_preference sweeps, also CELL
# (the good<G>_bad<B> / succ<S>_fail<F> subdirectory name).
#
# Tiny postprocess job. One instance is submitted per (planner [, K])
# combination by the submit scripts, with --dependency=afterok on the
# matching array job ID. afterok means: only run if the entire array
# succeeded — partial failures cancel just this planner's aggregate
# (state: DependencyNeverSatisfied), never overwrite a per-planner JSON
# with partial shards. Aggregating only the submitted planner also
# guarantees we never touch the JSON of a planner that was skipped by
# the submit script.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source cluster/setup_env_cpu.sh

STAGE="${STAGE:?STAGE not set}"
PLANNER="${PLANNER:?PLANNER not set}"

if [ "$STAGE" = "rock_sample_kscaling" ]; then
    K_VAL="${K:?K not set for rock_sample_kscaling}"
    OUT_DIR="output/rock_sample_kscaling/K${K_VAL}"
    LABEL="K=${K_VAL} ${PLANNER}"
elif [ "$STAGE" = "door_code_simscaling" ]; then
    SIMS_VAL="${SIMS:?SIMS not set for door_code_simscaling}"
    OUT_DIR="output/door_code_simscaling/sims${SIMS_VAL}"
    LABEL="sims=${SIMS_VAL} ${PLANNER}"
elif [ "$STAGE" = "aif_convergence" ]; then
    ITERS_VAL="${ITERS:?ITERS not set for aif_convergence}"
    DAMP_VAL="${DAMP:?DAMP not set for aif_convergence}"
    OUT_DIR="output/aif_convergence/iter${ITERS_VAL}_damp${DAMP_VAL}"
    LABEL="iters=${ITERS_VAL} damping=${DAMP_VAL} ${PLANNER}"
elif [ "$STAGE" = "door_code_depth" ] || [ "$STAGE" = "rock_sample_depth" ]; then
    DEPTH_VAL="${DEPTH:?DEPTH not set for ${STAGE}}"
    OUT_DIR="output/${STAGE}/depth${DEPTH_VAL}"
    LABEL="depth=${DEPTH_VAL} ${PLANNER}"
elif [ "$STAGE" = "door_code_rollout" ]; then
    POLICY_VAL="${POLICY:?POLICY not set for door_code_rollout}"
    OUT_DIR="output/door_code_rollout/${POLICY_VAL}"
    LABEL="policy=${POLICY_VAL} ${PLANNER}"
elif [ "$STAGE" = "door_code_preference" ] \
    || [ "$STAGE" = "rock_sample_preference" ] \
    || [ "$STAGE" = "factored_tiger_preference" ]; then
    # The preference sweeps are 2-D, so the cell directory name is passed
    # verbatim rather than reassembled from its two axes here.
    CELL_VAL="${CELL:?CELL not set for ${STAGE}}"
    OUT_DIR="output/${STAGE}/${CELL_VAL}"
    LABEL="${CELL_VAL} ${PLANNER}"
else
    OUT_DIR="output/${STAGE}"
    LABEL="${PLANNER}"
fi

echo "host=$(hostname) date=$(date -Is) aggregate stage=${STAGE} ${LABEL}"

uv run python scripts/aggregate_shards.py \
    --input-glob "${OUT_DIR}/shards/${PLANNER}.shard*.json" \
    --output "${OUT_DIR}/${PLANNER}.json"

echo "done aggregate stage=${STAGE} ${LABEL} at $(date -Is)"
