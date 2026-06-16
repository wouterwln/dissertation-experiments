#!/usr/bin/env bash
#SBATCH --job-name=lp-plot
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/plot_%x_%A.out
#SBATCH --error=logs/plot_%x_%A.err
# Required env: STAGE
#
# Runs scripts/plot_<STAGE>.py against output/<STAGE>/, producing
# output/<STAGE>/figure.pdf. Submitted by submit_all.sh with
# --dependency=afterok on the aggregate JIDs for that stage.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source cluster/setup_env_cpu.sh

STAGE="${STAGE:?STAGE not set}"
OUT_DIR="output/${STAGE}"

echo "host=$(hostname) date=$(date -Is) plot stage=${STAGE}"

uv run python "scripts/plot_${STAGE}.py" \
    --input-dir "${OUT_DIR}" \
    --output "${OUT_DIR}/figure.pdf"

echo "done plot stage=${STAGE} at $(date -Is)"
