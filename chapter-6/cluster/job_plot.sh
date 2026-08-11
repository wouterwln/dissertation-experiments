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

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

# --input-dir/--output cover most plot scripts. The ones below take an
# extra argument that carries meaning the figure loses without it, so it is
# passed here rather than left to the script's default: the depth sweeps
# mark the AIF-MP horizon (the saturation point the sweep runs up to), and
# the Factored Tiger preference figure reports the listen_cost arm, which
# is submitted by the same script and otherwise runs for nothing.
EXTRA=()
case "${STAGE}" in
    door_code_depth)
        EXTRA+=(--horizon "$(read_param "['runner']['horizon']")")
        ;;
    rock_sample_depth)
        EXTRA+=(--horizon "$(read_param "['rock_sample']['runner']['horizon']")")
        ;;
    factored_tiger_preference)
        ARM="output/factored_tiger_listen_cost/aif.json"
        if [ -f "$ARM" ]; then
            EXTRA+=(--listen-cost-json "$ARM")
        else
            echo "WARNING: ${ARM} absent; figure omits the listen_cost note" >&2
        fi
        ;;
esac

echo "host=$(hostname) date=$(date -Is) plot stage=${STAGE} extra=${EXTRA[*]-}"

uv run python "scripts/plot_${STAGE}.py" \
    --input-dir "${OUT_DIR}" \
    --output "${OUT_DIR}/figure.pdf" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "done plot stage=${STAGE} at $(date -Is)"
