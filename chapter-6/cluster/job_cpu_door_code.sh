#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-door-code
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cpu_door_code_%x_%A_%a.out
#SBATCH --error=logs/cpu_door_code_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS
# Array bounds are passed by the submit script: --array=0-(N_SHARDS-1)
#
# CPU sibling of job_door_code.sh. Each shard runs ceil(n_seeds/N_SHARDS)
# seeds of one CPU planner (bamcp/boss/pomcp/ids/belief_q) on a single
# Genoa core; the runner here doesn't accept --workers.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output/door_code/shards

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['runner']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_door_code.py \
    --planner "$PLANNER" \
    --K "$(read_param "['env']['K']")" \
    --lock-length "$(read_param "['env']['lock_length']")" \
    --probe-noise "$(read_param "['env']['probe_noise']")" \
    --n-episodes "$(read_param "['runner']['n_episodes']")" \
    --max-steps "$(read_param "['runner']['max_steps']")" \
    --horizon "$(read_param "['runner']['horizon']")" \
    --seed "$(read_param "['runner']['seed']")" \
    --learn-on-failure "$(read_param "['runner']['learn_on_failure']")" \
    --clip-horizon "$(read_param "['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['planner']['iterations']")" \
    --planner-damping "$(read_param "['planner']['damping']")" \
    --q-lr "$(read_param "['planner']['q_lr']")" \
    --q-epsilon "$(read_param "['planner']['q_epsilon']")" \
    --q-gamma "$(read_param "['planner']['q_gamma']")" \
    --boss-n-samples "$(read_param "['planner']['boss_n_samples']")" \
    --bamcp-n-simulations "$(read_param "['planner']['bamcp_n_simulations']")" \
    --bamcp-depth "$(read_param "['planner']['bamcp_depth']")" \
    --bamcp-ucb-c "$(read_param "['planner']['bamcp_ucb_c']")" \
    --bamcp-gamma "$(read_param "['planner']['bamcp_gamma']")" \
    --ids-bamcp-n-simulations "$(read_param "['planner']['ids_bamcp_n_simulations']")" \
    --ids-epsilon-info "$(read_param "['planner']['ids_epsilon_info']")" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/door_code/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
