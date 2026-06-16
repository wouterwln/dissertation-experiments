#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-minigrid
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=3
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/cpu_minigrid_%x_%A_%a.out
#SBATCH --error=logs/cpu_minigrid_%x_%A_%a.err
# Required env: PLANNER, N_SHARDS
#
# CPU sibling of job_minigrid_door_code.sh. Single-seed shards on one
# Genoa core; the minigrid runner doesn't accept --workers.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output/minigrid_door_code/shards

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

PLANNER="${PLANNER:?PLANNER not set}"
N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['minigrid']['runner']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=${PLANNER} shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_minigrid_door_code.py \
    --planner "$PLANNER" \
    --K "$(read_param "['minigrid']['env']['K']")" \
    --lock-length "$(read_param "['minigrid']['env']['lock_length']")" \
    --probe-noise "$(read_param "['minigrid']['env']['probe_noise']")" \
    --grid-size "$(read_param "['minigrid']['env']['grid_size']")" \
    --slip-prob "$(read_param "['minigrid']['env']['slip_prob']")" \
    --agent-view-size "$(read_param "['minigrid']['env']['agent_view_size']")" \
    --n-episodes "$(read_param "['minigrid']['runner']['n_episodes']")" \
    --max-steps "$(read_param "['minigrid']['runner']['max_steps']")" \
    --horizon "$(read_param "['minigrid']['runner']['horizon']")" \
    --seed "$(read_param "['minigrid']['runner']['seed']")" \
    --learn-on-failure "$(read_param "['minigrid']['runner']['learn_on_failure']")" \
    --clip-horizon "$(read_param "['minigrid']['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['minigrid']['planner']['iterations']")" \
    --planner-damping "$(read_param "['minigrid']['planner']['damping']")" \
    --q-lr "$(read_param "['minigrid']['planner']['q_lr']")" \
    --q-epsilon "$(read_param "['minigrid']['planner']['q_epsilon']")" \
    --q-gamma "$(read_param "['minigrid']['planner']['q_gamma']")" \
    --boss-n-samples "$(read_param "['minigrid']['planner']['boss_n_samples']")" \
    --bamcp-n-simulations "$(read_param "['minigrid']['planner']['bamcp_n_simulations']")" \
    --bamcp-depth "$(read_param "['minigrid']['planner']['bamcp_depth']")" \
    --bamcp-ucb-c "$(read_param "['minigrid']['planner']['bamcp_ucb_c']")" \
    --bamcp-gamma "$(read_param "['minigrid']['planner']['bamcp_gamma']")" \
    --ids-bamcp-n-simulations "$(read_param "['minigrid']['planner']['ids_bamcp_n_simulations']")" \
    --ids-epsilon-info "$(read_param "['minigrid']['planner']['ids_epsilon_info']")" \
    --video-dir "output/minigrid_door_code/${PLANNER}/videos" \
    --video-fps "$(read_param "['minigrid']['runner']['video_fps']")" \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/minigrid_door_code/shards/${PLANNER}.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=${PLANNER} shard=${SHARD_INDEX} at $(date -Is)"
