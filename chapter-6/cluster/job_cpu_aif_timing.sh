#!/usr/bin/env bash
#SBATCH --job-name=lp-cpu-aif-timing
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpu_aif_timing_%x_%A_%a.out
#SBATCH --error=logs/cpu_aif_timing_%x_%A_%a.err
# Required env: N_SHARDS
# Array bounds are passed by the submit script: --array=0-(N_SHARDS-1)
#
# Matched-hardware timing anchor. AIF is normally a GPU planner; this runs
# it at the canonical config on a Genoa core through .venv-cpu (no
# --extra gpu, so the CUDA jaxlib is off sys.path) with JAX_PLATFORMS=cpu
# as a belt-and-braces guard. That makes AIF's seconds-per-decision
# comparable with the tree-search baselines, which are all CPU-measured.
#
# The whole point of the stage is a seconds-per-decision number that sits
# in the same table as the tree-search baselines', so what it gets from
# SLURM has to be what they got: --partition=genoa and --cpus-per-task=2,
# both copied from job_cpu_door_code.sh, and the same single environment
# override (JAX_PLATFORMS=cpu) and nothing else. Seeds and episodes are
# reduced (params.yaml aif_cpu_timing) because per-decision cost is
# shape-determined -- this stage's RETURNS ARE NOT A RESULT.
#
# Deliberately NO thread caps, here or in job_cpu_door_code.sh. Matching
# them is what matters, and matching by *removal* costs no genoa credits:
# capping would only take effect on a rerun, and the baselines' JSONs are
# already written and skipped by submit_cpu_door_code.sh, so a cap here
# would pin AIF to one thread against unpinned baselines -- a bias against
# the method the table exists to defend. Both sides therefore get the same
# 2-core allocation and let XLA/OpenMP size their pools from the affinity
# mask. The residual asymmetry is a property of the methods, not of the
# configuration: AIF's timed region is XLA and may use both allocated
# cores, while the tree-search baselines' is pure NumPy/Python and uses
# one. That is bounded by the shared 2-core allocation (at most ~2x, in
# AIF's favour), and every run records the regime it measured in
# `config.runtime` so a reader can adjudicate.
#
# --warmup-planner is not optional for a timing run: under clip_horizon
# the planner jits once per distinct effective horizon (up to 8 at the
# canonical max_steps=16, horizon=8), and one seed per shard means each
# process pays those compiles against a single seed of work. Without the
# warm-up they all land inside planning_time_s.
#
# --time is a generous 2h: AIF on one CPU core at K=6 (|theta|=720) is
# far slower per decision than on an A100, and over-provisioning a
# 10-shard array is cheap.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output/aif_cpu_timing/shards

source cluster/setup_env_cpu.sh
export JAX_PLATFORMS=cpu

N_SHARDS="${N_SHARDS:?N_SHARDS not set}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

read_param() {
    uv run python -c "import yaml; p=yaml.safe_load(open('params.yaml')); v=p$1; print(str(v).lower() if isinstance(v, bool) else v)"
}

TOTAL_SEEDS=$(read_param "['aif_cpu_timing']['n_seeds']")

echo "host=$(hostname) date=$(date -Is) planner=aif shard=${SHARD_INDEX}/${N_SHARDS}"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/run_door_code.py \
    --planner aif \
    --K "$(read_param "['env']['K']")" \
    --lock-length "$(read_param "['env']['lock_length']")" \
    --probe-noise "$(read_param "['env']['probe_noise']")" \
    --n-episodes "$(read_param "['aif_cpu_timing']['n_episodes']")" \
    --max-steps "$(read_param "['runner']['max_steps']")" \
    --horizon "$(read_param "['runner']['horizon']")" \
    --seed "$(read_param "['runner']['seed']")" \
    --learn-on-failure "$(read_param "['runner']['learn_on_failure']")" \
    --clip-horizon "$(read_param "['runner']['clip_horizon']")" \
    --planner-iterations "$(read_param "['planner']['iterations']")" \
    --planner-damping "$(read_param "['planner']['damping']")" \
    --warmup-planner \
    --total-seeds "$TOTAL_SEEDS" \
    --n-shards "$N_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --output "output/aif_cpu_timing/shards/aif.shard$(printf '%02d' "$SHARD_INDEX").json"

echo "done planner=aif shard=${SHARD_INDEX} at $(date -Is)"
