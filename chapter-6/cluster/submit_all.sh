#!/usr/bin/env bash
# Submit every experiment stage on both GPU and CPU partitions.
# Each submit script self-chains an aggregate job via SLURM dependency
# (afterany on the array JIDs). This wrapper additionally chains a
# per-stage plot job (afterok on that stage's aggregate JIDs across
# GPU + CPU) so output/<stage>/figure.pdf is produced automatically.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Fail before submitting anything if a dvc.yaml run-stage has no submit
# script, or has one that this wrapper never calls. Both happened
# silently before (door_code_simscaling had a script but was not called;
# door_code_preference had none), so an "everything" run quietly wasn't.
echo "=== Preflight: stage coverage ==="
uv run python cluster/preflight.py

ENVS=(door_code minigrid_door_code rock_sample rock_sample_kscaling factored_tiger)

# Ablation stages: one partition each, so no GPU/CPU pairing. Entries are
# "<stage> <submit script> [reference env]"; skipped with a warning if the
# script is absent, since preflight already distinguishes a declared-pending
# gap from a real one.
#
# The third field names a canonical env whose aggregates the ablation's
# *figure* also needs. The preference plots draw the canonical baselines as
# reference levels and hard-fail on a missing output/<env>/<planner>.json,
# so on a from-scratch run their plot job has to wait for that env's
# aggregates as well as its own — chaining on the ablation alone would fire
# the plot before the reference exists. The depth/rollout/simscaling
# figures read only their own directory and need no reference.
ABLATIONS=(
    "door_code_simscaling      cluster/submit_cpu_door_code_simscaling.sh"
    "door_code_depth           cluster/submit_cpu_door_code_depth.sh"
    "door_code_rollout         cluster/submit_cpu_door_code_rollout.sh"
    "rock_sample_depth         cluster/submit_cpu_rock_sample_depth.sh"
    "aif_convergence           cluster/submit_aif_convergence.sh"
    "door_code_preference      cluster/submit_door_code_preference.sh       door_code"
    "rock_sample_preference    cluster/submit_rock_sample_preference.sh     rock_sample"
    "factored_tiger_preference cluster/submit_factored_tiger_preference.sh  factored_tiger"
)

JID_DIR="$(mktemp -d)"
trap 'rm -rf "$JID_DIR"' EXIT
for env in "${ENVS[@]}"; do
    : > "$JID_DIR/${env}.jids"
done

# Run a per-env submit script, show its output live, and append the
# aggregate JIDs it printed (lines containing "aggregate <number>") to
# this env's JID file. Used to fan-in plot deps across GPU+CPU.
collect() {
    local env="$1"
    local script="$2"
    local tmp
    tmp="$(mktemp)"
    bash "$script" | tee "$tmp"
    grep -oE "aggregate [0-9]+" "$tmp" | awk '{print $2}' >> "$JID_DIR/${env}.jids" || true
    rm -f "$tmp"
}

collect door_code              cluster/submit_door_code.sh
collect minigrid_door_code     cluster/submit_minigrid_door_code.sh
collect rock_sample            cluster/submit_rock_sample.sh
collect rock_sample_kscaling   cluster/submit_rock_sample_kscaling.sh
collect factored_tiger         cluster/submit_factored_tiger.sh

collect door_code              cluster/submit_cpu_door_code.sh
collect minigrid_door_code     cluster/submit_cpu_minigrid_door_code.sh
collect rock_sample            cluster/submit_cpu_rock_sample.sh
collect rock_sample_kscaling   cluster/submit_cpu_rock_sample_kscaling.sh
collect factored_tiger         cluster/submit_cpu_factored_tiger.sh

echo ""
echo "=== Submitting per-stage plot jobs ==="
for env in "${ENVS[@]}"; do
    jids="$(paste -sd, "$JID_DIR/${env}.jids" 2>/dev/null || true)"
    if [ -n "$jids" ]; then
        PJID=$(sbatch --parsable \
            --job-name="plot-${env}" \
            --dependency=afterok:"$jids" \
            --kill-on-invalid-dep=yes \
            --export="ALL,STAGE=${env}" \
            cluster/job_plot.sh)
        echo "  ${env}: plot ${PJID} (afterok ${jids})"
    else
        PJID=$(sbatch --parsable \
            --job-name="plot-${env}" \
            --export="ALL,STAGE=${env}" \
            cluster/job_plot.sh)
        echo "  ${env}: plot ${PJID} (no upstream aggregates — all planner JSONs already present)"
    fi
done

# Ablations come last, and each is guarded: under `set -e` an sbatch
# rejection here would otherwise abort the script, leaving the canonical
# arrays submitted with no plot jobs chained behind them.
echo ""
echo "=== Submitting ablation stages ==="
for entry in "${ABLATIONS[@]}"; do
    read -r stage script refenv <<<"$entry"
    if [ ! -f "$script" ]; then
        echo "  SKIP ${stage}: ${script} not present" >&2
        continue
    fi
    echo "--- ${stage}"
    tmp="$(mktemp)"
    if ! bash "$script" | tee "$tmp"; then
        echo "  WARNING ${stage}: submit failed, continuing" >&2
        rm -f "$tmp"
        continue
    fi
    # `|| true` is load-bearing: grep exits 1 when the submit script
    # printed no aggregate line (the normal resume path — every budget
    # already has its output JSON), and under `set -o pipefail` that
    # status becomes the assignment's status, so `set -e` would kill the
    # script here, *after* the canonical arrays were submitted. Same
    # reason collect() carries it.
    ajids="$(grep -oE "aggregate [0-9]+" "$tmp" | awk '{print $2}' | paste -sd, - || true)"
    rm -f "$tmp"
    # Fold in the reference env's aggregates for the figures that read it.
    if [ -n "${refenv:-}" ]; then
        rjids="$(paste -sd, "$JID_DIR/${refenv}.jids" 2>/dev/null || true)"
        if [ -n "$rjids" ]; then
            ajids="${ajids:+${ajids},}${rjids}"
            echo "  ${stage}: + ${refenv} aggregates ${rjids} (reference levels)"
        fi
    fi
    # Chain the ablation's own figure, so an enabled sweep does not
    # produce shards and an aggregate but never a plot.
    if [ -f "scripts/plot_${stage}.py" ]; then
        if [ -n "$ajids" ]; then
            PJID=$(sbatch --parsable --job-name="plot-${stage}" \
                --dependency=afterok:"$ajids" --kill-on-invalid-dep=yes \
                --export="ALL,STAGE=${stage}" cluster/job_plot.sh) \
                || { echo "  WARNING ${stage}: plot submit failed" >&2; continue; }
            echo "  ${stage}: plot ${PJID} (afterok ${ajids})"
        else
            PJID=$(sbatch --parsable --job-name="plot-${stage}" \
                --export="ALL,STAGE=${stage}" cluster/job_plot.sh) \
                || { echo "  WARNING ${stage}: plot submit failed" >&2; continue; }
            echo "  ${stage}: plot ${PJID} (no upstream aggregates)"
        fi
    fi
done

echo ""
echo "All GPU + CPU stages submitted; plot jobs chained after aggregates."
echo "Monitor:  squeue -u \$USER"
