#!/usr/bin/env bash
# Submit all five experiment stages on both GPU and CPU partitions.
# Each submit script self-chains an aggregate job via SLURM dependency
# (afterany on the array JIDs). This wrapper additionally chains a
# per-stage plot job (afterok on that stage's aggregate JIDs across
# GPU + CPU) so output/<stage>/figure.pdf is produced automatically.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

ENVS=(door_code minigrid_door_code rock_sample rock_sample_kscaling factored_tiger)

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

echo ""
echo "All GPU + CPU stages submitted; plot jobs chained after aggregates."
echo "Monitor:  squeue -u \$USER"
