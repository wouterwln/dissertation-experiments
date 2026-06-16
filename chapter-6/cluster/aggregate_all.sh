#!/usr/bin/env bash
# Stitch per-shard JSONs into per-planner JSONs for one or all stages.
# Run on the login node once the corresponding sbatch arrays finish.
#
# Usage:
#   bash cluster/aggregate_all.sh                       # all stages
#   bash cluster/aggregate_all.sh door_code             # one stage
#   bash cluster/aggregate_all.sh rock_sample_kscaling  # iterates over K
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GPU_PLANNERS=(aif vbp psrl oracle)

aggregate_one_dir() {
    local out_dir="$1"
    for planner in "${GPU_PLANNERS[@]}"; do
        local glob="${out_dir}/shards/${planner}.shard*.json"
        # shellcheck disable=SC2086
        if ! compgen -G "${glob}" >/dev/null; then
            echo "  ${out_dir}/${planner}: no shards found, skipped"
            continue
        fi
        uv run python scripts/aggregate_shards.py \
            --input-glob "${glob}" \
            --output "${out_dir}/${planner}.json"
    done
}

aggregate_stage() {
    local stage="$1"
    echo "=== aggregating ${stage} ==="
    if [ "$stage" = "rock_sample_kscaling" ]; then
        local k_list
        k_list=$(uv run python -c \
            "import yaml; print(' '.join(str(k) for k in yaml.safe_load(open('params.yaml'))['rock_sample']['k_sweep']))")
        for K in $k_list; do
            aggregate_one_dir "output/rock_sample_kscaling/K${K}"
        done
    else
        aggregate_one_dir "output/${stage}"
    fi
}

STAGES=("${@:-door_code minigrid_door_code rock_sample rock_sample_kscaling factored_tiger}")
# If user passed no args, expand the default list to positional words:
if [ "$#" -eq 0 ]; then
    set -- door_code minigrid_door_code rock_sample rock_sample_kscaling factored_tiger
fi

for stage in "$@"; do
    aggregate_stage "$stage"
done

echo ""
echo "Aggregation complete. Plot with the existing dvc.yaml plot stages, e.g.:"
echo "  uv run python scripts/plot_door_code.py --input-dir output/door_code --output output/door_code/figure.pdf"
