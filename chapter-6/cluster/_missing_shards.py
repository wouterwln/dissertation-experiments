"""Print the comma-separated SLURM array spec for missing shard indices.

Called by `cluster/submit_*.sh` to skip shards whose JSON already exists
and validates. Empty output means every shard is complete and the array
job should be skipped entirely.

Usage:

    uv run python cluster/_missing_shards.py \\
        --shard-dir output/door_code/shards \\
        --planner aif \\
        --runner door_code \\
        --n-shards 12 --total-seeds 12 \\
        --n-episodes 30
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from learningpomdp.experiment.io import (
    is_valid_shard_json,
    format_array_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--planner", required=True)
    parser.add_argument(
        "--runner", required=True,
        choices=["door_code", "rock_sample", "factored_tiger"],
        help="which runner produced the shard JSON (drives schema check)",
    )
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--total-seeds", type=int, required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    args = parser.parse_args()

    seeds_per_shard = math.ceil(args.total_seeds / args.n_shards)
    missing: list[int] = []
    for idx in range(args.n_shards):
        shard_start = idx * seeds_per_shard
        shard_size = min(seeds_per_shard, args.total_seeds - shard_start)
        if shard_size <= 0:
            continue
        path = args.shard_dir / f"{args.planner}.shard{idx:02d}.json"
        if not is_valid_shard_json(
            path,
            runner=args.runner,
            expected_n_seeds=shard_size,
            expected_n_episodes=args.n_episodes,
        ):
            missing.append(idx)

    print(format_array_spec(missing))


if __name__ == "__main__":
    main()
