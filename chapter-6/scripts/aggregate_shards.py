"""Combine per-shard JSON outputs produced by a sharded SLURM array run
into a single per-planner JSON identical in shape to a non-sharded run.

Per-episode arrays from each shard are concatenated along the seed axis
(axis 0). Curves are recomputed with nanmean / nanpercentile — matching
the rock_sample and factored_tiger runners, and reducing to plain
mean/percentile when the input has no NaNs (door_code).

Example:

    uv run python scripts/aggregate_shards.py \\
        --input-glob 'output/door_code/aif.shard*.json' \\
        --output output/door_code/aif.json
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import re
from pathlib import Path

import numpy as np

from learningpomdp.experiment.door_code_runner import build_derived_metrics


_SHARD_RE = re.compile(r"shard(\d+)")

# Per-episode key that holds posterior mass on the true persistent theta.
# Door-code / minigrid use the first name; factored_tiger / rock_sample
# use the second. The derived block (Brier / NLL / episodes-to-identify)
# is computed against whichever is present after aggregation.
_POSTERIOR_MASS_KEYS = (
    "posterior_mass_on_true_theta",
    "posterior_mass_on_persistent_theta",
)


def _shard_index_from_path(path: Path) -> int:
    m = _SHARD_RE.search(path.name)
    if m is None:
        raise SystemExit(
            f"cannot infer shard index from filename: {path.name} "
            f"(expected '...shard<N>...')"
        )
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate sharded per-planner JSON outputs into a single JSON "
            "matching the non-sharded runner schema."
        ),
    )
    parser.add_argument(
        "--input-glob", type=str, required=True,
        help="shell glob matching shard JSONs (e.g. "
             "'output/door_code/aif.shard*.json')",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="path to write the aggregated JSON",
    )
    args = parser.parse_args()

    paths = sorted(
        (Path(p) for p in _glob.glob(args.input_glob)),
        key=_shard_index_from_path,
    )
    if not paths:
        raise SystemExit(f"no shard files matched {args.input_glob!r}")

    shards = []
    for p in paths:
        with p.open() as f:
            shards.append(json.load(f))

    n_episodes = shards[0]["n_episodes"]
    # `n_seeds` and `seed` are expected to differ per shard (each shard
    # runs its own slice with shard_seed = base + idx * 1_000_003);
    # everything else in `config` must match shard 0 or we're silently
    # mixing JSONs from different runs.
    #
    # `runtime` is exempt for a different reason: it records the process's
    # thread/backend regime, which is a property of where a shard ran, and
    # shards written before the key existed do not have it at all. A
    # divergence there does not mean the runs are incompatible — it means
    # the *timings* are not comparable, so it warns instead of aborting.
    _PER_SHARD_CONFIG_KEYS = {"n_seeds", "seed", "runtime"}
    ref_config = {
        k: v for k, v in shards[0]["config"].items()
        if k not in _PER_SHARD_CONFIG_KEYS
    }
    for s, p in zip(shards, paths):
        if s["n_episodes"] != n_episodes:
            raise SystemExit(
                f"shard {p} has n_episodes={s['n_episodes']} "
                f"!= {n_episodes} from shard 0"
            )
        cfg = {
            k: v for k, v in s["config"].items()
            if k not in _PER_SHARD_CONFIG_KEYS
        }
        if cfg != ref_config:
            diffs = [
                f"{k}: {ref_config.get(k)!r} vs {cfg.get(k)!r}"
                for k in set(ref_config) | set(cfg)
                if ref_config.get(k) != cfg.get(k)
            ]
            raise SystemExit(
                f"shard {p} config disagrees with shard 0 on: "
                + "; ".join(diffs)
            )

    runtimes = [s["config"].get("runtime") for s in shards]
    if any(r != runtimes[0] for r in runtimes):
        print(
            "warning: shards disagree on config.runtime — they ran under "
            "different thread/backend regimes, so per-decision timings "
            "from this aggregate are not comparable. Shard 0's record is "
            "the one retained; check the per-shard JSONs.",
            flush=True,
        )

    per_episode_keys = list(shards[0]["per_episode"].keys())
    per_episode: dict[str, list[list[float]]] = {}
    for key in per_episode_keys:
        stacked = np.concatenate(
            [np.asarray(s["per_episode"][key]) for s in shards],
            axis=0,
        )
        per_episode[key] = stacked.tolist()

    curve_keys = list(shards[0]["curves"].keys())
    curves: dict[str, dict[str, list[float]]] = {}
    for key in curve_keys:
        arr = np.asarray(per_episode[key], dtype=float)
        curves[key] = {
            "mean": np.nanmean(arr, axis=0).tolist(),
            "p05": np.nanpercentile(arr, 5, axis=0).tolist(),
            "p95": np.nanpercentile(arr, 95, axis=0).tolist(),
        }

    theta_trues: list = []
    for s in shards:
        theta_trues.extend(s.get("theta_trues", []))

    # Trajectories (per-seed list of per-episode dicts) — concat along
    # the seed axis. Plain list.extend; contents are jagged JSON
    # (variable per-step length) and don't need numpy round-tripping.
    # Shards from a pre-trajectories run won't have this key, so skip
    # rather than fail; the aggregate will simply lack `trajectories`.
    trajectories: list = []
    have_trajectories = all("trajectories" in s for s in shards)
    if have_trajectories:
        for s in shards:
            trajectories.extend(s["trajectories"])

    total_n_seeds = sum(int(s["n_seeds"]) for s in shards)

    # Overwrite `config.n_seeds` with the total so plot_*.py — which read
    # `cfg["n_seeds"]` for figure titles — see the aggregated count, not
    # shard 0's per-shard slice.
    agg_config = dict(shards[0]["config"])
    agg_config["n_seeds"] = total_n_seeds

    aggregated = {
        "config": agg_config,
        "n_seeds": total_n_seeds,
        "n_episodes": n_episodes,
        "theta_trues": theta_trues,
        "per_episode": per_episode,
        "curves": curves,
    }
    if have_trajectories:
        aggregated["trajectories"] = trajectories

    # Rebuild `derived` from the aggregated posterior. Per-shard derived
    # blocks would only describe that shard's seed slice; the figures
    # need the across-all-seeds scalar.
    posterior_key = next(
        (k for k in _POSTERIOR_MASS_KEYS if k in per_episode), None,
    )
    if posterior_key is not None:
        posterior = np.asarray(per_episode[posterior_key], dtype=float)
        aggregated["derived"] = build_derived_metrics(posterior)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(aggregated, f, indent=2)

    print(
        f"Aggregated {len(shards)} shards "
        f"({total_n_seeds} seeds total) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
