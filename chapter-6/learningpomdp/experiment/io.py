"""Shard-output IO helpers: idempotency check and invocation provenance
for `scripts/run_*.py`.

A shard JSON is "valid" if it parses, has the required top-level keys,
and its per-episode arrays have shape (expected_n_seeds, n_episodes).
That lets `scripts/run_*.py` and `cluster/submit_*.sh` short-circuit
when a previous run already produced a shard, so resubmitting a SLURM
array converges to a complete `output/` tree without recomputing
finished shards.

`invocation_record` writes the CLI values a run was invoked with into
`config.invocation`, so a result JSON is auditable without reference to
`params.yaml` git history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# Per-runner: a small subset of per_episode keys that must be present
# for the shard JSON to count as a complete run of that runner. Not the
# full schema — just enough to catch "wrong runner wrote here" or
# "shard JSON is from a pre-trajectories code version".
EXPECTED_PER_EPISODE_KEYS: dict[str, tuple[str, ...]] = {
    "door_code": (
        "success",
        "n_steps",
        "posterior_mass_on_true_theta",
        "cumulative_reward",
    ),
    "rock_sample": (
        "n_steps",
        "cumulative_reward",
        "posterior_mass_on_persistent_theta",
        "reward_normalised",
    ),
    "factored_tiger": (
        "n_steps",
        "cumulative_reward",
        "posterior_mass_on_persistent_theta",
        "reward_normalised",
    ),
}


def is_valid_shard_json(
    path: Path,
    runner: str,
    expected_n_seeds: int,
    expected_n_episodes: int,
) -> bool:
    """Return True iff `path` is a complete shard JSON for `runner`.

    Checks:
      - file parses as JSON
      - top-level keys: config, n_seeds, n_episodes, per_episode, curves
      - per_episode contains the expected keys for this runner
      - each per_episode scalar array has shape
        (expected_n_seeds, expected_n_episodes)
      - `n_seeds` and `n_episodes` top-level scalars match expectations
    """
    if runner not in EXPECTED_PER_EPISODE_KEYS:
        raise ValueError(
            f"unknown runner {runner!r}; "
            f"expected one of {sorted(EXPECTED_PER_EPISODE_KEYS)}"
        )
    if not path.exists():
        return False
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    for key in ("config", "n_seeds", "n_episodes", "per_episode", "curves"):
        if key not in data:
            return False

    if int(data["n_seeds"]) != int(expected_n_seeds):
        return False
    if int(data["n_episodes"]) != int(expected_n_episodes):
        return False

    per_episode = data["per_episode"]
    if not isinstance(per_episode, dict):
        return False
    for key in EXPECTED_PER_EPISODE_KEYS[runner]:
        if key not in per_episode:
            return False
        arr = per_episode[key]
        if not _is_2d_with_shape(arr, expected_n_seeds, expected_n_episodes):
            return False

    return True


def _is_2d_with_shape(arr, n_outer: int, n_inner: int) -> bool:
    """Lightweight shape check on a nested-list JSON array."""
    if not isinstance(arr, list) or len(arr) != n_outer:
        return False
    for row in arr:
        if not isinstance(row, list) or len(row) != n_inner:
            return False
    return True


def missing_shard_indices(
    shard_dir: Path,
    planner: str,
    n_shards: int,
    runner: str,
    expected_n_episodes: int,
    seeds_per_shard_fn,
) -> list[int]:
    """Return the array indices in [0, n_shards) whose shard JSON is
    missing or invalid. Used by `cluster/submit_*.sh` to compute the
    `--array=<comma-list>` of shards that still need to run.

    `seeds_per_shard_fn(idx) -> int` gives the per-shard seed count
    (matches the splitting logic in `scripts/run_*.py:main`).
    """
    missing: list[int] = []
    for idx in range(n_shards):
        path = shard_dir / f"{planner}.shard{idx:02d}.json"
        if not is_valid_shard_json(
            path,
            runner=runner,
            expected_n_seeds=seeds_per_shard_fn(idx),
            expected_n_episodes=expected_n_episodes,
        ):
            missing.append(idx)
    return missing


def format_array_spec(indices: Iterable[int]) -> str:
    """Format `indices` as a SLURM `--array=` spec (comma-separated)."""
    return ",".join(str(i) for i in indices)


# Args excluded from `invocation_record`, all for the same reason: they
# differ between shards of one run. `aggregate_shards.py` aborts when
# shards disagree on any `config` key outside its own small exemption
# set, and that check is the reason the record is worth writing at all —
# it is what makes "every shard used the same bamcp_depth" a verified
# property rather than an assumption. Anything shard-varying must
# therefore stay out. Nothing is excluded for being uninteresting.
_SHARD_VARYING_ARGS = frozenset({
    "output", "video_dir", "n_shards", "shard_index", "total_seeds",
    "n_seeds", "seed",
})


def invocation_record(args, **resolved) -> dict:
    """The CLI values this run was invoked with, for `config.invocation`.

    Written because `config` previously carried the planner *name* and
    nothing else about the planner: `bamcp_depth` sets the tree-search
    family's total simulated lookahead, and no file under `output/`
    recorded it, so which depth produced a published number could only be
    recovered from `params.yaml` git history. That is exactly the axis a
    reviewer contested, and history is not an artifact a reader has.

    Default-include, with an explicit exclude set, rather than an
    allowlist of planner knobs. The failure mode being fixed is a knob
    nobody wrote down, so a knob added later has to be captured without
    anyone remembering to extend this function. The cost is that the
    record repeats the env/runner values `config` already holds, which is
    harmless and leaves the record readable on its own.

    `resolved` carries values computed in the run script rather than read
    off the CLI, which therefore cannot be reconstructed from
    `params.yaml`: pass `ids_bamcp_depth` post-`_ids_depth`-fallback, for
    instance. Keys must be shard-invariant for the same reason as
    `_SHARD_VARYING_ARGS` — the shard-offset RNG seeds are not, and are
    omitted deliberately (each shard's is already in `config["seed"]`,
    and `SHARD_SEED_STRIDE` is a documented constant).
    """
    record = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in sorted(vars(args).items())
        if key not in _SHARD_VARYING_ARGS
    }
    overlap = _SHARD_VARYING_ARGS.intersection(resolved)
    if overlap:
        raise ValueError(
            f"resolved keys {sorted(overlap)} are shard-varying; recording "
            "them would make aggregate_shards.py reject every multi-shard run"
        )
    record.update(resolved)
    return record
