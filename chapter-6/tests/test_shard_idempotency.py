"""Idempotency: `is_valid_shard_json` and the skip-if-valid check in
`scripts/run_*.py`. The cluster pipeline depends on re-submission being
a no-op once shards exist; this pins the skip predicate's contract.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from learningpomdp.experiment.io import (
    EXPECTED_PER_EPISODE_KEYS,
    is_valid_shard_json,
    missing_shard_indices,
)


def _door_code_shard(n_seeds: int = 2, n_episodes: int = 3) -> dict:
    """A minimal but schema-valid door_code shard JSON."""
    arr = [[0.0] * n_episodes for _ in range(n_seeds)]
    per_episode = {key: arr for key in EXPECTED_PER_EPISODE_KEYS["door_code"]}
    return {
        "config": {"env": {"K": 3}, "n_seeds": n_seeds, "seed": 0},
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "theta_trues": list(range(n_seeds)),
        "per_episode": per_episode,
        "curves": {},
    }


def test_valid_shard_passes(tmp_path: Path) -> None:
    path = tmp_path / "aif.shard00.json"
    path.write_text(json.dumps(_door_code_shard(2, 3)))
    assert is_valid_shard_json(
        path, runner="door_code", expected_n_seeds=2, expected_n_episodes=3,
    )


def test_missing_file_fails(tmp_path: Path) -> None:
    assert not is_valid_shard_json(
        tmp_path / "nope.json",
        runner="door_code", expected_n_seeds=2, expected_n_episodes=3,
    )


def test_corrupt_json_fails(tmp_path: Path) -> None:
    path = tmp_path / "aif.shard00.json"
    path.write_text("{ not valid json")
    assert not is_valid_shard_json(
        path, runner="door_code", expected_n_seeds=2, expected_n_episodes=3,
    )


def test_wrong_n_seeds_fails(tmp_path: Path) -> None:
    """A shard from a different seed split is treated as missing; the
    re-run will produce one matching the expected split."""
    path = tmp_path / "aif.shard00.json"
    path.write_text(json.dumps(_door_code_shard(2, 3)))
    assert not is_valid_shard_json(
        path, runner="door_code", expected_n_seeds=3, expected_n_episodes=3,
    )


def test_missing_required_key_fails(tmp_path: Path) -> None:
    """A pre-trajectories shard JSON that lacks a required per_episode
    key is rejected — forces an overwrite with the new schema."""
    shard = _door_code_shard(2, 3)
    del shard["per_episode"]["posterior_mass_on_true_theta"]
    path = tmp_path / "aif.shard00.json"
    path.write_text(json.dumps(shard))
    assert not is_valid_shard_json(
        path, runner="door_code", expected_n_seeds=2, expected_n_episodes=3,
    )


def test_unknown_runner_raises() -> None:
    with pytest.raises(ValueError):
        is_valid_shard_json(
            Path("/dev/null"),
            runner="not_a_runner",
            expected_n_seeds=1,
            expected_n_episodes=1,
        )


def test_missing_shard_indices_lists_holes(tmp_path: Path) -> None:
    """3 shards total; shard 0 and 2 exist and validate, shard 1 is
    missing. `missing_shard_indices` must return [1]."""
    shard_dir = tmp_path
    # Total seeds 6, n_shards 3 -> seeds_per_shard 2.
    for idx in (0, 2):
        (shard_dir / f"aif.shard0{idx}.json").write_text(
            json.dumps(_door_code_shard(2, 3))
        )
    out = missing_shard_indices(
        shard_dir, planner="aif", n_shards=3,
        runner="door_code", expected_n_episodes=3,
        seeds_per_shard_fn=lambda _idx: 2,
    )
    assert out == [1]


def test_run_door_code_skips_when_output_valid(tmp_path: Path) -> None:
    """End-to-end: pre-write a valid shard JSON and assert
    `run_door_code.py` returns without doing work (no rewriting)."""
    output = tmp_path / "aif.shard00.json"
    # Match the expected shard size: total_seeds=1, n_shards=1 -> size 1.
    valid = _door_code_shard(1, 2)
    output.write_text(json.dumps(valid))
    original_mtime = output.stat().st_mtime_ns

    proc = subprocess.run(
        [
            "uv", "run", "python", "scripts/run_door_code.py",
            "--planner", "aif",
            "--K", "3",
            "--n-episodes", "2",
            "--max-steps", "2",
            "--horizon", "2",
            "--n-seeds", "1",
            "--total-seeds", "1",
            "--n-shards", "1",
            "--shard-index", "0",
            "--output", str(output),
        ],
        check=True, capture_output=True, text=True,
    )
    assert "skip" in proc.stdout.lower(), (
        f"expected skip message, got:\n{proc.stdout}\n---STDERR---\n{proc.stderr}"
    )
    # File untouched.
    assert output.stat().st_mtime_ns == original_mtime
    with output.open() as f:
        assert json.load(f) == valid
