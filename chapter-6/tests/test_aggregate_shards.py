"""Cluster-shard aggregation: verify the new `derived` block survives.

`scripts/aggregate_shards.py` concatenates per-shard JSON outputs along
the seed axis. Before this fix it dropped the `derived` block entirely;
cluster-aggregated JSONs were missing Brier / NLL / episodes-to-identify
even though each shard's JSON had them.

These tests pin down the rebuild path: after aggregation, `derived` must
exist, must reflect the across-all-shards seed pool, and must be derived
from the correct per_episode posterior key for each env family.
"""

import json
import subprocess
from pathlib import Path

import numpy as np


def _write_shard(
    path: Path,
    *,
    posterior_key: str,
    posterior: list[list[float]],
    seed: int,
    extra_per_episode: dict | None = None,
) -> None:
    n_seeds = len(posterior)
    n_episodes = len(posterior[0])
    per_episode = {posterior_key: posterior}
    if extra_per_episode:
        per_episode.update(extra_per_episode)
    # Match the runner JSON schema closely enough for the aggregator's
    # validation. `curves` need only contain keys present in per_episode
    # (the aggregator pulls keys from shards[0]).
    curves = {
        posterior_key: {
            "mean": list(np.mean(posterior, axis=0)),
            "p05": list(np.percentile(posterior, 5, axis=0)),
            "p95": list(np.percentile(posterior, 95, axis=0)),
        }
    }
    shard = {
        "config": {"env": {"K": 3}, "n_seeds": n_seeds, "seed": seed},
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "theta_trues": list(range(n_seeds)),
        "per_episode": per_episode,
        "curves": curves,
        # Per-shard derived blocks intentionally diverge from each
        # other so we can confirm the aggregator recomputes rather than
        # copying shard 0's.
        "derived": {"stale_from_shard": seed},
    }
    with path.open("w") as f:
        json.dump(shard, f)


def _run_aggregator(input_glob: str, output: Path) -> dict:
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/aggregate_shards.py",
            "--input-glob",
            input_glob,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    with output.open() as f:
        return json.load(f)


def test_aggregator_rebuilds_derived_for_door_code(tmp_path: Path) -> None:
    # Two shards × 2 seeds × 3 episodes. Shard 0 identifies by ep 2;
    # shard 1 never crosses. Aggregated fraction_identified = 2/4 = 0.5.
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "aif.shard0.json",
        posterior_key="posterior_mass_on_true_theta",
        posterior=[[0.3, 0.7, 0.95], [0.2, 0.5, 0.92]],
        seed=0,
    )
    _write_shard(
        shard_dir / "aif.shard1.json",
        posterior_key="posterior_mass_on_true_theta",
        posterior=[[0.1, 0.2, 0.4], [0.1, 0.3, 0.5]],
        seed=1,
    )
    out = tmp_path / "aif.json"
    agg = _run_aggregator(str(shard_dir / "aif.shard*.json"), out)

    assert "derived" in agg
    assert "stale_from_shard" not in agg["derived"]
    assert "brier" in agg["derived"]
    assert "nll" in agg["derived"]
    eti = agg["derived"]["episodes_to_identify"]
    assert eti["fraction_identified"] == 0.5
    # Only the two crossing seeds (shard 0) contribute to the median;
    # both cross at episode 2.
    assert eti["median"] == 2.0


def test_aggregator_rebuilds_derived_for_factored_envs(tmp_path: Path) -> None:
    # FT/RS use a different posterior key — confirm the aggregator
    # auto-detects it.
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "aif.shard0.json",
        posterior_key="posterior_mass_on_persistent_theta",
        posterior=[[0.5, 0.95]],
        seed=0,
    )
    out = tmp_path / "aif.json"
    agg = _run_aggregator(str(shard_dir / "aif.shard*.json"), out)
    assert "derived" in agg
    assert agg["derived"]["episodes_to_identify"]["fraction_identified"] == 1.0
    assert agg["derived"]["episodes_to_identify"]["median"] == 1.0


def test_aggregator_concatenates_trajectories(tmp_path: Path) -> None:
    """Trajectories — per-seed list of per-episode dicts — must
    concat along the seed axis without dropping entries or changing
    order. This is the load-bearing part of the "save every action,
    recompute metrics later" workflow."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()

    def _trajectory(actions: list[int]) -> dict:
        return {
            "actions": actions,
            "observations": [[0.0]] * len(actions),
            "states_visited": list(range(len(actions) + 1)),
            "initial_q_static": [0.5, 0.5],
            "final_q_static": [0.5, 0.5],
            "terminated_early": False,
        }

    _write_shard(
        shard_dir / "aif.shard0.json",
        posterior_key="posterior_mass_on_true_theta",
        posterior=[[0.1, 0.2]],
        seed=0,
    )
    _write_shard(
        shard_dir / "aif.shard1.json",
        posterior_key="posterior_mass_on_true_theta",
        posterior=[[0.3, 0.4]],
        seed=1,
    )
    # Inject trajectories into both shards (one seed each, two episodes).
    for p, marker in [(shard_dir / "aif.shard0.json", 100),
                       (shard_dir / "aif.shard1.json", 200)]:
        with p.open() as f:
            data = json.load(f)
        data["trajectories"] = [[_trajectory([marker, marker + 1]),
                                  _trajectory([marker + 2])]]
        with p.open("w") as f:
            json.dump(data, f)

    out = tmp_path / "aif.json"
    agg = _run_aggregator(str(shard_dir / "aif.shard*.json"), out)
    assert "trajectories" in agg
    # 2 shards × 1 seed each -> 2 seeds total, preserving shard order.
    assert len(agg["trajectories"]) == 2
    assert agg["trajectories"][0][0]["actions"] == [100, 101]
    assert agg["trajectories"][0][1]["actions"] == [102]
    assert agg["trajectories"][1][0]["actions"] == [200, 201]
    assert agg["trajectories"][1][1]["actions"] == [202]


def test_aggregator_omits_trajectories_when_legacy_shard(tmp_path: Path) -> None:
    """If even one shard lacks trajectories (pre-trajectories runner
    output), the aggregator must produce a valid JSON without the
    `trajectories` key rather than crash or mix shards."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "aif.shard0.json",
        posterior_key="posterior_mass_on_true_theta",
        posterior=[[0.1]],
        seed=0,
    )
    out = tmp_path / "aif.json"
    agg = _run_aggregator(str(shard_dir / "aif.shard*.json"), out)
    assert "trajectories" not in agg


def test_aggregator_skips_derived_when_no_posterior_key(tmp_path: Path) -> None:
    # Shards without either posterior key (hypothetical future env) get
    # no derived block rather than a crash.
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard = {
        "config": {"env": {"K": 3}, "n_seeds": 1, "seed": 0},
        "n_seeds": 1,
        "n_episodes": 2,
        "theta_trues": [0],
        "per_episode": {"some_other_metric": [[1.0, 2.0]]},
        "curves": {
            "some_other_metric": {"mean": [1.0, 2.0], "p05": [1.0, 2.0], "p95": [1.0, 2.0]}
        },
    }
    with (shard_dir / "aif.shard0.json").open("w") as f:
        json.dump(shard, f)
    out = tmp_path / "aif.json"
    agg = _run_aggregator(str(shard_dir / "aif.shard*.json"), out)
    assert "derived" not in agg


# --- config.invocation provenance -----------------------------------------


def _planner_args(**overrides):
    """Argparse-like namespace spanning both arg families the record sees."""
    import argparse

    base = dict(
        planner="ids", K=3, bamcp_depth=8, ids_bamcp_depth=None,
        bamcp_n_simulations=10000, planner_iterations=70,
        # shard-varying surface, all of which must be dropped
        output=Path("/tmp/x.json"), video_dir=None, n_shards=5,
        shard_index=3, total_seeds=100, n_seeds=20, seed=3000009,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_invocation_record_drops_every_shard_varying_arg() -> None:
    """The record must be shard-invariant or `aggregate_shards.py` aborts.

    The aggregator rejects shards disagreeing on any `config` key outside
    its own exemption set, which is what turns "every shard used the same
    bamcp_depth" into a checked property. A shard-varying key in the
    record would make that check reject every multi-shard run.
    """
    from learningpomdp.experiment.io import (
        _SHARD_VARYING_ARGS,
        invocation_record,
    )

    a = _planner_args(shard_index=0, seed=0)
    b = _planner_args(shard_index=3, seed=3000009)
    assert invocation_record(a) == invocation_record(b)
    assert not _SHARD_VARYING_ARGS.intersection(invocation_record(a))


def test_invocation_record_keeps_planner_knobs_and_resolves_ids_depth() -> None:
    """`bamcp_depth` is the point of the record; IDS's is post-fallback.

    No cluster job script passes `--ids-bamcp-depth`, so the raw arg is
    None on every cluster run and only the resolved value says what depth
    IDS actually searched to.
    """
    from learningpomdp.experiment.io import invocation_record

    record = invocation_record(_planner_args(), ids_bamcp_depth=8)
    assert record["bamcp_depth"] == 8
    assert record["ids_bamcp_depth"] == 8
    assert record["bamcp_n_simulations"] == 10000
    assert record["planner_iterations"] == 70


def test_invocation_record_stringifies_paths() -> None:
    """A Path-valued arg must not reach `json.dump`.

    Every Path arg the run scripts define today (`--output`, `--video-dir`)
    is shard-varying and so excluded, making this branch unreachable from
    the current CLIs. It is kept because the failure it prevents is a
    serialization crash *after* the run finishes -- on a 14h cluster shard
    that discards the whole job -- and a future non-excluded Path arg would
    trip it with nothing else to catch it.
    """
    from learningpomdp.experiment.io import invocation_record

    record = invocation_record(_planner_args(layout_seed=Path("/tmp/l")))
    assert record["layout_seed"] == "/tmp/l"
    json.dumps(record)


def test_invocation_record_rejects_shard_varying_resolved_key() -> None:
    """A shard-varying `resolved` key is the same bug by another route."""
    import pytest

    from learningpomdp.experiment.io import invocation_record

    with pytest.raises(ValueError, match="shard-varying"):
        invocation_record(_planner_args(), seed=17)
