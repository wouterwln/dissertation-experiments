"""Guards on params.yaml (and the run scripts' seed arithmetic) that no
runtime test can catch.

`bamcp_depth` bounds the TOTAL simulated lookahead of one MCTS simulation
(tree descent plus leaf rollout — see BAMCPPlanner's docstring), so it is
the tree-search family's counterpart to AIF-MP's `runner.horizon`. Before
this was matched, AIF-MP planned 8 / 25 / 15 / 10 steps ahead while every
tree-search baseline got 6 / 15 / 6 / 6, which confounded the central
comparison in AIF-MP's favour by construction.

Nothing in the planner code can detect that: the depth is just a number
it is handed. So the invariant lives here, as a test over the config
itself, and a future edit that silently unmatches them fails the suite.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "params.yaml"

# (params key path to the benchmark block, human name). door_code is at
# the top level; the other three are nested under their own key.
BENCHMARKS = [
    ((), "door_code"),
    (("minigrid",), "minigrid_door_code"),
    (("rock_sample",), "rock_sample"),
    (("factored_tiger",), "factored_tiger"),
]

# scripts/run_*.py::SHARD_SEED_STRIDE — shard i's RNG bases are the
# unsharded bases plus i * this.
SHARD_STRIDE = 1_000_003


def _load() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


def _block(params: dict, path: tuple[str, ...]) -> dict:
    node = params
    for key in path:
        node = node[key]
    return node


@pytest.mark.parametrize("path, name", BENCHMARKS)
def test_bamcp_depth_matches_planning_horizon(path, name) -> None:
    """Tree-search lookahead == AIF-MP lookahead on every benchmark."""
    block = _block(_load(), path)
    horizon = block["runner"]["horizon"]
    depth = block["planner"]["bamcp_depth"]
    assert depth == horizon, (
        f"{name}: planner.bamcp_depth ({depth}) != runner.horizon "
        f"({horizon}). bamcp_depth bounds total tree+rollout lookahead, so "
        f"an inequality hands one planner family more lookahead than the "
        f"other and reopens the confound this invariant exists to close."
    )


@pytest.mark.parametrize("path, name", BENCHMARKS)
def test_ids_bamcp_depth_matches_planning_horizon(path, name) -> None:
    """IDS runs its own BAMCP instance, so it needs the same parity."""
    block = _block(_load(), path)
    horizon = block["runner"]["horizon"]
    depth = block["planner"]["ids_bamcp_depth"]
    assert depth == horizon, (
        f"{name}: planner.ids_bamcp_depth ({depth}) != runner.horizon "
        f"({horizon})."
    )


@pytest.mark.parametrize("path, name", BENCHMARKS)
def test_ids_rollout_seed_never_collides_with_a_shard_seed(path, name) -> None:
    """IDS's Monte-Carlo noise must be independent of the BAMCP arm's, on
    every shard.

    The BAMCP arm seeds its rollouts from runner.seed. If IDS did too, its
    tree would be the *same* tree and its reported row would be BAMCP's
    row relabelled — which is precisely the reviewer's objection that IDS
    carries no independent evidential weight.

    Sharded runs offset BOTH bases by ``shard_index * SHARD_STRIDE`` (see
    ``scripts/run_*.py::_planner_seeds``), so ``ids_base + i*S`` must never
    equal ``runner_base + j*S`` for any pair of shards (i, j) — i.e. the
    two bases must differ by something that is *not* a multiple of the
    stride. Plain inequality is not enough: the previous default was
    1_000_003, the stride itself, which lands exactly on shard 1's runner
    base when runner.seed is 0.
    """
    block = _block(_load(), path)
    ids_base = block["planner"]["ids_rollout_seed"]
    runner_base = block["runner"]["seed"]
    assert (ids_base - runner_base) % SHARD_STRIDE != 0, (
        f"{name}: ids_rollout_seed ({ids_base}) differs from runner.seed "
        f"({runner_base}) by a multiple of the shard stride "
        f"({SHARD_STRIDE}), so IDS and BAMCP share a rollout RNG stream on "
        f"some shard and IDS is not an independent arm there."
    )


RUN_SCRIPTS = [
    "run_door_code.py", "run_minigrid_door_code.py",
    "run_rock_sample.py", "run_factored_tiger.py",
]


def _load_run_script(name: str):
    """Import a scripts/*.py module by path (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", RUN_SCRIPTS)
def test_planner_seeds_vary_per_shard(script) -> None:
    """Each shard must get its own MCTS stream.

    The runner base has always been offset per shard, but the planners were
    handed ``args.seed`` directly — so every seed a sharded run averaged
    over drew *bit-identical* rollout noise. That makes the Monte-Carlo
    error common across seeds instead of independent, and the cross-seed
    CI too narrow. Shard 0 must still be unshifted, or the canonical
    single-shard DVC pipeline would change results.
    """
    module = _load_run_script(script)
    args = [
        argparse.Namespace(seed=0, ids_rollout_seed=7919, shard_index=i)
        for i in range(4)
    ]
    rollout, ids = zip(*(module._planner_seeds(a) for a in args))
    assert rollout[0] == 0, "shard 0 must keep the unsharded planner seed"
    assert len(set(rollout)) == len(rollout), (
        f"shards share a planner stream: {rollout}"
    )
    assert len(set(ids)) == len(ids), f"IDS shards share a stream: {ids}"
    assert not set(rollout) & set(ids), (
        f"an IDS shard collides with a BAMCP shard: {rollout} vs {ids}"
    )


# The two runners that build planners through a `planner_factory` (the
# door_code / minigrid scripts construct a single instance and shard one
# seed per task instead).
FACTORY_SCRIPTS = ["run_rock_sample.py", "run_factored_tiger.py"]

# (planner name, which base seed its stream comes from).
FACTORY_PLANNERS = [
    ("bamcp", "seed"),
    ("factored_bamcp", "seed"),
    ("boss", "seed"),
    ("ids", "ids_rollout_seed"),
]


def _factory_args(planner: str) -> argparse.Namespace:
    """Minimal Namespace covering everything `_build_planner_factory` reads."""
    return argparse.Namespace(
        planner=planner,
        seed=0, ids_rollout_seed=7919, shard_index=0,
        planner_iterations=10, planner_damping=0.9,
        bamcp_n_simulations=8, bamcp_depth=2,
        bamcp_ucb_c=1.0, bamcp_gamma=0.99, boss_n_samples=2,
        rollout_policy="uniform", rollout_epsilon=0.1, rollout_q_lr=0.1,
        ids_bamcp_n_simulations=8, ids_bamcp_depth=2,
        ids_epsilon_info=1.0e-6,
    )


def _planner_rng(planner) -> "object":
    """The MCTS generator: IDS keeps its tree in a composed BAMCP."""
    return getattr(planner, "_rng", None) or planner._bamcp._rng


@pytest.mark.parametrize("script", FACTORY_SCRIPTS)
@pytest.mark.parametrize("planner, base_attr", FACTORY_PLANNERS)
def test_planner_streams_vary_per_outer_seed(script, planner, base_attr) -> None:
    """Each outer-loop seed must get its own MCTS stream.

    `_planner_seeds` fixed the cross-*shard* collision, but the factory
    bound a single `rollout_seed` and both runners call it once per outer
    seed — so every seed built `default_rng(<same int>)`, and because the
    number of draws per plan() call is fixed by n_simulations, even the
    derived Numba seeds coincided seed-for-seed. The canonical rock_sample
    and factored_tiger stages are single-shard, so *all* of their seeds
    shared one MCTS noise stream: the means survive, but the cross-seed
    ±1.96·SEM understates planner variance, which is the wrong direction to
    err in a paper arguing AIF-MP beats those baselines.

    Index 0 must stay unshifted, so seed 0 of every shard keeps the stream
    it had and the shard-level guarantee above is untouched.
    """
    module = _load_run_script(script)
    args = _factory_args(planner)
    factory = module._build_planner_factory(args)

    draws = [
        _planner_rng(factory(i)).integers(0, 2**31 - 1, size=4).tolist()
        for i in range(3)
    ]
    assert len({tuple(d) for d in draws}) == len(draws), (
        f"{script}:{planner}: outer seeds share one planner stream: {draws}"
    )

    base = getattr(args, base_attr)
    expected = np.random.default_rng(base).integers(
        0, 2**31 - 1, size=4,
    ).tolist()
    assert draws[0] == expected, (
        f"{script}:{planner}: seed index 0 no longer reproduces "
        f"default_rng({base}), so shard 0's canonical stream moved"
    )


def test_depth_sweeps_reach_the_matched_depth_without_exceeding_it() -> None:
    """A depth sweep must reach the canonical depth and must not go past it.

    Reaching it is what makes the curve informative: without a cell below
    and a cell at the matched depth, the sweep cannot show whether the
    tree-search arms are lookahead-limited.

    Not going past it is a correctness requirement, not taste.
    ``BAMCPPlanner`` uses ``max_depth = min(self.depth, max(horizon, 1))``
    and ``self.depth`` enters nowhere else — not even the RNG stream — and
    the horizon handed to ``plan()`` never exceeds ``runner.horizon``. So a
    cell with ``depth > horizon`` is *bit-identical* to the cell at
    ``depth == horizon``, and its flat right tail would be flat by
    construction rather than measured. This test previously mandated such a
    cell; the earlier ``[2, 4, 6, 8, 12]`` and ``[3, 6, 10, 15, 20]``
    sweeps each shipped one duplicate cell per planner.

    The positive reading is the stronger fairness claim: matched lookahead
    is the *saturation point* of the depth axis, so no setting of
    ``bamcp_depth`` gives the tree-search family more lookahead than
    AIF-MP's horizon.
    """
    params = _load()
    cases = [
        ("door_code_depth", params["runner"]["horizon"]),
        ("rock_sample_depth", params["rock_sample"]["runner"]["horizon"]),
    ]
    for block_name, horizon in cases:
        sweep = params[block_name]["depth_sweep"]
        assert min(sweep) < horizon, (
            f"{block_name}.depth_sweep has no cell below the canonical "
            f"depth {horizon}: {sweep}"
        )
        assert max(sweep) == horizon, (
            f"{block_name}.depth_sweep must top out at the canonical depth "
            f"{horizon}, got max {max(sweep)} in {sweep}. Cells above the "
            f"horizon are bit-identical no-ops (min(depth, horizon) "
            f"saturates), so they buy a duplicate run, not a data point."
        )


def test_every_dvc_interpolation_resolves() -> None:
    """Every `${...}` in dvc.yaml must name a real params.yaml key.

    A missing one is not a stale comment, it is a stage that cannot run:
    `factored_tiger_preference.n_seeds` was referenced by two stages and
    the cluster submit script while the key did not exist, so
    `dvc repro factored_tiger_preference` died on a parse error and the
    overnight submit would have crashed on a KeyError. Nothing else
    notices until something tries to run.
    """
    params = _load()

    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from strings(value)

    with (REPO_ROOT / "dvc.yaml").open() as f:
        dvc = yaml.safe_load(f)

    missing = set()
    for text in strings(dvc):
        for expr in re.findall(r"\$\{([^}]+)\}", text):
            # `item.*` is matrix/foreach interpolation, not a params key.
            if expr.startswith("item."):
                continue
            node = params
            for key in expr.split("."):
                if not isinstance(node, dict) or key not in node:
                    missing.add(expr)
                    break
                node = node[key]

    assert not missing, (
        f"dvc.yaml interpolates params keys that do not exist: "
        f"{sorted(missing)}"
    )


def test_rollout_ablation_covers_both_policies() -> None:
    """The rollout ablation is only decisive if it runs the shipped
    default alongside the corrected policy."""
    policies = _load()["door_code_rollout"]["policies"]
    assert set(policies) == {"uniform", "eps_greedy_q"}, policies
