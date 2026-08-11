"""Plot the AIF-MP convergence sensitivity report.

Reads output/aif_convergence/iter<N>_damp<D>/aif.json for the grid in
params.aif_convergence.{iterations_sweep, damping_sweep} and renders:

  Top row (behaviour):
    return vs iteration count | return vs damping | return vs cost
  Bottom row (fixed point):
    residual trace | terminal residual vs iterations | ... vs damping

The channel residual is a planner-internal diagnostic and is not recorded
in the run JSONs, so this script recomputes it directly: one
`AIFPlanner.plan(..., return_residuals=True)` call per cell, on the same
env config the cell was run with, from the first decision of episode 0
(agent at START, q(theta) uniform — the hardest belief in the block).

The residual is `bp_core.channel_tv_residual`: the worst-conditional
total variation between a channel and its *undamped* image. Being the
residual of the fixed-point map rather than of the damped step, it is
comparable across damping as well as across iterations, which is what
licenses the two panels that vary damping at fixed iteration count.

Two things the figure caption states and the reader must carry:

1. The sweep is confounded. `planner.iterations` and `planner.damping`
   are handed to `_precompute_obs_channels` as well as to the main
   fixed-point loop (`aif_planner._aif_planning_impl`, "n_iterations" is
   passed straight through), and that function runs its own `fori_loop`.
   So a 140-iteration cell starts its main loop from more-converged
   observation channels than a 70-iteration cell does: the grid varies
   two coupled quantities while reading as if it varied one.
2. The residual trace panel and the terminal-residual panels are not the
   same number at the same x. The trace is taken from the *longest* cell
   at each damping, so its value at iteration 70 is a mid-trace value of
   the 140-iteration cell; "terminal residual at 70 iterations" is the
   last value of the 70-iteration cell's own trace, a different run whose
   obs channels also ran 70 rather than 140 iterations. The two agree
   only if the obs-channel loop has settled by then, which the figure
   does not establish.

   Measured off-figure on the canonical door_code config (K=6, L=4, H=8,
   first decision of episode 0) they do agree: across the whole
   5x4 grid the per-cell terminal residual and the 140-iteration cell's
   trace read at the same index differ by at most 2.7e-4 relative, and
   are bit-identical in most cells. So on *this* env the obs channels are
   effectively converged well before 10 iterations and the confound in
   (1) is numerically negligible. That is a fact about door_code, not a
   property of the measure, so the panels stay separately labelled.

"Return vs cost" uses `aif_mp_contraction_count`, a hardware-independent
count of tensor elements read by the fixed-point loop's contractions. It
replaces wall-clock for the accuracy/compute tradeoff, since wall-clock
here would be an A100 number compared against CPU baselines. It is a
per-decision **ceiling**: it is evaluated at the nominal `runner.horizon`,
while under `runner.clip_horizon` the horizon the planner actually
receives shrinks as the episode budget runs down. Cells are directly
comparable to each other (the same ceiling factor applies to all of
them); the absolute value is an upper bound.

This is a SENSITIVITY REPORT, not a hyperparameter search: the canonical
(70, 0.9) cell is outlined rather than selected, and no cell from this
grid feeds back into params.yaml.

Usage:
    uv run python scripts/plot_aif_convergence.py \\
        --input-dir output/aif_convergence \\
        --output output/aif_convergence/figure.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo-root dissertation theme (single source of truth for figure colors).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402

from _cards import save_panel_cards  # noqa: E402

# The canonical message-passing cell (params.yaml planner.{iterations,
# damping}). Outlined in the figure as a reference point, not a selection.
CANONICAL_CELL = (70, 0.9)

# This slate's card geometry: the 16 x 9 report lays its panels out wider
# and squatter than the benchmark dashboards' 5.0 x 4.5 cells, so the
# shared CARD_SIZE_IN would break width instead of equalising it.
# Measured across the 6 panels the tight boxes land at 5.28 x 3.29 in,
# which this rounds just past (same re-measure rule as _cards.py).
_CARD_SIZE_IN = (5.3, 3.35)

_CELL_RE = re.compile(r"^iter(\d+)_damp([0-9.]+)$")


def _discover_cells(input_dir: Path) -> list[tuple[int, float]]:
    """List (iterations, damping) cells present under input_dir, sorted."""
    cells: list[tuple[int, float]] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _CELL_RE.match(child.name)
        if m is not None:
            cells.append((int(m.group(1)), float(m.group(2))))
    return sorted(cells)


def _cell_dir(iters: int, damping: float) -> str:
    """Cell directory name; must match dvc.yaml's interpolation of the
    matrix values.

    DVC interpolates the *parsed* YAML value, so the rendering is Python's
    str() of a float: 0.5 -> "0.5", 1.0 -> "1.0". Hence float(), not
    format specs — a "{:g}" would render 1.0 as "1" and silently miss the
    directory dvc.yaml wrote, reporting it as a missing run instead.
    """
    return f"iter{iters}_damp{float(damping)}"


def _final_return(summary: dict) -> tuple[float, float]:
    """Mean +/- 1.96 SEM of last-episode terminal reward across seeds.

    1.96 SEM matches the main paper figures' convention (the older
    ablation plots use percentile bands, which is a separate
    inconsistency; do not add a third convention here).
    """
    rewards = np.asarray(summary["per_episode"]["terminal_reward"], dtype=float)
    final = rewards[:, -1]
    n = max(final.size, 1)
    return float(final.mean()), float(1.96 * final.std(ddof=1) / np.sqrt(n))


def _residual_traces(
    cells: list[tuple[int, float]], config: dict,
) -> dict[tuple[int, float], np.ndarray]:
    """Per-cell channel-residual trace from the planner itself.

    Imported lazily so the module-level plotting path stays importable
    without pulling JAX in.
    """
    import jax.numpy as jnp

    from learningpomdp.environments.door_code import DoorCodeEnv, state_indices
    from learningpomdp.planning.aif_planner import AIFPlanner

    env_kwargs = dict(config["env"])
    env = DoorCodeEnv(**env_kwargs)
    lock_length = env_kwargs.get("lock_length", 1)

    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32)
        for B_c in env.get_observation_tensors()
    ]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    layout = state_indices(env.M, lock_length)
    q_state = (
        jnp.zeros(env.n_states, dtype=jnp.float32).at[layout["START"]].set(1.0)
    )
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    plan_kwargs = dict(
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
        theta_mask=env.get_modality_registry().theta_dependent_mask,
    )

    traces = {}
    for iters, damping in cells:
        planner = AIFPlanner(n_iterations=iters, damping=damping)
        _, residuals = planner.plan(
            q_state, q_static, T, B_list, goal, config["horizon"],
            return_residuals=True, **plan_kwargs,
        )
        traces[(iters, damping)] = np.asarray(residuals)
    return traces


def _contraction_counts(
    cells: list[tuple[int, float]], config: dict,
) -> dict[tuple[int, float], int]:
    """Hardware-free per-decision cost ceiling for each cell.

    At the nominal `runner.horizon`, i.e. an upper bound — see the module
    docstring on `clip_horizon`. Common across cells, so the panel's
    cell-to-cell comparison is unaffected.
    """
    from learningpomdp.environments.door_code import DoorCodeEnv
    from learningpomdp.planning.cost_model import aif_mp_contraction_count

    env = DoorCodeEnv(**config["env"])
    obs_channels = [
        (B.shape[0], int(B.shape[2]) if B.ndim == 3 else 1)
        for B in env.get_observation_tensors()
    ]
    return {
        (iters, damping): aif_mp_contraction_count(
            n_iterations=iters,
            horizon=config["horizon"],
            n_states=env.n_states,
            n_actions=env.n_actions,
            obs_channels=obs_channels,
        )
        for iters, damping in cells
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the AIF-MP convergence sensitivity report.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/aif_convergence"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cards-dir", type=Path, default=None,
        help="if set, also write each panel as its own PDF card here",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"{args.input_dir} does not exist; "
            f"run `dvc repro aif_convergence` first"
        )
    cells = _discover_cells(args.input_dir)
    if not cells:
        raise FileNotFoundError(
            f"no iter<N>_damp<D> subdirectories under {args.input_dir}; "
            f"run `dvc repro aif_convergence` first"
        )

    runs = {}
    for iters, damping in cells:
        path = args.input_dir / _cell_dir(iters, damping) / "aif.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run `dvc repro aif_convergence` first"
            )
        with path.open() as f:
            runs[(iters, damping)] = json.load(f)

    iters_values = sorted({c[0] for c in cells})
    damp_values = sorted({c[1] for c in cells})
    cfg = runs[cells[0]]["config"]

    traces = _residual_traces(cells, cfg)
    costs = _contraction_counts(cells, cfg)

    theme.apply_rcparams()
    # Both sweeps are ordered, so both wear the theme's ordinal ramp.
    damp_colors = dict(zip(damp_values, theme.ordinal_colors(len(damp_values))))
    iters_colors = dict(
        zip(iters_values, theme.ordinal_colors(len(iters_values)))
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (ax_ret_it, ax_ret_dp, ax_ret_cost), (ax_trace, ax_res_it, ax_res_dp) = axes
    fig.suptitle(
        f"AIF-MP convergence sensitivity (report, not a search) — "
        f"K={cfg['env']['K']}, L={cfg['env'].get('lock_length', 1)}, "
        f"probe_noise={cfg['env']['probe_noise']}, H={cfg['horizon']}, "
        f"n_seeds={cfg['n_seeds']}, n_episodes={cfg['n_episodes']}  |  "
        f"canonical cell = ({CANONICAL_CELL[0]} iterations, "
        f"damping {CANONICAL_CELL[1]})",
        fontsize=11,
    )

    # --- Top row: does behaviour move with the knobs? ---
    for damping in damp_values:
        xs = [i for i in iters_values if (i, damping) in runs]
        means, errs = zip(*(_final_return(runs[(i, damping)]) for i in xs))
        ax_ret_it.errorbar(
            xs, means, yerr=errs, color=damp_colors[damping], marker="o",
            lw=2, capsize=3, label=f"damping {damping:g}",
        )
        ax_ret_cost.errorbar(
            [costs[(i, damping)] for i in xs], means, yerr=errs,
            color=damp_colors[damping], marker="o", lw=2, capsize=3,
            label=f"damping {damping:g}",
        )

    for iters in iters_values:
        xs = [d for d in damp_values if (iters, d) in runs]
        means, errs = zip(*(_final_return(runs[(iters, d)]) for d in xs))
        ax_ret_dp.errorbar(
            xs, means, yerr=errs, color=iters_colors[iters], marker="o",
            lw=2, capsize=3, label=f"{iters} iterations",
        )

    # --- Bottom row: how close to the fixed point does the loop get? ---
    # Overlaying dampings is only legitimate because the residual is that
    # of the undamped map (see module docstring); a residual of the damped
    # step would carry a factor of alpha and make these curves unrelated.
    for damping in damp_values:
        # Trace from the longest cell at this damping — the most iterations
        # of evidence available about how far the residual decays.
        longest = max(i for i in iters_values if (i, damping) in traces)
        trace = traces[(longest, damping)]
        ax_trace.plot(
            np.arange(1, trace.size + 1), trace,
            color=damp_colors[damping], lw=2, label=f"damping {damping:g}",
        )
        xs = [i for i in iters_values if (i, damping) in traces]
        ax_res_it.plot(
            xs, [traces[(i, damping)][-1] for i in xs],
            color=damp_colors[damping], marker="o", lw=2,
            label=f"damping {damping:g}",
        )

    for iters in iters_values:
        xs = [d for d in damp_values if (iters, d) in traces]
        ax_res_dp.plot(
            xs, [traces[(iters, d)][-1] for d in xs],
            color=iters_colors[iters], marker="o", lw=2,
            label=f"{iters} iterations",
        )

    # Marks iteration 70 *within these traces*, which are the longest
    # cell's. That is not the 70-iteration cell's terminal residual (panel
    # below) — see caption note (2).
    ax_trace.axvline(
        CANONICAL_CELL[0], color=theme.PALETTE["ink"], lw=0.8, ls="--",
        alpha=0.6, label=f"iteration {CANONICAL_CELL[0]} of this trace",
    )

    for ax, xlabel, ylabel, title in (
        (ax_ret_it, "fixed-point iterations", "final-episode terminal reward",
         "Return vs iteration count"),
        (ax_ret_dp, "damping", "final-episode terminal reward",
         "Return vs damping"),
        (ax_ret_cost, "tensor-element visits per decision (nominal ceiling)",
         "final-episode terminal reward",
         "Return vs hardware-free cost ceiling"),
        (ax_trace, "iteration", "worst-conditional TV residual",
         "Residual trace (longest cell per damping)"),
        (ax_res_it, "fixed-point iterations",
         "terminal worst-conditional TV residual",
         "Terminal residual vs iteration count (per-cell)"),
        (ax_res_dp, "damping", "terminal worst-conditional TV residual",
         "Terminal residual vs damping (per-cell)"),
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    ax_ret_it.set_xscale("log")
    ax_ret_cost.set_xscale("log")
    for ax in (ax_ret_it, ax_ret_dp, ax_ret_cost):
        ax.axhline(0, color=theme.PALETTE["ink"], lw=0.5, alpha=0.4)
    ax_res_it.set_xscale("log")
    # The residual spans decades; on a linear axis the decay is invisible.
    for ax in (ax_trace, ax_res_it, ax_res_dp):
        ax.set_yscale("log")

    # Caption. Both notes are load-bearing for reading the bottom row;
    # see the module docstring for the code references.
    fig.text(
        0.5, 0.005,
        "(1) The grid varies two coupled quantities, not one: "
        "planner.iterations and planner.damping are also passed to "
        "_precompute_obs_channels, which converges the observation "
        "channels in its own loop, so the 140-iteration cell begins from "
        "more-converged obs channels than the 70-iteration cell. "
        "Differences across cells include that effect and it is not "
        "separated out here.\n"
        "(2) Bottom-left and bottom-centre/right are different "
        "quantities: the trace is the longest cell at each damping, so "
        "its value at iteration 70 is a mid-trace value of the "
        "140-iteration cell, whereas the per-cell terminal residual at 70 "
        "iterations is the end of the 70-iteration cell's own trace — a "
        "run whose obs channels also stopped at 70. They coincide only if "
        "the obs-channel loop has converged by then, which nothing here "
        "verifies.\n"
        "Residual = worst-conditional total variation between a channel "
        "and its undamped image, maximised over the two channels the main "
        "loop damps (the action channel r(u|x) and the dynamics channel "
        "r(x'|x,u)). The obs channels are not traced.",
        ha="center", va="bottom", fontsize=7.5, wrap=True,
    )
    plt.tight_layout(rect=(0, 0.10, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"figure saved to {args.output}", flush=True)

    if args.cards_dir is not None:
        named_axes = [
            ("return_vs_iterations", ax_ret_it),
            ("return_vs_damping", ax_ret_dp),
            ("return_vs_cost", ax_ret_cost),
            ("residual_trace", ax_trace),
            ("residual_vs_iterations", ax_res_it),
            ("residual_vs_damping", ax_res_dp),
        ]
        save_panel_cards(
            fig, named_axes, args.cards_dir, "aif_convergence",
            size=_CARD_SIZE_IN,
        )
        print(f"cards saved to {args.cards_dir}", flush=True)

    plt.close(fig)

    # The headline numbers, printed so they land in the DVC stage log
    # rather than only in the figure.
    print(
        "\ncell            terminal TV res   cost ceiling/dec   final return"
    )
    for iters, damping in cells:
        mean, err = _final_return(runs[(iters, damping)])
        mark = " <- canonical" if (iters, damping) == CANONICAL_CELL else ""
        print(
            f"  iter={iters:<4} damp={damping:<5g} "
            f"{traces[(iters, damping)][-1]:>10.3e} "
            f"{costs[(iters, damping)]:>15,d} "
            f"{mean:>8.3f} +/- {err:.3f}{mark}",
            flush=True,
        )


if __name__ == "__main__":
    main()
