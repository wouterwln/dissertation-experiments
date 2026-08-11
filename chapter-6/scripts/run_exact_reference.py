"""Exact Bayes-adaptive reference point on a reduced door_code / Factored Tiger.

Runs four arms on the *same* reduced instance, with the same seeds (hence
the same theta_trues) and the same total lookahead, and emits one JSON:

  exact_lowest_index   ExactBAPlanner — memoized expectimax over reachable
  exact_highest_index  action-observation histories
                       (learningpomdp/planning/exact_ba.py). No lookahead
                       knob: it plans to the end of the episode. The two
                       arms differ *only* in which member of the optimal
                       action set they act on — see "Ties" below.
  aif                  AIFPlanner, wrapped so the exact solver scores every
                       belief AIF-MP plans from without changing AIF-MP's
                       trajectory.
  bamcp                BAMCPPlanner, the sampled-model baseline.

The reported numbers, in the order they are printed and stored, weakest
claim last:

  1. ``action_agreement`` — per-decision, at a *fixed* belief: is AIF-MP's
     argmax action a member of the exact optimal set, and if not, how much
     exact action value does it forgo? This is the well-posed measurement.
     Both quantities are scored against ``action_values`` at the belief
     AIF-MP itself holds, so nothing here depends on which trajectory an
     exact arm would have taken or on how ties are broken. Since the
     membership rate *is* defined by the tie tolerance, every run re-scores
     every decision at 1e-6 / 1e-7 / 1e-8 and emits
     ``action_agreement.tie_tolerance_sweep``, so "is the agreement rate an
     artifact of the tie tolerance?" is answered by that run's own numbers
     (it is not — see ``exact_ba.optimal_action_set``). The regret is
     tolerance-free by construction.
  2. ``episode0`` — mean return over episode 0 only. This is the one place a
     *value* claim is well defined: every arm enters episode 0 at the same
     shared prior, so the exact arm's mean is an unbiased estimate of the
     exact per-episode Bayes-adaptive optimum of the very problem AIF-MP
     faces, and every tie-break rule estimates that same quantity (the two
     exact arms differ only by sampling noise here, not by construction).
  3. ``block`` — mean return over all n episodes. This compares two belief
     *trajectories*, not a policy against a bound, and it is **not** an
     optimality ratio. Read it as descriptive.

Ties, and why (3) is a band rather than a number
------------------------------------------------
Optimal actions tie structurally on both instances: on door_code every
probe station is exactly optimal (the posterior depends only on the multiset
of probe results), and on Factored Tiger LISTEN_0 / LISTEN_1 tie under a
symmetric prior. Every tied action is equally optimal *for the episode*, but
not for the *block* — which probe you visit decides which bit of θ you
learn, and that carries into the next episode's prior. Reporting one
``argmax`` tie-break as "the Bayes-adaptive optimum" would therefore divide
by an arbitrary member of the tied set. So the exact arm is run twice,
lowest- and highest-index, and (3) is published as the interval they
bracket. The block optimum is *above* both: an episode-scoped optimum never
buys information that only repays in a later episode (see the Scope section
of ``learningpomdp/planning/exact_ba.py``), so neither end of the band is a
bound and the AIF-MP gap in (3) is an underestimate of the true one.

Read ``mean_cumulative_reward`` alongside the normalised return: that is the
quantity the solver maximizes, whereas Factored Tiger's
``reward_normalised`` is a ratio whose denominator varies by seed.

Every arm is driven with ``clip_horizon=True`` and ``horizon = max_steps``,
so the runner's ``min(horizon, max_steps - step)`` hands every planner
exactly the number of decisions remaining. That is what makes the exact arm
a lookahead-parity control rather than a differently-configured planner, and
``ExactBAPlanner`` now enforces it per decision rather than trusting it.

Two confounds this stage does *not* remove, stated so they are not read as
parity:

  - **Discount.** The exact arm and AIF-MP are undiscounted (``--gamma``
    defaults to 1.0, matching the undiscounted ``cumulative_reward`` the
    runners report), while BAMCP discounts its rollouts at
    ``--bamcp-gamma`` 0.99, its canonical value everywhere else in the
    repo. So the ``bamcp``-vs-exact gap mixes approximation error with a
    small objective mismatch; the ``aif``-vs-exact gap does not. Neither
    value is changed here — this is the canonical BAMCP configuration.
  - **The instance is reduced**, and the reduction is not the canonical
    benchmark: see ``params.yaml``'s ``exact_reference`` block. Numbers
    from this stage characterise the reduced instance only.

Example:

    uv run python scripts/run_exact_reference.py \\
        --env door_code --K 3 --lock-length 2 --max-steps 6 \\
        --n-seeds 12 --n-episodes 5 \\
        --output output/exact_reference/door_code/reference.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment import door_code_runner, factored_tiger_runner
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.exact_ba import (
    _TIE_REL_TOL,
    ExactBAPlanner,
    optimal_action_set,
)

# Tie tolerances the agreement probe scores every decision at, so the headline
# rate ships with its own sensitivity check rather than an argument. 1e-7 is
# the float32 noise floor of the values themselves and 1e-8 is below it, which
# is what makes the two of them bracket `_TIE_REL_TOL`: see that constant's
# sensitivity paragraph for how the measured sweep is read.
_TIE_TOL_SWEEP = (1e-6, 1e-7, 1e-8)


def _door_code_reward_table(env: DoorCodeEnv) -> np.ndarray:
    """``(n_states, n_actions, n_static)`` expected immediate env reward.

    ``DoorCodeEnv.step`` scores the *arrival* state (+1 SUCCESS, -1 FAILURE,
    0 otherwise), so contract T over its arrival axis. Absorbing rows are
    zeroed: the runner breaks the episode on termination, so no further
    reward accrues from SUCCESS / FAILURE even though they self-loop.

    The Factored Tiger sibling of this helper already exists as
    ``factored_tiger_runner._build_factored_tiger_reward_table``.
    """
    arrival = np.zeros(env.n_states, dtype=np.float64)
    arrival[env.success_state_index] = 1.0
    arrival[env.failure_state_index] = -1.0
    table = np.einsum("nsta,n->sat", env.get_transition_tensor(), arrival)
    table[env.success_state_index] = 0.0
    table[env.failure_state_index] = 0.0
    return table


class _AgreementProbe:
    """AIF-MP acts; the exact solver scores the identical belief.

    Returns AIF-MP's own action distribution untouched, so the measured
    trajectory and return are AIF-MP's. The exact solver is queried purely
    to record, per decision, whether AIF-MP's argmax action is exactly
    optimal and how much exact value it forgoes when it is not.

    This is the measurement that ties leave well defined: it asks whether
    AIF-MP's action is *in* the exact optimal set at a belief both planners
    share, not whether it matches one arbitrary member of that set. The
    optimal set is ``exact_ba.optimal_action_set``, the same definition the
    exact arm's own tie-break uses, so the arm and the measurement cannot
    disagree about what ties. ``lowest_index_match_rate`` is the
    index-equality number alongside it — reported because it is what a naive
    ``argmax``-vs-``argmax`` comparison would produce, and the distance
    between the two rates is the size of the tie-break artifact.

    Because both planners run inside one ``plan()`` call, the runner's
    ``planning_time_s`` for this arm would be AIF-MP *plus* the exact solve.
    The two are timed separately here and ``_arm_summary`` drops the
    contaminated field — see ``aif_plan_time_s`` / ``exact_probe_time_s``.
    """

    def __init__(
        self,
        aif: AIFPlanner,
        exact: ExactBAPlanner,
        # The shipped constant itself, not a copy of its literal: the reported
        # rate has to move if `optimal_action_set`'s default is retuned.
        rel_tol: float = _TIE_REL_TOL,
        sweep_rel_tols: tuple[float, ...] = _TIE_TOL_SWEEP,
    ) -> None:
        self._aif = aif
        self._exact = exact
        self._rel_tol = rel_tol
        self._tols = tuple(sorted({*sweep_rel_tols, rel_tol}, reverse=True))
        self._n_optimal = {tol: 0 for tol in self._tols}
        self._n_lowest_match = {tol: 0 for tol in self._tols}
        self._optimal_set_size = {tol: 0 for tol in self._tols}
        self.n_decisions = 0
        self.value_regret: list[float] = []
        self.aif_plan_time_s = 0.0
        self.exact_probe_time_s = 0.0

    def prepare_episode(self, q_static: jnp.ndarray, rng) -> None:
        self._exact.prepare_episode(q_static, rng)

    def plan(
        self,
        q_state,
        q_static,
        T,
        B_list,
        goal,
        horizon,
        theta_mask=None,
        theta_spec=None,
        t_factor_spec=None,
        b_factor_specs=None,
        reward_table=None,
    ):
        _t0 = time.perf_counter()
        dist = self._aif.plan(
            q_state, q_static, T, B_list, goal, horizon,
            theta_spec=theta_spec, t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs, theta_mask=theta_mask,
        )
        # Block before stopping the clock, exactly as the runners do, so
        # AIF-MP's async compute is not charged to the exact solve below.
        if hasattr(dist, "block_until_ready"):
            dist.block_until_ready()
        self.aif_plan_time_s += time.perf_counter() - _t0

        _t1 = time.perf_counter()
        self._exact.plan(
            q_state, q_static, T, B_list, goal, horizon,
            theta_mask=theta_mask, theta_spec=theta_spec,
            t_factor_spec=t_factor_spec, b_factor_specs=b_factor_specs,
            reward_table=reward_table,
        )
        self.exact_probe_time_s += time.perf_counter() - _t1

        values = self._exact.last_action_values
        if values is not None:
            action = int(np.argmax(np.asarray(dist)))
            self.n_decisions += 1
            for tol in self._tols:
                optimal = optimal_action_set(values, rel_tol=tol)
                self._n_optimal[tol] += int(action in optimal)
                self._n_lowest_match[tol] += int(action == int(optimal[0]))
                self._optimal_set_size[tol] += int(optimal.size)
            self.value_regret.append(float(values.max()) - float(values[action]))
        return dist

    def summary(self) -> dict:
        n = max(self.n_decisions, 1)
        regret = np.asarray(self.value_regret, dtype=np.float64)
        sweep = {
            f"{tol:g}": {
                "optimal_set_rate": self._n_optimal[tol] / n,
                "lowest_index_match_rate": self._n_lowest_match[tol] / n,
                "mean_optimal_set_size": self._optimal_set_size[tol] / n,
            }
            for tol in self._tols
        }
        primary = sweep[f"{self._rel_tol:g}"]
        return {
            "n_decisions": self.n_decisions,
            # AIF-MP's action is in the exact optimal set.
            "optimal_set_rate": primary["optimal_set_rate"],
            # ... and equals its lowest-index member. A tie-break artifact,
            # not a measure of suboptimality: see the class docstring.
            "lowest_index_match_rate": primary["lowest_index_match_rate"],
            # How many actions the tolerance calls optimal per decision. The
            # discriminator for the sweep below: a rate that falls as the
            # tolerance tightens is only evidence of inflation if the set size
            # does *not* fall through the structural tie count with it.
            "mean_optimal_set_size": primary["mean_optimal_set_size"],
            "rel_tol": self._rel_tol,
            "tie_tolerance_sweep": sweep,
            # Tolerance-free by construction — max(values) - values[action]
            # never consults the tie tolerance — so one number covers the
            # whole sweep.
            "mean_value_regret": float(regret.mean()) if regret.size else 0.0,
            "max_value_regret": float(regret.max()) if regret.size else 0.0,
            # AIF-MP's own planning cost, and the probe's overhead on top of
            # it. The runner's per-episode planning_time_s for the aif arm
            # is the sum of the two, so it is dropped from that arm's summary.
            "aif_plan_time_s": self.aif_plan_time_s,
            "exact_probe_time_s": self.exact_probe_time_s,
        }


def _door_code_env_kwargs(args) -> dict:
    return dict(
        K=args.K,
        probe_noise=args.probe_noise,
        lock_length=args.lock_length,
        success_logit=args.success_logit,
        failure_logit=args.failure_logit,
    )


def _factored_tiger_env_kwargs(args) -> dict:
    return dict(
        K=args.K,
        listen_accuracy=args.listen_accuracy,
        open_safe_reward=args.open_safe_reward,
        open_tiger_penalty=args.open_tiger_penalty,
        listen_cost=args.listen_cost,
        good_logit=args.good_logit,
        bad_logit=args.bad_logit,
        max_steps=args.max_steps,
    )


def _run_arm(args, planner, desc):
    """Drive one arm through its benchmark's canonical runner."""
    if args.env == "door_code":
        return door_code_runner.run_outer_loop(
            env_kwargs=_door_code_env_kwargs(args),
            planner=planner,
            n_seeds=args.n_seeds,
            n_episodes=args.n_episodes,
            horizon=args.max_steps,
            max_steps=args.max_steps,
            deterministic=True,
            seed=args.seed,
            learn_on_failure=(args.learn_on_failure == "true"),
            clip_horizon=True,
            progress_desc=desc,
        )
    return factored_tiger_runner.run_outer_loop(
        env_kwargs=_factored_tiger_env_kwargs(args),
        planner=planner,
        n_seeds=args.n_seeds,
        n_episodes=args.n_episodes,
        horizon=args.max_steps,
        max_steps=args.max_steps,
        deterministic=True,
        seed=args.seed,
        clip_horizon=True,
        progress_desc=desc,
        workers=1,
    )


def _per_seed_returns(
    env_name: str, result, episode: int | None = None
) -> np.ndarray:
    """Each benchmark's own normalised return, one value per seed.

    door_code's ``cumulative_reward`` is already +1 / 0 / -1 per episode, so
    its mean is the normalised return and 1.0 is the ceiling. Factored Tiger
    reuses its runner's ``reward_normalised = cumulative /
    (n_safe · open_safe_reward)``, which is 1.0 when every safe door is
    opened and no tiger door is, but is unbounded below (a tiger open costs
    10x a safe one) and NaN for a θ with no safe door — hence the NaN-aware
    block average (spelled out rather than ``nanmean`` so an all-NaN seed
    stays NaN without a warning).

    ``episode=None`` averages each seed's whole block; ``episode=0`` takes
    that seed's first episode, where every arm still sits at the shared prior
    — the only slice on which the exact arm's mean bounds the others (module
    docstring, number 2). Per seed rather than pooled so the caller can put a
    standard error next to the mean; the pooled mean is unchanged, since every
    seed contributes the same number of episodes.
    """
    values = np.asarray(
        result.cumulative_reward if env_name == "door_code"
        else result.reward_normalised,
        dtype=np.float64,
    )
    if episode is not None:
        return values[:, episode]
    counts = np.isfinite(values).sum(axis=1)
    return np.where(
        counts > 0, np.nansum(values, axis=1) / np.maximum(counts, 1), np.nan
    )


def _mean_over_seeds(per_seed: np.ndarray) -> float:
    finite = per_seed[np.isfinite(per_seed)]
    return float(finite.mean()) if finite.size else float("nan")


def _sem_over_seeds(per_seed: np.ndarray) -> float:
    """Standard error of the seed mean.

    Reported next to every return because the episode-0 slice is one ±1
    outcome per seed on door_code: at ``n_seeds`` in the tens its standard
    error is comparable to the gaps being compared, which is precisely why
    the per-decision agreement — measured once per decision, not once per
    block — is the number that leads.
    """
    finite = per_seed[np.isfinite(per_seed)]
    if finite.size < 2:
        return 0.0
    return float(finite.std(ddof=1) / np.sqrt(finite.size))


def _band(exact_returns: tuple[float, float], other: float) -> dict:
    """Gap and ratio of ``other`` against the two tie-break exact arms.

    An interval, not a point, because the two arms are equally exact per
    episode and there is no principled reason to divide by either one (see
    the module docstring's Ties section). ``ratio_*`` is reported only when
    both arms earned positive return, so the ratio is meaningful at all.
    """
    lo, hi = min(exact_returns), max(exact_returns)
    out = {"gap_min": lo - other, "gap_max": hi - other}
    if lo > 0.0:
        out["ratio_min"] = other / hi
        out["ratio_max"] = other / lo
    return out


def _format_band(band: dict) -> str:
    text = f"{band['gap_min']:+.4f} .. {band['gap_max']:+.4f}"
    if "ratio_min" in band:
        text += (
            f"   (fraction of the exact arm "
            f"{band['ratio_min']:.3f} .. {band['ratio_max']:.3f})"
        )
    return text


def _mean_cumulative_reward(result) -> float:
    """The quantity the exact solver actually maximizes.

    Worth reporting next to the normalised return because Factored Tiger's
    ``reward_normalised`` is a *ratio* whose denominator varies with the
    seed's number of safe doors, so averaging it is not the solver's
    objective. For door_code the two coincide.
    """
    return float(np.mean(result.cumulative_reward))


def _exact_planner(args, tie_break: str = "lowest") -> ExactBAPlanner:
    if args.env == "door_code":
        env = DoorCodeEnv(**_door_code_env_kwargs(args))
        return ExactBAPlanner(
            max_steps=args.max_steps,
            reward_table=_door_code_reward_table(env),
            terminal_states=(env.success_state_index, env.failure_state_index),
            gamma=args.gamma,
            max_nodes=args.max_nodes,
            tie_break=tie_break,
        )
    # Factored Tiger is bounded-horizon with no absorbing state, and its
    # runner already passes a reward table into plan().
    return ExactBAPlanner(
        max_steps=args.max_steps,
        gamma=args.gamma,
        max_nodes=args.max_nodes,
        tie_break=tie_break,
    )


def _arm_summary(result, drop_planning_time: bool = False) -> dict:
    """Runner summary minus the trajectory dump (kept out to bound size).

    ``drop_planning_time`` removes ``planning_time_s`` for the aif arm, whose
    ``plan()`` also runs the exact solver: the runner times the whole call,
    so that field would report AIF-MP plus the probe under AIF-MP's name.
    The clean split lives in ``action_agreement.aif_plan_time_s`` /
    ``exact_probe_time_s``.
    """
    summary = result.to_summary_dict()
    summary.pop("trajectories", None)
    if drop_planning_time:
        summary["per_episode"].pop("planning_time_s", None)
        summary["curves"].pop("planning_time_s", None)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact Bayes-adaptive reference on a reduced instance.",
    )
    parser.add_argument(
        "--env", choices=["door_code", "factored_tiger"], required=True,
    )

    # ---- Environment (door_code) ----
    parser.add_argument("--K", type=int, default=3,
                        help="doors / symbols (door_code) or doors (tiger)")
    parser.add_argument("--lock-length", type=int, default=2,
                        help="door_code: doors to open in sequence")
    parser.add_argument("--probe-noise", type=float, default=0.2,
                        help="door_code: per-probe bit-flip probability")
    parser.add_argument("--success-logit", type=float, default=4.0,
                        help="door_code AIF goal prior on SUCCESS")
    parser.add_argument("--failure-logit", type=float, default=-4.0,
                        help="door_code AIF goal prior on FAILURE")

    # ---- Environment (factored_tiger) ----
    parser.add_argument("--listen-accuracy", type=float, default=0.85)
    parser.add_argument("--open-safe-reward", type=float, default=10.0)
    parser.add_argument("--open-tiger-penalty", type=float, default=100.0)
    parser.add_argument("--listen-cost", type=float, default=0.0)
    parser.add_argument("--good-logit", type=float, default=2.0,
                        help="factored_tiger AIF goal prior per safe open")
    parser.add_argument("--bad-logit", type=float, default=4.0,
                        help="factored_tiger AIF goal prior per tiger open")

    # ---- Runner ----
    parser.add_argument("--max-steps", type=int, default=6,
                        help="decisions per episode; also the shared "
                             "lookahead of every arm")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--learn-on-failure", type=str, choices=["true", "false"],
        default="false",
        help="door_code only; matches the canonical benchmark's 'false'",
    )

    # ---- Exact solver ----
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="exact-solver discount; 1.0 optimizes exactly "
                             "the undiscounted return the runners report")
    parser.add_argument("--max-nodes", type=int, default=200_000,
                        help="cap on expanded expectimax nodes per episode; "
                             "exceeding it raises with the count reached")

    # ---- Baseline planner hyperparams ----
    parser.add_argument("--planner-iterations", type=int, default=70)
    parser.add_argument("--planner-damping", type=float, default=0.9)
    parser.add_argument("--bamcp-n-simulations", type=int, default=10000)
    parser.add_argument("--bamcp-ucb-c", type=float, default=1.0)
    parser.add_argument("--bamcp-gamma", type=float, default=0.99,
                        help="BAMCP rollout discount; canonical 0.99, left "
                             "deliberately unequal to --gamma — see the "
                             "module docstring's confound note")

    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(
        f"env={args.env}  K={args.K}  max_steps={args.max_steps}  "
        f"n_seeds={args.n_seeds}  n_episodes={args.n_episodes}  "
        f"gamma={args.gamma}  max_nodes={args.max_nodes}",
        flush=True,
    )

    # Two exact arms, differing only in which member of the optimal action
    # set they act on. Both are exactly optimal at every decision; they
    # bracket the block return that choice moves. Their action_values — and
    # so the whole action_agreement block — are identical by construction.
    exact_planners = {
        "exact_lowest_index": _exact_planner(args, tie_break="lowest"),
        "exact_highest_index": _exact_planner(args, tie_break="highest"),
    }
    exact_results = {
        name: _run_arm(args, planner, name)
        for name, planner in exact_planners.items()
    }

    probe = _AgreementProbe(
        aif=AIFPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        ),
        exact=_exact_planner(args),
    )
    aif_result = _run_arm(args, probe, "aif")

    bamcp_result = _run_arm(
        args,
        BAMCPPlanner(
            n_simulations=args.bamcp_n_simulations,
            depth=args.max_steps,
            ucb_c=args.bamcp_ucb_c,
            gamma=args.bamcp_gamma,
            rollout_seed=args.seed,
            resample_theta_per_sim=True,
        ),
        "bamcp",
    )

    arms = {**exact_results, "aif": aif_result, "bamcp": bamcp_result}
    exact_names = tuple(exact_planners)
    block_seeds = {
        name: _per_seed_returns(args.env, result) for name, result in arms.items()
    }
    episode0_seeds = {
        name: _per_seed_returns(args.env, result, episode=0)
        for name, result in arms.items()
    }
    block = {name: _mean_over_seeds(v) for name, v in block_seeds.items()}
    episode0 = {name: _mean_over_seeds(v) for name, v in episode0_seeds.items()}
    cumulative = {
        name: _mean_cumulative_reward(result) for name, result in arms.items()
    }
    exact_block = tuple(block[name] for name in exact_names)
    exact_episode0 = tuple(episode0[name] for name in exact_names)

    payload = {
        "config": {
            "env": args.env,
            "env_kwargs": (
                _door_code_env_kwargs(args) if args.env == "door_code"
                else _factored_tiger_env_kwargs(args)
            ),
            "n_seeds": args.n_seeds,
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "gamma": args.gamma,
            "max_nodes": args.max_nodes,
            "exact_tie_breaks": list(exact_names),
            "planner_iterations": args.planner_iterations,
            "planner_damping": args.planner_damping,
            "bamcp_n_simulations": args.bamcp_n_simulations,
            "bamcp_depth": args.max_steps,
        },
        # (1) The well-posed number: per-decision, at a fixed belief,
        # tie-break-invariant. See _AgreementProbe.
        "action_agreement": probe.summary(),
        # (2) The only slice on which a value claim is a bound: every arm
        # enters episode 0 at the shared prior, so the two exact arms are two
        # estimates of the *same* quantity and their spread here is sampling
        # noise rather than a tie-break effect.
        "episode0": {
            "normalised_return": episode0,
            "normalised_return_sem": {
                name: _sem_over_seeds(v) for name, v in episode0_seeds.items()
            },
            "aif_vs_exact": _band(exact_episode0, episode0["aif"]),
            "bamcp_vs_exact": _band(exact_episode0, episode0["bamcp"]),
        },
        # (3) Descriptive: two belief *trajectories* over the whole block, not
        # a policy against a bound. The exact arms buy no cross-episode
        # information, so the block optimum is above both ends of the band.
        "block": {
            "normalised_return": block,
            "normalised_return_sem": {
                name: _sem_over_seeds(v) for name, v in block_seeds.items()
            },
            "mean_cumulative_reward": cumulative,
            "exact_tie_break_spread": abs(exact_block[0] - exact_block[1]),
            "aif_vs_exact_arm": _band(exact_block, block["aif"]),
            "bamcp_vs_exact_arm": _band(exact_block, block["bamcp"]),
        },
        "exact_solver": {
            # Per tie-break arm: they follow different histories, so their
            # trees differ in size even though their action values do not.
            "max_nodes_expanded": {
                name: p.max_nodes_expanded for name, p in exact_planners.items()
            },
            "max_nodes_cap": args.max_nodes,
            "n_plan_calls": {
                name: p.n_plan_calls for name, p in exact_planners.items()
            },
            "gamma": args.gamma,
        },
        "arms": {
            **{name: _arm_summary(result) for name, result in exact_results.items()},
            "aif": _arm_summary(aif_result, drop_planning_time=True),
            "bamcp": _arm_summary(bamcp_result),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    agreement = payload["action_agreement"]
    ep0 = payload["episode0"]
    blk = payload["block"]
    lo_name, hi_name = exact_names
    print(f"\nSummary written to {args.output}", flush=True)
    print("\n  [1] per-decision agreement — scored at a fixed belief, "
          "invariant to ties", flush=True)
    print(f"      aif action in the exact optimal set  "
          f"{agreement['optimal_set_rate']:.3f} over "
          f"{agreement['n_decisions']} decisions", flush=True)
    print(f"      ... and is its lowest-index member    "
          f"{agreement['lowest_index_match_rate']:.3f} "
          f"(tie-break artifact, not suboptimality)", flush=True)
    print(f"      mean action-value regret vs exact    "
          f"{agreement['mean_value_regret']:.4f} "
          f"(max {agreement['max_value_regret']:.4f}, tolerance-free)",
          flush=True)
    print("      tie tolerance  optimal-set rate / lowest-index rate / "
          "mean |optimal set|", flush=True)
    for tol, row in agreement["tie_tolerance_sweep"].items():
        mark = "  <- reported" if float(tol) == agreement["rel_tol"] else ""
        print(f"        {tol:<8} {row['optimal_set_rate']:.3f} / "
              f"{row['lowest_index_match_rate']:.3f} / "
              f"{row['mean_optimal_set_size']:.2f}{mark}", flush=True)
    print("\n  [2] episode-0 return — the tie-break-invariant bound "
          "(all arms at the shared prior)", flush=True)
    for name in (*exact_names, "aif", "bamcp"):
        print(f"      {name:<20} {ep0['normalised_return'][name]:+.4f} "
              f"± {ep0['normalised_return_sem'][name]:.4f} (sem over "
              f"{args.n_seeds} seeds)", flush=True)
    print(f"      aif gap {_format_band(ep0['aif_vs_exact'])}", flush=True)
    print("      Read the gap against those standard errors: one episode per "
          "seed is a thin sample, and this is the *only* slice where a value "
          "gap is a bound at all.", flush=True)
    print("\n  [3] block return — two belief TRAJECTORIES, not a bound; "
          "the block optimum is above both", flush=True)
    for name in (*exact_names, "aif", "bamcp"):
        print(f"      {name:<20} {blk['normalised_return'][name]:+.4f} "
              f"± {blk['normalised_return_sem'][name]:.4f}", flush=True)
    print(f"      exact tie-break spread {blk['exact_tie_break_spread']:.4f} "
          f"— both arms are exactly optimal per episode", flush=True)
    print(f"      aif gap vs the band "
          f"{_format_band(blk['aif_vs_exact_arm'])}", flush=True)
    print(f"      mean cumulative reward  exact "
          f"{cumulative[lo_name]:+.4f} / {cumulative[hi_name]:+.4f}   "
          f"aif={cumulative['aif']:+.4f}   bamcp={cumulative['bamcp']:+.4f}",
          flush=True)
    print(f"\n  aif arm plan time  aif={agreement['aif_plan_time_s']:.1f}s + "
          f"exact probe={agreement['exact_probe_time_s']:.1f}s "
          f"(the arm's planning_time_s is dropped: it is their sum)",
          flush=True)
    nodes = payload["exact_solver"]["max_nodes_expanded"]
    print(f"  exact solver       {nodes[lo_name]} / {nodes[hi_name]} nodes "
          f"(cap {args.max_nodes}) over "
          f"{sum(payload['exact_solver']['n_plan_calls'].values())} decisions "
          f"across the two tie-break arms", flush=True)


if __name__ == "__main__":
    main()
