#!/usr/bin/env python3
"""Scaling study, part 2 (chapter-3 / gradient side).

Times the gradient-based VFE planners on the SAME generative tensors that the
chapter-4 MP side timed and exported. Reads the config list from chapter-4's
`mp_timings.json`, loads each exported tensor `.npz`, and records the gradient
planner's per-call wall-clock.

Two planners, chosen per environment to expose the axis each study sweeps:

  - **T-Maze (horizon axis)** -> the FACTORIZED planner (`plan_actions_factorized`).
    It enumerates full action/state/obs sequences, so its cost is O((S*A*n_obs)^H) --
    exponential in the planning horizon. This is exactly the locality axis where MP
    (linear in H) is expected to win, so it is the honest worst case to time. Because
    the cost blows up combinatorially, configs above `--factorized-max-horizon` are
    SKIPPED (and logged + recorded -- never silently dropped).

  - **Stochastic Maze (state-space axis)** -> the TEMPORAL planner
    (`plan_actions_temporal`, Adam on a reactive policy). Its Markovian/Bethe
    factorization is O(H*S^2) per step -- the gentle baseline appropriate for the
    state-space sweep, where full enumeration would be hopeless at S>=25.

Run from the chapter-3 directory (its env has optax):
    uv run python scripts/scaling/run_gradient.py \
        --tensors-dir ../chapter-4/data/scaling/tensors \
        --mp-timings ../chapter-4/data/scaling/mp_timings.json \
        --out ../chapter-4/data/scaling/gradient_timings.json

Timing protocol mirrors the MP side: one warm-up call per config (excluded), then
the median of `--repeats` blocked wall-clock measurements.
"""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.planning.temporal_optimizer import plan_actions_temporal, TemporalPlanningConfig
from src.planning.factorized_optimizer import plan_actions_factorized, FactorizedPlanningConfig

# The exported T-Maze planning channel is the cue, zero-padded to n_obs_types=5 by
# the chapter-4 adapter; only the first 2 outcomes are real (the rest carry
# probability 0). The factorized planner enumerates n_obs^H obs sequences, so we
# slice back to the true cardinality to avoid paying 5^H for nothing.
TMAZE_CUE_N_OBS = 2


def time_temporal(g, horizon, n_opt, lr, mode, repeats):
    initial_state = jnp.asarray(g["initial_state"])
    transition = jnp.asarray(g["transition_tensor"])
    theta_obs = jnp.asarray(g["theta_observation_tensor"])
    goal = jnp.asarray(g["goal_mapping"])
    action_prior = jnp.asarray(g["action_prior"])
    theta_prior = jnp.asarray(g["theta_prior"])
    cfg = TemporalPlanningConfig(
        planning_horizon=int(horizon), n_states=int(g["n_states"]),
        n_actions=int(g["n_actions"]), n_theta=int(g["n_theta"]), n_obs=1,
        n_optimization_steps=int(n_opt), learning_rate=float(lr), inference_mode=mode,
    )

    def call():
        return plan_actions_temporal(
            initial_state, transition, theta_obs, goal, action_prior, theta_prior, cfg,
        )

    r = call()
    jax.block_until_ready(r.q_first_action)                     # warm-up / compile
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = call()
        jax.block_until_ready(r.q_first_action)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3, int(jnp.argmax(r.q_first_action))


def time_factorized(g, horizon, n_opt, lr, mode, repeats):
    prior_state = jnp.asarray(g["initial_state"])
    transition = jnp.asarray(g["transition_tensor"])                       # (S, S, A)
    theta_obs = jnp.asarray(g["theta_observation_tensor"])                 # (n_obs_pad, S, n_theta)
    n_obs = min(TMAZE_CUE_N_OBS, int(theta_obs.shape[0]))
    observation = theta_obs[:n_obs]                                        # (n_obs, S, n_theta)
    goal = jnp.asarray(g["goal_mapping"])                                  # (S, n_theta)
    theta_prior = jnp.asarray(g["theta_prior"])
    cfg = FactorizedPlanningConfig(
        planning_horizon=int(horizon), n_obs=n_obs,
        n_states=int(g["n_states"]), n_actions=int(g["n_actions"]),
        n_theta=int(g["n_theta"]), n_optimization_steps=int(n_opt),
        learning_rate=float(lr), inference_mode=mode,
    )

    def call():
        return plan_actions_factorized(
            prior_state, theta_prior, transition, observation, goal, cfg,
        )

    r = call()
    jax.block_until_ready(r.first_action_probs)                 # warm-up / compile
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = call()
        jax.block_until_ready(r.first_action_probs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3, int(jnp.argmax(r.first_action_probs))


def main():
    p = argparse.ArgumentParser(description="Gradient-planner scaling timing")
    p.add_argument("--tensors-dir", type=str, required=True)
    p.add_argument("--mp-timings", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    # Temporal planner (Stochastic Maze, state-space axis).
    p.add_argument("--n-optimization-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--inference-mode", type=str, default="planning",
                   choices=["marginal", "active", "planning"])
    # Factorized planner (T-Maze, horizon axis).
    p.add_argument("--factorized-opt-steps", type=int, default=1500,
                   help="optimization steps for the factorized planner (chapter-3 T-Maze default)")
    p.add_argument("--factorized-lr", type=float, default=0.05)
    p.add_argument("--factorized-mode", type=str, default="planning",
                   choices=["marginal", "active", "planning"])
    p.add_argument("--factorized-max-horizon", type=int, default=5,
                   help="skip factorized timing above this horizon (enumeration is O(S^H); "
                        "skipped configs are logged + recorded, not silently dropped)")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    tensors_dir = Path(args.tensors_dir)
    mp = json.load(open(args.mp_timings))

    records = []
    skipped = []
    for rec in mp["records"]:
        g = dict(np.load(tensors_dir / f"{rec['tag']}.npz"))

        if rec["env"] == "tmaze":
            if rec["horizon"] > args.factorized_max_horizon:
                skipped.append({"tag": rec["tag"], "horizon": rec["horizon"],
                                "reason": f"horizon > factorized_max_horizon "
                                          f"({args.factorized_max_horizon}); enumeration intractable"})
                print(f"[SKIP] {rec['tag']:18s} S={rec['n_states']:3d} H={rec['horizon']:2d}  "
                      f"factorized enumeration capped at H={args.factorized_max_horizon}")
                continue
            planner = "factorized"
            ms, first = time_factorized(
                g, rec["horizon"], args.factorized_opt_steps,
                args.factorized_lr, args.factorized_mode, args.repeats,
            )
            opt_steps, mode, lr = args.factorized_opt_steps, args.factorized_mode, args.factorized_lr
        else:
            planner = "temporal"
            ms, first = time_temporal(
                g, rec["horizon"], args.n_optimization_steps, args.learning_rate,
                args.inference_mode, args.repeats,
            )
            opt_steps, mode, lr = args.n_optimization_steps, args.inference_mode, args.learning_rate

        records.append({
            "tag": rec["tag"], "env": rec["env"], "axis": rec["axis"],
            "axis_value": rec["axis_value"], "n_states": rec["n_states"],
            "horizon": rec["horizon"], "grad_planner": planner,
            "grad_opt_steps": opt_steps, "grad_mode": mode, "grad_lr": lr,
            "grad_ms": ms, "grad_first_action": first,
        })
        print(f"[{planner.upper()[:4]}] {rec['tag']:18s} S={rec['n_states']:3d} H={rec['horizon']:2d}  "
              f"{ms:9.2f} ms  first={first}")

    with open(args.out, "w") as f:
        json.dump({"records": records, "skipped": skipped, "repeats": args.repeats,
                   "factorized_max_horizon": args.factorized_max_horizon}, f, indent=2)
    note = f"  ({len(skipped)} skipped: {[s['tag'] for s in skipped]})" if skipped else ""
    print(f"\nWrote {args.out}  ({len(records)} configs){note}")


if __name__ == "__main__":
    main()
