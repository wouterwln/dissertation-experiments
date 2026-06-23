#!/usr/bin/env python3
"""Scaling study, part 2 (chapter-3 / gradient side).

Times the gradient-based temporal-VFE planner (`plan_actions_temporal`, Adam on a
reactive policy) on the SAME generative tensors that the chapter-4 MP side timed
and exported. Reads the config list from chapter-4's `mp_timings.json`, loads each
exported tensor `.npz`, and records the gradient planner's per-call wall-clock.

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


def time_gradient(g, horizon, n_opt, lr, mode, repeats):
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


def main():
    p = argparse.ArgumentParser(description="Gradient-planner scaling timing")
    p.add_argument("--tensors-dir", type=str, required=True)
    p.add_argument("--mp-timings", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-optimization-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--inference-mode", type=str, default="planning",
                   choices=["marginal", "active", "planning"])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    tensors_dir = Path(args.tensors_dir)
    mp = json.load(open(args.mp_timings))

    records = []
    for rec in mp["records"]:
        g = dict(np.load(tensors_dir / f"{rec['tag']}.npz"))
        ms, first = time_gradient(
            g, rec["horizon"], args.n_optimization_steps, args.learning_rate,
            args.inference_mode, args.repeats,
        )
        records.append({
            "tag": rec["tag"], "env": rec["env"], "axis": rec["axis"],
            "axis_value": rec["axis_value"], "n_states": rec["n_states"],
            "horizon": rec["horizon"], "grad_opt_steps": args.n_optimization_steps,
            "grad_mode": args.inference_mode, "grad_ms": ms, "grad_first_action": first,
        })
        print(f"[GRAD] {rec['tag']:18s} S={rec['n_states']:3d} H={rec['horizon']:2d}  "
              f"{ms:9.2f} ms  first={first}")

    with open(args.out, "w") as f:
        json.dump({"records": records, "repeats": args.repeats,
                   "n_optimization_steps": args.n_optimization_steps,
                   "inference_mode": args.inference_mode}, f, indent=2)
    print(f"\nWrote {args.out}  ({len(records)} configs)")


if __name__ == "__main__":
    main()
