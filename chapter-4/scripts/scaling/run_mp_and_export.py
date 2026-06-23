#!/usr/bin/env python3
"""Scaling study, part 1 (chapter-4 / MP side).

Times the Nuijten-MP message-passing planner across two axes:
  - HORIZON, on the T-Maze (S=5 fixed) -- the locality axis.
  - STATE SPACE, on the Stochastic Maze (square grids, horizon fixed) -- the
    O(S^2)-per-gradient-step axis where the gradient planner is expected to blow up.

For every configuration it ALSO exports the generative model in the chapter-3
gradient planner's tensor convention (transition (S,S,A), theta-obs (n_obs,S,n_theta),
goal (S,n_theta), ...) to an .npz, so the gradient planner is timed on the *same*
tensors -- not a re-derived look-alike. Run `chapter-3/scripts/scaling/run_gradient.py`
next, then `plot_scaling.py`.

Timing protocol: one warm-up call per config (triggers JIT/XLA compile, excluded),
then the median of `--repeats` wall-clock measurements, each blocked to completion.
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

from src.adapters.generative_model import (
    build_tmaze_generative_model, build_stochastic_maze_generative_model,
)
from src.inference.nuijten_mp import nuijten_mp_planning


def to_gradient_tensors(model, start_state_idx, n_theta):
    """Express a chapter-4 GenerativeModel in the chapter-3 gradient convention.

    ch4 transition is (x_new, x_old, theta, a); ch3 wants (next, prev, a). The
    T-Maze transition is theta-independent and the Stochastic Maze has n_theta=1,
    so slicing theta=0 is exact in both cases. The planning observation channel
    (theta-informative) becomes the single ch3 theta-observation modality.
    """
    transition = np.asarray(model.transition_tensor)[:, :, 0, :]        # (S,S,A)
    theta_obs = np.asarray(model.planning_observation_tensor)[0]        # (n_obs, S, n_theta)
    goal = np.asarray(model.goal)                                       # (S, n_theta)
    initial_state = np.zeros(model.n_states, dtype=np.float32)
    initial_state[start_state_idx] = 1.0
    return {
        "transition_tensor": transition.astype(np.float32),
        "theta_observation_tensor": theta_obs.astype(np.float32),
        "goal_mapping": goal.astype(np.float32),
        "action_prior": np.asarray(model.action_prior, dtype=np.float32),
        "theta_prior": np.asarray(model.theta_prior, dtype=np.float32),
        "initial_state": initial_state,
        "n_states": int(model.n_states),
        "n_actions": int(model.n_actions),
        "n_theta": int(n_theta),
    }


def time_mp(model, horizon, n_iterations, repeats):
    q_state = jnp.zeros(model.n_states).at[model.start_state_idx].set(1.0)
    q_static = jnp.asarray(model.theta_prior) / jnp.sum(jnp.asarray(model.theta_prior))

    def call():
        ad, _, _ = nuijten_mp_planning(
            q_state, q_static, model.transition_tensor,
            model.planning_observation_tensor, model.goal,
            horizon=horizon, n_iterations=n_iterations,
            action_prior=model.action_prior,
        )
        return ad

    ad = call()
    jax.block_until_ready(ad)                                   # warm-up / compile
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ad = call()
        jax.block_until_ready(ad)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3, int(jnp.argmax(ad))   # ms, first action


def main():
    p = argparse.ArgumentParser(description="MP scaling timing + gradient-tensor export")
    p.add_argument("--horizons", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12])
    p.add_argument("--grids", type=int, nargs="+", default=[5, 7, 9, 11, 13],
                   help="square grid side lengths for the state-space sweep")
    p.add_argument("--tmaze-iterations", type=int, default=20)
    p.add_argument("--stochastic-iterations", type=int, default=40)
    p.add_argument("--stochastic-horizon", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="data/scaling")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    records = []   # one per config: env, axis, axis_value, sizes, mp timing, gradient hyperparams

    # --- Horizon sweep on the T-Maze (S=5) ---
    tmaze = build_tmaze_generative_model(goal_temperature=4.5)
    for H in args.horizons:
        ms, first = time_mp(tmaze, H, args.tmaze_iterations, args.repeats)
        g = to_gradient_tensors(tmaze, tmaze.start_state_idx, n_theta=2)
        tag = f"tmaze_H{H}"
        np.savez(tensors_dir / f"{tag}.npz", **g)
        records.append({
            "tag": tag, "env": "tmaze", "axis": "horizon", "axis_value": H,
            "n_states": g["n_states"], "n_actions": g["n_actions"], "n_theta": g["n_theta"],
            "horizon": H, "mp_iterations": args.tmaze_iterations,
            "mp_ms": ms, "mp_first_action": first,
        })
        print(f"[MP] tmaze H={H:2d}  S=5   {ms:8.3f} ms  first={first}")

    # --- State-space sweep on the Stochastic Maze (horizon fixed) ---
    H = args.stochastic_horizon
    for n in args.grids:
        sm = build_stochastic_maze_generative_model(grid_size_x=n, grid_size_y=n)
        ms, first = time_mp(sm, H, args.stochastic_iterations, args.repeats)
        g = to_gradient_tensors(sm, sm.start_state_idx, n_theta=1)
        tag = f"stochastic_S{g['n_states']}"
        np.savez(tensors_dir / f"{tag}.npz", **g)
        records.append({
            "tag": tag, "env": "stochastic_maze", "axis": "n_states",
            "axis_value": g["n_states"], "n_states": g["n_states"],
            "n_actions": g["n_actions"], "n_theta": g["n_theta"],
            "horizon": H, "mp_iterations": args.stochastic_iterations,
            "mp_ms": ms, "mp_first_action": first,
        })
        print(f"[MP] stochastic {n}x{n}  S={g['n_states']:3d}  H={H}  {ms:8.3f} ms  first={first}")

    with open(out_dir / "mp_timings.json", "w") as f:
        json.dump({"records": records, "repeats": args.repeats}, f, indent=2)
    print(f"\nWrote {out_dir/'mp_timings.json'} and {len(records)} tensor sets to {tensors_dir}")


if __name__ == "__main__":
    main()
