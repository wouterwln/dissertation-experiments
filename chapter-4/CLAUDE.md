# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Chapter-4 experiment set: plan with the chapter-5 **Nuijten-MP** message-passing planner
(instead of chapter-3's gradient-based VFE optimizers) in two environments:

- **T-Maze** — the chapter-3 epistemic T-Maze (hidden reward location θ, revealed at a cue).
- **Stochastic Maze** — a noisy, partially observed 5×5 grid-world (Python/JAX port of the
  Julia `StochasticMaze` from `biaslab/EFEasVFE`, config `create_stochastic_maze(5,5,4,
  start_state=11)`): sink states, a "bridge" row of stochastic-transition cells, noisy
  location observations, ±1 reward cells. It has **no hidden context θ** (`n_theta = 1`).

The chapter additionally produces **Bethe Free Energy (BFE) convergence plots** for the
Nuijten-MP scheme on both environments.

## Commands

Always use `uv` to run Python code (never bare `python`). Add dependencies with `uv add`
(do not edit `pyproject.toml` by hand).

```bash
uv sync                                            # install dependencies
uv run python -m pytest tests/ -v                  # run tests
uv run python scripts/tmaze/experiment.py --n-episodes 50
uv run python scripts/stochastic_maze/experiment.py --n-episodes 50
uv run python scripts/convergence/run_convergence.py --environment tmaze
uv run python scripts/convergence/run_convergence.py --environment stochastic_maze
uv run python scripts/convergence/plot_convergence.py
uv run dvc repro                                   # run full DVC pipeline
```

Experiment parameters live in `params.yaml`; CLI args override them.

## What Lives Where

- `src/environments/` — `tmaze.py` copied **verbatim** from chapter 3; `stochastic_maze.py`
  is a faithful 0-indexed Python/JAX port of the Julia `StochasticMaze` (the Julia is
  1-indexed; default coords are translated, originals noted in comments).
- `src/inference/` — `messages.py`, `region_extended_loopy_bp.py`, `nuijten_mp.py` copied
  **verbatim** from chapter 5. `convergence.py` is a trimmed copy of chapter-5's convergence
  module containing only the Nuijten path (`compute_nuijten_vfe`, `nuijten_mp_convergence`).
- `src/adapters/generative_model.py` — converts each environment's tensors into the planner's
  `(transition (S,S,n_θ,A), observation (n_channels,n_obs_types,S,n_θ), goal (S,n_θ))` inputs.
  θ plays the role of the planner's `n_static`.
- `src/adapters/state_inference.py` — `infer_state`, a dense joint Bayesian `(x,θ)` filter.
- `src/agents/nuijten_agent.py` — `NuijtenAgent`: state inference + `nuijten_mp_planning` + argmax per step.
- `scripts/{tmaze,stochastic_maze}/experiment.py` — episode loops + JSON output (`results.json`, `stats.json`).
- `scripts/convergence/{run_convergence,plot_convergence}.py` — BFE convergence sweep + plots.

## Architecture / key facts

- T-Maze transitions are **θ-independent** `(S,S,A)`; the adapter broadcasts them over θ.
  The Stochastic Maze uses `n_theta = 1` (no hidden context).
- The adapter pads heterogeneous observation modalities to a uniform `n_obs_types` and stacks
  them along a channel axis (chapter-5 RockSample convention: unused outcomes get probability 0).
- T-Maze planning is fed only the **θ-informative** cue channel (`planning_observation_tensor`);
  state inference uses the **full** observation tensor. The Stochastic Maze has a single
  (θ-uninformative) location channel used for both.
- **T-Maze** uses `terminal_goal_only=True` (2D goal marginalised under the θ-belief, bound to
  the final node) **plus a receding horizon**: the terminal goal permits the epistemic cue
  detour, while the shrinking horizon forces commitment at the deadline. The chapter-3 cue-cost
  action prior (`cue_cost_epsilon < 1`) is **not** used — the MP planner has its own epistemic
  drive, so biasing toward the cue makes it loop. See [[chapter4-mp-planner-behavior]].
- **Stochastic Maze** uses a **terminal** goal (near-delta on the +1 state) +
  `terminal_goal_only=True` + receding horizon, mirroring the original `s[end] ~ goal`.
  The key mechanism is Nuijten-MP's **EFE ambiguity term**: it penalizes high-observation-
  noise states, so the planner routes along the zero-noise safe corridor (columns x=0 / x=4
  on the 5×5 grid) and avoids the stochastic interior entirely → ~100% success. Do **not**
  encode the goal as a strong per-step preference: that lets the pragmatic term swamp the
  ambiguity drive and the agent gambles through the stochastic barrier (~⅔ ceiling). The
  5×5 grid (not 4×5) is essential — the 5th column is the right-hand safe corridor.
- `nuijten_mp_planning` / `nuijten_mp_convergence` are JIT with static `(horizon, n_iterations)`.
- **Convergence study**: `run_convergence.py` records a per-iteration BFE trace at every
  planning step across episodes (fixed horizon, randomized non-sink/non-reward start states;
  it stops once the belief collapses onto an absorbing sink, where planning is degenerate).
  `plot_convergence.py` plots median ± IQR vs iteration. Output: `data/convergence_plots/`.

## Conventions

- Run Python with `uv run python`; add deps with `uv add`.
- Experiment results go to `data/<env>/nuijten/{results.json,stats.json}` (schema mirrors
  chapter 3). Convergence traces go to `data/<env>/nuijten/convergence.json`
  (+ `convergence_stats.json` metric); plots to `data/convergence_plots/`.
