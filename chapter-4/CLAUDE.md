# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Chapter-4 experiment set: plan in the chapter-3 epistemic environments (T-Maze and the
Reactivity/Epistemic Maze) using the chapter-5 **Nuijten-MP** message-passing planner,
instead of chapter-3's gradient-based VFE optimizers. This isolates the effect of the
planner while keeping the environments fixed.

## Commands

Always use `uv` to run Python code (never bare `python`).

```bash
uv sync                                # install dependencies
uv run python -m pytest tests/ -v      # run tests
uv run python scripts/tmaze/experiment.py --n-episodes 50
uv run python scripts/reactivity_maze/experiment.py --n-episodes 50
uv run dvc repro                       # run full DVC pipeline
```

Experiment parameters live in `params.yaml`; CLI args override them.

## What Lives Where

- `src/environments/` — `tmaze.py`, `epistemic_maze.py` copied **verbatim** from chapter 3 (tensor producers + env classes). Do not edit; re-copy from chapter 3 if they change there.
- `src/inference/` — `messages.py`, `region_extended_loopy_bp.py`, `nuijten_mp.py` copied **verbatim** from chapter 5. The three are co-located so their relative imports work.
- `src/adapters/generative_model.py` — converts each environment's chapter-3 tensors into the planner's `(transition (S,S,n_θ,A), observation (n_channels,n_obs_types,S,n_θ), goal (S,n_θ))` inputs. θ plays the role of the planner's `n_static`.
- `src/adapters/state_inference.py` — `infer_state`, a dense joint Bayesian `(x,θ)` filter copied from chapter-5's RockSample agent.
- `src/agents/nuijten_agent.py` — `NuijtenAgent`: state inference + `nuijten_mp_planning` + argmax per step.
- `scripts/{tmaze,reactivity_maze}/experiment.py` — episode loops + JSON output.

## Architecture / key facts

- Chapter-3 transitions are **θ-independent** `(S,S,A)`; the adapter broadcasts them over θ.
- Chapter-3 has **heterogeneous observation modalities**; the adapter pads each to a uniform `n_obs_types` and stacks them along a channel axis (chapter-5 RockSample convention: unused outcomes get probability 0).
- The planner is fed only the **θ-informative** channel (`planning_observation_tensor`, the cue/context channel); the location channel is θ-uninformative. State inference uses the **full** observation tensor.
- Both environments use `terminal_goal_only=True` (2D goal marginalised under the θ-belief, bound to the final node) **plus a receding horizon**. This pairing is what makes the planner solve the tasks: the terminal goal permits the epistemic cue detour, while the shrinking horizon forces commitment to the goal at the deadline. With a fixed long horizon the agent cue-seeks but then dawdles/loops and never commits (the terminal-goal action marginal is diffuse when there is horizon slack — it only commits when horizon ≈ remaining distance).
- The chapter-3 cue-cost action prior (`cue_cost_epsilon < 1`) is **not** used and is in fact harmful: the MP planner has its own epistemic drive, so biasing toward the cue makes it loop. `cue_cost_epsilon = 1.0` (uniform action prior) works best.
- `nuijten_mp_planning` is JIT with static `(horizon, n_iterations)`; the receding horizon recompiles once per distinct horizon value (a handful per episode given the small max_steps).
- Tuned `params.yaml` gives ~100% success on both environments.

## Conventions

- Run Python with `uv run python`.
- Results go to `data/<env>/nuijten/{results.json,stats.json}` (schema mirrors chapter 3 for cross-chapter comparison).
