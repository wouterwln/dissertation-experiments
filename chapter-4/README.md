# Nuijten-MP Planning in Epistemic Environments (JAX) — Dissertation Chapter 4

Takes the two epistemic environments from chapter 3 — the **T-Maze** and the
**Reactivity (Epistemic) Maze** — and plans in them with the **Nuijten-MP
message-passing planner** from chapter 5, instead of chapter 3's gradient-based
variational free energy (VFE) optimizers. The point is to show that message-passing
planning (region beliefs with EFE-based action priors) produces the same
goal-directed, cue-seeking behaviour in these environments.

## Environments

### T-Maze
5-state T-shape; reward at the left or right arm (latent context θ). Visiting the
cue state (bottom) reveals the reward location. Location is perfectly observed.

### Reactivity (Epistemic) Maze
The chapter-3 "Minimal Reactive Epistemic Maze": 35 states (7 locations × 5
reactivity-knob values), 8 actions. Reward is delivered only at the terminal step.
A costly cue action reveals the goal context θ; a reactivity knob governs action
success probability. Location is observed noisily.

## How it works

Both environments are originally defined by chapter-3 tensors with **θ-independent
transitions** and **heterogeneous observation modalities**. `src/adapters/generative_model.py`
converts them into the format the Nuijten-MP planner expects:

- transitions `(S, S, A)` → `(S, S, n_θ, A)` (broadcast over θ);
- observation modalities → a single `(n_channels, n_obs_types, S, n_θ)` tensor
  (padding unused outcomes to probability 0, following the chapter-5 RockSample
  convention);
- the goal mapping `(S, n_θ)` is used directly as a per-config preference `C(x, θ)`.

The planner is fed only the **θ-informative** observation channel (the cue/context
channel); full state inference (`src/adapters/state_inference.py`, a dense joint
Bayesian filter copied from chapter-5's RockSample agent) uses all channels.
`src/agents/nuijten_agent.py` ties inference and planning together per step.

Both environments use a **terminal-only** goal (the 2D preference is marginalised
against the current θ-belief and bound to the final node) together with a
**receding horizon**. This combination is what makes the planner solve the tasks:
the terminal goal lets it choose the epistemic cue detour, and the shrinking
horizon forces it to commit to the goal as the deadline approaches. Note that the
chapter-3 cue-cost action prior (`cue_cost_epsilon < 1`) is *not* used — the MP
planner has its own epistemic drive, and biasing toward the cue makes it loop;
`cue_cost_epsilon = 1.0` (uniform) works best. With the tuned `params.yaml` both
environments reach ~100% success. See `params.yaml` for the rationale.

## Usage

```bash
uv sync

# T-Maze
uv run python scripts/tmaze/experiment.py --n-episodes 50

# Reactivity maze
uv run python scripts/reactivity_maze/experiment.py --n-episodes 50

# Tests
uv run python -m pytest tests/ -v

# Full DVC pipeline (both environments)
uv run dvc repro
```

Experiment parameters live in `params.yaml`; CLI args override them.

## Structure

```
src/
├── environments/   tmaze.py, epistemic_maze.py        # copied verbatim from chapter 3
├── inference/      messages.py, region_extended_loopy_bp.py, nuijten_mp.py  # copied from chapter 5
├── adapters/       generative_model.py, state_inference.py  # chapter-3 tensors → planner inputs
└── agents/         nuijten_agent.py                   # inference + planning per step
scripts/
├── tmaze/experiment.py
└── reactivity_maze/experiment.py
tests/
├── test_adapter_shapes.py
└── test_smoke_episode.py
data/                # DVC-tracked results (results.json + stats.json per environment)
```

## DVC Pipeline

```bash
uv run dvc repro                       # both stages
uv run dvc repro -s tmaze_nuijten      # single stage
```
