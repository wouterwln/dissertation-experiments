# LearningPOMDP

Discrete Active Inference message-passing (AIF-MP) planner for
**parametric Bayes-Adaptive POMDPs**, benchmarked against an online
Bayes-Adaptive RL slate (PSRL, BAMCP / BA-POMCP, POMCP, BOSS, BeliefQ,
IDS) on discrete system-identification tasks.

The single-sentence claim:

> AIF message passing solves parametric BA-POMDPs by exploiting the
> factor structure between hidden state `x` and persistent task
> parameter `θ` — something posterior-sampling, tree-search, and
> tabular belief-feature baselines cannot do natively.

See [`roadmap.md`](roadmap.md) for paper framing, milestones, and
venue plan.

## Environments

Three discrete BA-POMDPs, each evaluated under a two-level outer
(`m` ground-truth θ samples) / inner (`n` episodes with persistent
`q(θ)` across resets) protocol:

- **Door-Code** (M1) — graph BA-POMDP. Persistent permutation
  `θ ∈ S_K`; probe / unlock task.
- **MiniGrid Door-Code** (M2) — Farama-MiniGrid spatial sibling of
  Door-Code. Same θ, partial 3×3 FOV, slip.
- **RockSample** (M4) — factorized θ split into K rock-quality
  components (episodic) and K sensor-reliability components
  (persistent).
- **Factored Tiger** — canonical-literature factored BA-POMDP (Katt
  et al. 2019, FBA-POMDP); pairs with M4 to show the
  `ThetaSpec` / `FactorSpec` machinery covers persistent-only and
  persistent + episodic θ uniformly.

## Install

`uv` only — never pip.

```bash
uv sync                       # install deps
uv run pytest                 # run tests (slow tests opt-in via `-m slow`)
```

## Run

Reproduce all canonical benchmarks + figures via DVC:

```bash
dvc repro                     # all stages
dvc repro plot                # only the door-code figure
```

Single planner via CLI (matches `dvc.yaml`):

```bash
uv run python scripts/run_door_code.py --planner aif \
    --K 4 --lock-length 2 --probe-noise 0.2 --max-steps 4 \
    --n-seeds 12 --n-episodes 30 --learn-on-failure false \
    --output /tmp/aif.json
```

Hyperparameters live in [`params.yaml`](params.yaml); edit and
`dvc repro` to rerun affected stages.

## Repo map

```
learningpomdp/
  environments/   door_code.py, minigrid_door_code.py,
                  rock_sample.py, factored_tiger.py, base.py
  planning/       {aif,vbp,psrl,belief_q,boss,bamcp,ids,
                   oracle,random}_planner.py
                  bp_core.py, messages.py — BP primitives
  inference/      state_inference.py — Bayesian filter over (x, θ)
  experiment/     door_code_runner.py — m × n outer/inner driver
  observations/   ModalityRegistry / ObservationModality
  factors.py      ThetaSpec / FactorSpec

scripts/          run_*.py CLIs, plot_*.py figures,
                  aggregate_shards.py (SLURM array stitching)
tests/            uv run pytest
cluster/          Snellius SLURM array driver
dvc.yaml          canonical pipeline
params.yaml       hyperparameters
roadmap.md        paper direction
output/           per-benchmark JSONs + figure.pdf
```

## Tensor convention (4D)

```
T:      (n_states_new, n_states_old, n_static, n_actions)
B_list: per-channel; (n_obs, n_states) if θ-independent,
        (n_obs, n_states, n_static) if θ-conditional
Goal:   (n_states, n_static)
```
