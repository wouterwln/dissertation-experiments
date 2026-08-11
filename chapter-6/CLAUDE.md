# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

Discrete Active Inference message-passing planner for parametric
Bayes-Adaptive POMDPs, benchmarked against an online BA-RL baseline
slate on a door-code system-ID task. See `roadmap.md` for the paper
plan.

## Repo map

```
learningpomdp/
  environments/   door_code.py, minigrid_door_code.py, base.py
  planning/       {aif,vbp,psrl,belief_q,boss,bamcp,oracle,
                   random}_planner.py; bamcp also serves the literal-
                   POMCP ablation via resample_theta_per_sim=False;
                   bp_core.py, messages.py — BP primitives
  inference/      state_inference.py — Bayesian filter over (x, θ)
  experiment/     door_code_runner.py — m × n outer/inner loop driver
  observations/   ModalityRegistry / ObservationModality

scripts/
  run_door_code.py            canonical CLI (graph BA-POMDP)
  run_minigrid_door_code.py   spatial sibling CLI
  plot_door_code.py           paper figure
  plot_minigrid_door_code.py
  plot_environments.py        static benchmark schematics (no run data)
  _cards.py                   per-panel PDF cards (--cards-dir)

tests/                        uv run pytest
dvc.yaml, params.yaml         canonical pipeline + hyperparams
dissertation_theme.py         figure colors/rcParams — single source of
                              truth, shared with DissertationExperiments
                              chapter-6; a dep of every plot stage
roadmap.md                    paper direction (parametric BA-POMDP)
output/                       per-benchmark JSONs + figure.pdf + cards/
```

## Commands

```bash
uv sync                       # install deps
uv run pytest                 # run tests
dvc repro                     # all canonical stages
dvc repro plot                # only the plot stage
```

Single planner via CLI (matches `dvc.yaml`):

```bash
uv run python scripts/run_door_code.py --planner aif \
    --K 4 --lock-length 2 --probe-noise 0.2 --max-steps 4 \
    --n-seeds 12 --n-episodes 30 --learn-on-failure false \
    --output /tmp/aif.json
```

## Tensor convention (4D, from MinigridMP)

```
T:      (n_states_new, n_states_old, n_static, n_actions)
B_list: per-channel; (n_obs, n_states) if θ-independent,
        (n_obs, n_states, n_static) if θ-conditional
Goal:   (n_states, n_static)
```

## Constraints

- **Planners**: only inference-based or RL — see `roadmap.md` for the
  BA-RL slate. No greedy / heuristic stubs.
- **Package manager**: `uv` only, never pip.
- **Run scripts**: always `uv run`, never `python` / `python3`.
- **Experiments**: use `dvc repro`; add new ablations as DVC stages,
  not one-off `python -c` scripts.
- **Roadmap requests**: write the doc, don't start coding unless asked.
- **Cross-repo**: only the repos named here; don't substitute siblings.


Any code changes you make in this repo will be reviewed by Codex, which prompts me to assess which product is more worth the subscription fee.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
