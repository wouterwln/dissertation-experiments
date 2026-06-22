# Dissertation Experiments

This repository collects all experiments from the dissertation, one self-contained
folder per chapter. Each chapter has its **own** `uv` environment and its **own** DVC
pipeline, so any chapter's results can be reproduced independently.

## Chapter ↔ source mapping

| Chapter | Folder | Topic | Source repo |
|---------|--------|-------|-------------|
| 3 | `chapter-3/` | Epistemic priors in VFE-gradient AIF planning (T-Maze, Epistemic Maze) | `EpistemicPriorsTMazeJAX` |
| 4 | `chapter-4/` | Nuijten-MP message-passing planner in the chapter-3 epistemic environments | hand-assembled from ch.3 & ch.5 |
| 5 | `chapter-5/` | "What type of inference is active inference?" — BP variants over four environments | `MinigridMP-AIF-JAX` |
| 6 | `chapter-6/` | `LearningPOMDP` — AIF message passing for parametric Bayes-Adaptive POMDPs | `LearningPOMDP` |

Chapters were **copied** (not git submodules) from their source repos so they can be
tweaked here without affecting the originals. Chapter 4 has no single source repo: it
reuses the chapter-3 environments and the chapter-5 Nuijten-MP planner (both copied
verbatim into `chapter-4/src/`) to isolate the planner's effect. Only code and pipeline
definitions were copied; generated data, virtualenvs, and caches are excluded and
regenerated on demand.

## Reproducing everything

`reproduce-all.sh` walks the chapters in order and, for each, builds its environment
and runs the full pipeline:

```bash
./reproduce-all.sh                 # uv sync + dvc repro for every chapter, in order
./reproduce-all.sh chapter-5       # only the named chapter(s)
./reproduce-all.sh --cuda          # build GPU envs (uv sync --extra gpu); Linux+NVIDIA
./reproduce-all.sh --no-sync       # skip `uv sync` (environments already built)
./reproduce-all.sh -- --force      # forward extra args verbatim to each `dvc repro`
```

DVC is incremental, so re-running is cheap once results exist — only stages whose code
or params changed are re-run.

## GPU / CUDA

JAX is the only GPU-relevant dependency, and every chapter declares CUDA the same way:
a `gpu` **extra** that pulls `jax[cuda12]`. The mechanism is uniform across chapters:

- **CPU (default, any platform):** `uv sync` — plain `jax`/`jaxlib`, no CUDA wheels.
- **GPU (Linux + NVIDIA):** `uv sync --extra gpu`, or `./reproduce-all.sh --cuda` for all
  chapters at once.
- **macOS:** the `gpu` extra has no wheels (CUDA is Linux-only), so `--cuda` errors here by
  design — use the plain CPU sync locally. (chapter-4/chapter-5 also carry a `macos`
  dependency-group with `jax-mps` for Apple-Silicon GPU: `uv sync --group macos`.)

On the Snellius cluster, `source cluster/setup_env.sh` (chapters 3, 5, 6) is uv-native: it
builds a `.venv-gpu` from the lockfile via `uv sync --extra gpu`. Run it once on the login
node first (`FORCE_UV_SYNC=1 source cluster/setup_env.sh`) — compute nodes have no internet,
and the guard stops array tasks from re-syncing in parallel.

## Reproducing a single chapter

Each chapter folder is an independent unit. From the chapter directory:

```bash
cd chapter-3                 # or chapter-4, chapter-5, chapter-6
uv sync                      # build the chapter's virtualenv from uv.lock
uv run dvc repro             # run the full DVC pipeline -> regenerates data/ outputs
```

Notes:
- Pipeline stages invoke `uv run python ...`, so they use the chapter's own `.venv`.
- `dvc dag` shows the stage graph; `dvc status` shows what is out of date.
- chapter-3 pulls `pymdp` from a pinned git revision (see its `pyproject.toml`).
- `dvc` is a dependency of every chapter, so `uv run dvc ...` always works; a global
  `dvc` (3.67+) works too.

## Visualization is separate from the runs

Post-hoc plotting lives in its **own** DVC stage in every chapter, taking the run's
data output as a dependency. No run stage depends on visualization code or emits
figures. This is what makes re-rendering figures cheap: editing a plot script
invalidates only the plot stage, so `dvc repro` re-renders the figure without
re-running the expensive experiment.

- **chapter-6** — each run stage (`door_code`, `rock_sample`, …) has a matching
  `plot_*` stage.
- **chapter-5** — environment runs emit JSON metrics only; the figure stage
  (`convergence_sweep_plots`) depends on the run CSVs.
- **chapter-4** — runs emit `results.json` / `stats.json` only; no plotting in the pipeline.
- **chapter-3** — convergence figures are separate stages
  (`{tmaze,epistemic}_convergence_figures`), and the T-Maze episode video/TikZ frames
  are rendered by separate `tmaze_video_{marginal,active,planning}` stages (plus
  `tmaze_reference`) that reload each run's `results.json` via
  `scripts/tmaze/plot_experiment.py`. The experiment stages run with
  `--no-video --no-tikz` and depend only on their own code.

## Layout conventions

- `pyproject.toml` + `uv.lock` — the chapter's pinned environment (committed).
- `dvc.yaml` + `params.yaml` (+ `dvc.lock`) — the experiment pipeline.
- `data/`, `output/`, `logs/` — DVC-produced, git-ignored, regenerated by `dvc repro`.
- Each chapter's `.dvc/` is an independent DVC repo initialized with `dvc init --subdir`,
  so the chapters share this single git repo but keep separate pipelines and caches.
- Each chapter has its own `README.md` and `CLAUDE.md` with chapter-specific detail.
