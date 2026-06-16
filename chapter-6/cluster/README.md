# Snellius (SURF) cluster runner

Submit-and-aggregate scaffolding for running the GPU-eligible JAX
planners (`aif`, `vbp`, `psrl`, `oracle`, `random`) of all five
experiment stages on Snellius `gpu_a100`. The local `dvc repro` path is
unaffected.

## One-time setup (login node)

```bash
ssh snellius
cd <path>/LearningPOMDP-snellius
source cluster/setup_env.sh    # bootstraps ~/.local/bin/uv and .venv-gpu
uv run python -c "import jax; print(jax.devices())"
```

Compute nodes have no outbound internet, so `uv sync` must succeed once
on the login node before any `sbatch`.

## Submit

| stage | submit script |
|---|---|
| door_code | `bash cluster/submit_door_code.sh` |
| minigrid_door_code | `bash cluster/submit_minigrid_door_code.sh` |
| rock_sample | `bash cluster/submit_rock_sample.sh` |
| rock_sample K-sweep | `bash cluster/submit_rock_sample_kscaling.sh` |
| factored_tiger | `bash cluster/submit_factored_tiger.sh` |
| all five (GPU + CPU) | `bash cluster/submit_all.sh` |

Each submit script:
- Reads `cluster.<stage>.n_shards` from `params.yaml`
- Reads `runner.n_seeds` (or `<stage>.runner.n_seeds`) as `--total-seeds`
- For each GPU-eligible planner, `sbatch`es one array job:
  `--array=0-(N_SHARDS-1)`, env vars `PLANNER` and `N_SHARDS`
- Skips planners whose aggregated `output/<stage>/<planner>.json`
  already exists
- After each planner's array is submitted, queues a per-planner
  aggregate job (`cluster/job_aggregate.sh`) with
  `--dependency=afterok:$JID --kill-on-invalid-dep=yes`. SLURM stitches
  that planner's shards into `output/<stage>/<planner>.json` the moment
  its array finishes successfully. If any shard fails or times out, the
  aggregate is cancelled (state: `DependencyNeverSatisfied`) so a
  partial run never writes a "valid-looking" but incomplete JSON. Only
  planners actually submitted in this invocation get an aggregate job,
  so stale shards under a skipped (already-complete) planner are never
  re-stitched.

Each array task runs one shard via
`scripts/run_<env>.py --shard-index N --n-shards M --total-seeds K`,
writing `output/<stage>/shards/<planner>.shard<NN>.json`.

The `rock_sample_kscaling` script adds an outer loop over
`rock_sample.k_sweep`, so it submits `len(k_sweep) × len(planners)`
arrays into `output/rock_sample_kscaling/K<K>/shards/`.

## Aggregate

Aggregation is normally chained automatically via SLURM dependencies
(see above). The standalone scripts remain as a manual fallback — useful
if you want to re-stitch after the dep-chained job was killed or
cancelled:

```bash
bash cluster/aggregate_all.sh                       # all stages
bash cluster/aggregate_all.sh door_code             # one stage
bash cluster/aggregate_all.sh rock_sample_kscaling  # iterates over K
```

This combines `*.shard*.json` files into per-planner JSONs that match
the non-sharded runner schema. Curves are recomputed with
`nanmean`/`nanpercentile`. Pass each per-planner JSON's parent
directory to the existing `scripts/plot_*.py` to regenerate the paper
figures.

## CPU partition (genoa)

CPU-only planners (`bamcp`, `boss`, `ids`, `pomcp`, `belief_q`,
`factored_bamcp`) are pure NumPy/Numba — they don't touch CUDA, so
they're submitted to the `genoa` partition under the separate CPU
budget. A second venv (`.venv-cpu`, built without `--extra gpu`) keeps
the CUDA jaxlib out of `sys.path`.

One-time setup (login node):

```bash
source cluster/setup_env_cpu.sh   # creates .venv-cpu, installs CPU jax
```

Submit / aggregate:

| stage | submit | per-stage planners |
|---|---|---|
| door_code | `bash cluster/submit_cpu_door_code.sh` | belief_q, boss, bamcp, pomcp, ids, random |
| minigrid_door_code | `bash cluster/submit_cpu_minigrid_door_code.sh` | belief_q, boss, bamcp, pomcp, ids, random |
| rock_sample | `bash cluster/submit_cpu_rock_sample.sh` | bamcp, boss, factored_bamcp, ids, random |
| rock_sample K-sweep | `bash cluster/submit_cpu_rock_sample_kscaling.sh` | bamcp, boss, factored_bamcp, ids, random |
| factored_tiger | `bash cluster/submit_cpu_factored_tiger.sh` | bamcp, boss, factored_bamcp, ids, random |

(For "submit everything in one go", use `bash cluster/submit_all.sh` —
it covers both the GPU and CPU stages.)

CPU submit scripts use the same self-chained aggregator (a CPU-kind
`job_aggregate.sh` queued behind the array JIDs), so no manual
aggregation step is needed. The standalone fallback:

```bash
bash cluster/aggregate_cpu_all.sh                       # all stages
bash cluster/aggregate_cpu_all.sh door_code             # one stage
bash cluster/aggregate_cpu_all.sh rock_sample_kscaling  # iterates over K
```

Parallelism model (set in `params.yaml` → `cluster.cpu.<stage>`):

- `door_code`, `minigrid_door_code`: 1 core per task, `n_shards = n_seeds`
  (the runners don't accept `--workers`).
- `rock_sample`, `rock_sample_kscaling`, `factored_tiger`: 1 shard,
  `--workers=N` spreading seeds across N cores in a single SLURM task.

CPU jobs write to the same `output/<stage>/shards/<planner>.shard*.json`
path the GPU pipeline uses, so per-planner outputs end up at
`output/<stage>/<planner>.json` and the existing `scripts/plot_*.py`
pick them up unchanged.

## What's deliberately omitted

- Each shard is an independent RNG stream (`shard_seed = base_seed +
  shard_index * 1_000_003`). The aggregated output is statistically
  equivalent to a single un-sharded run but **not bit-identical** to it.
- No DVC integration. The cluster path writes JSONs that any plot
  script can consume, but is not a tracked DVC stage. Add a
  `cluster_<stage>` stage if/when reproducibility-through-DVC matters.

## Layout

```
cluster/
  setup_env.sh                       # GPU venv (.venv-gpu, --extra gpu)
  setup_env_cpu.sh                   # CPU venv (.venv-cpu, no extras)
  job_door_code.sh                   # GPU SLURM template, one task per shard
  job_minigrid_door_code.sh
  job_rock_sample.sh
  job_rock_sample_kscaling.sh
  job_factored_tiger.sh
  job_cpu_door_code.sh               # CPU SLURM templates (genoa)
  job_cpu_minigrid_door_code.sh
  job_cpu_rock_sample.sh
  job_cpu_rock_sample_kscaling.sh
  job_cpu_factored_tiger.sh
  submit_door_code.sh                # GPU: iterates planners, sbatch one array each
  submit_minigrid_door_code.sh
  submit_rock_sample.sh
  submit_rock_sample_kscaling.sh     # double-loops K × planner
  submit_factored_tiger.sh
  submit_cpu_door_code.sh            # CPU siblings of the above
  submit_cpu_minigrid_door_code.sh
  submit_cpu_rock_sample.sh
  submit_cpu_rock_sample_kscaling.sh
  submit_cpu_factored_tiger.sh
  submit_all.sh                      # ten-in-one (all GPU + CPU stages)
  job_aggregate.sh                   # SBATCH wrapper, chained via afterany
  aggregate_all.sh                   # manual GPU aggregation fallback
  aggregate_cpu_all.sh               # manual CPU aggregation fallback
  README.md                          # you are here
```
