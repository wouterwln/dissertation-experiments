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
| AIF convergence sweep | `bash cluster/submit_aif_convergence.sh` |
| everything (GPU + CPU + ablations) | `bash cluster/submit_all.sh` |

`submit_aif_convergence.sh` double-loops
`aif_convergence.iterations_sweep` × `aif_convergence.damping_sweep`,
submitting one array per cell into
`output/aif_convergence/iter<N>_damp<D>/shards/`. AIF only — the
sample-based baselines have no message-passing fixed point to sweep.

`submit_all.sh` runs `cluster/preflight.py` first and aborts if any
`dvc.yaml` run-stage has no submit script, or has one it never calls.
That check exists because both failures had already happened silently:
`door_code_simscaling` had a submit script that `submit_all.sh` never
invoked, and `door_code_preference` had none at all — so "submit
everything" did not. Run it standalone any time with
`uv run python cluster/preflight.py`. Stages needing no cluster job go in
its `LOCAL_ONLY` set; stages whose scripts are still being written go in
`PENDING`, which downgrades every finding for that stage to a warning —
an entry left there is a stage that will not run. Every key in those maps
must name a real `dvc.yaml` stage, so a typo cannot hide as a dead
exemption, and a planner name the gate cannot place on a partition is a
hard failure rather than an empty requirement (an unplaceable name used
to skip the checks and report the stage as fine).

Goal-prior preference sweeps (AIF only, so GPU-only — there is no
CPU-only planner in these stages and hence no `submit_cpu_*` sibling and
no `cluster.cpu.<stage>` params entry). AIF-only because
BAMCP/POMCP/Factored-BAMCP/BOSS/IDS/Belief-Q/Random never read the goal
prior C and plan on the env reward table instead; PSRL and Oracle *do*
read C (both wrap VBP), but are left out of the sweeps to bound GPU cost,
so the swept cells have no baseline counterpart:

| stage | submit script |
|---|---|
| door_code_preference | `bash cluster/submit_door_code_preference.sh` |
| rock_sample_preference | `bash cluster/submit_rock_sample_preference.sh` |
| factored_tiger_preference | `bash cluster/submit_factored_tiger_preference.sh` |

Each loops its cells from `params.yaml` — the full `(good, bad)` /
`(succ, fail)` grid for factored_tiger / door_code, the enumerated 6-cell
cross in `rock_sample_preference.cells` for RockSample — and sbatches one
array per cell into
`output/<stage>/<cell>/shards/aif.shard*.json`. `submit_all.sh` calls all
three, and holds each one's plot job until the canonical baselines it
draws as reference levels have aggregated.

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

## Calibration pilot (price the slate before running it)

```bash
bash cluster/submit_pilot.sh                       # ~140 SBU, 17 jobs
uv run python scripts/price_slate.py \
    --results-dir output/_pilot --results-dir output
```

One reduced-scale shard per (benchmark, planner) cell, submitted through the
real job templates so the timings are in the right units, then
`scripts/price_slate.py` turns them into a slate total in SBU. Cells the
pilot skips as immaterial fall back to the older run — hence the second
`--results-dir`; without it they would drop out of the total rather than be
priced.

It reduces SEEDS, never EPISODES, and writes to `output/_pilot/` rather than
`output/<stage>/`, via the `OUT_DIR` / `SEEDS_OVERRIDE` optional env pair the
eight canonical job templates now honour. `submit_pilot.sh` documents why
both of those matter, and queues no aggregate jobs on purpose: an aggregated
JSON under `output/<stage>/` is what every `submit_*.sh` checks to decide a
planner is already done.

`sacct -j <ids> --format=JobName%30,Elapsed,AllocCPUS,State` gives the
billed elapsed time. Comparing it against the summed `planning_time_s` in
the pilot JSONs is what replaces the model's one remaining guess, its
per-shard startup allowance.

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
| AIF timing anchor | `bash cluster/submit_cpu_aif_timing.sh` | aif |

The AIF timing anchor is the one CPU stage that runs a *GPU-eligible*
planner. It exists so the compute table has an AIF seconds-per-decision
column measured on the same hardware family as the CPU-run tree-search
baselines — the table otherwise compares an A100 against EPYC cores. It
runs at reduced seeds/episodes: the output is a *timing*, and its returns
are not a result. **Nothing in this repo reads its JSON yet** — the table
emitter is the intended consumer and does not read it, so the file is
currently written and not read. That is why `submit_all.sh` leaves this
one stage out: see `preflight.py`'s `DEFERRED` entry, and delete it once a
consumer lands.

Because that column only means anything against a matched baseline,
`job_cpu_aif_timing.sh` takes what `job_cpu_door_code.sh` gives each
*shard* and nothing more: `--partition=genoa`, `JAX_PLATFORMS=cpu` as the
single environment override, and ~2 cores per process. What matches is now
the per-shard core budget rather than a literal identical `--cpus-per-task`:
the baselines run packed (`pack = 10` shards in one 24-core allocation, so
2.4 cores per process — see *Why blocks, not workers*), the anchor runs
unpacked on 2. The baselines are single-threaded NumPy and use one core of
whatever they're given, so 2 against 2.4 does not move their
seconds-per-decision. The residual is L3/memory-bandwidth contention
between a block's shards, which can only make the *baselines* look slower
and therefore cannot manufacture the conclusion the table wants. Neither
script caps thread counts. Matching is what matters, and matching by *removal*
is free: DVC does not track env vars and `submit_cpu_door_code.sh` skips
planners whose JSON already exists, so a cap added to the baselines only
takes effect on a rerun — which would leave AIF pinned to one thread
against unpinned baselines, biased against the method the table exists to
defend. What remains is a method asymmetry, not a configuration one: AIF's
timed region is XLA and may use both allocated cores while the tree-search
baselines' is pure NumPy/Python and uses one. That is bounded by the
anchor's 2-core allocation and it favours AIF, so it cannot manufacture the
conclusion the table wants. Every run records the regime it measured in
`config.runtime` (backend, affinity-mask size, thread env vars, warm-up
flag), so a reader can tell which regime produced a number. Change the two
scripts together or the table stops being comparable.

One seed per shard (`cluster.cpu.aif_timing.n_shards` = 10 against
`aif_cpu_timing.n_seeds` = 10) is for measurement isolation, not
throughput: it is **not** free. Each process pays its own XLA compiles,
and under `clip_horizon` the planner jits once per distinct effective
horizon — up to 8 at the canonical `max_steps=16, horizon=8`. That is why
the stage passes `--warmup-planner`: it runs one discarded `plan()` per
distinct horizon before the first timed decision, so the compiles land
outside `planning_time_s` in both the sharded cluster path and the
single-process `dvc repro` path. Without it the two paths measure
materially different things (10 compiles against one seed of work each,
versus one compile amortized over 10 seeds).

(For "submit everything in one go", use `bash cluster/submit_all.sh` — it
covers the five experiment stages on both partitions plus every ablation,
including `aif_convergence`. The AIF timing anchor is the one exception,
for the consumer reason above; submit it with its own script.)

CPU submit scripts use the same self-chained aggregator (a CPU-kind
`job_aggregate.sh` queued behind the array JIDs), so no manual
aggregation step is needed. The standalone fallback:

```bash
bash cluster/aggregate_cpu_all.sh                       # all stages
bash cluster/aggregate_cpu_all.sh door_code             # one stage
bash cluster/aggregate_cpu_all.sh rock_sample_kscaling  # iterates over K
```

Parallelism model (set in `params.yaml` → `cluster.cpu.<stage>`):

- `door_code`, `minigrid_door_code`, `door_code_{depth,rollout,simscaling}`:
  `n_shards = n_seeds`, and each SLURM array task runs a **block** of
  `pack` shards concurrently as separate processes. The runners don't
  accept `--workers`; see *Why blocks, not workers* below.
- `rock_sample`, `rock_sample_kscaling`, `factored_tiger`: 1 shard,
  `--workers=N` spreading seeds across N cores in a single SLURM task.
- `aif_timing`: 1 shard per task, unpacked, `max_concurrent` kept low —
  the output is a timing, so isolation beats throughput.

`max_concurrent` throttles the SLURM array, so for a packed stage it counts
**blocks**: what runs at once is `max_concurrent × pack` processes. Packing
dropped the block count below the old value and the throttle stopped
binding — `door_code` went from 100 shards at `%20` (20 processes per
planner) to 10 blocks at `%20`, which is no limit at all, so all six
planners ran 100 processes each. Every process imports the whole
jax/numpy/numba stack out of `.venv-cpu`, and genoa compute nodes are
diskless (`/scratch-local` resolves to GPFS, not local disk), so that was
~600 simultaneous walks of a 15k-file venv against one shared metadata
server: the storm `setup_env_cpu.sh` warns about, which got every wave
after the first killed on allocation. The values are now `2`, restoring the
pre-packing 20 processes per planner. Drop to `1` if jobs are still killed;
it costs wall-clock only, since genoa bills elapsed per job and throttling
changes neither the job count nor any job's elapsed.

Relatedly, the packed job scripts read `params.yaml` through
`cluster/_read_params.sh`, which loads it once per SLURM task instead of
running one `uv run python -c` per CLI flag — that was ~24 venv walks per
block before a single shard started.

### Why blocks, not workers

genoa bills `max(cpus-per-task, ceil(mem_MiB / 1792), 24)` per **job**. The
24 is a 1/8-node floor: a 192-core node is shared by at most 8 jobs, and no
flag gets under it. A single-seed shard asking for 2 cores was therefore
billed for 24 it never touched, 1575 times across the door_code family —
about 13,200 SBU to do roughly 1,100 core-hours of work.

Filling those cores with `--workers` would mean cutting `n_shards`, and
that changes results: every per-seed stream is `f(shard_seed, s)` with
`shard_seed = seed + shard_index * 1_000_003`, so a different shard count
gives every seed a different environment stream, a different MCTS stream
and a different `theta_true` draw. Packing instead leaves `n_shards`,
`shard_index` and every argument alone and runs `pack` of the *unchanged*
invocations in one allocation. The shards are the same processes writing
the same files, so their JSONs are byte-identical to the unpacked run —
only `planning_time_s`, being wall-clock, can move. Verified by running a
shard set one-at-a-time and packed and diffing.

`pack` is bounded by **memory**, not cores: 24 billed cores buys
24 × 1792 MiB = 42 GiB, and exceeding it raises the bill above 24 rather
than being refused. The values in `params.yaml` leave headroom over each
job script's per-shard `--mem` — `rock_sample` and `factored_tiger` both
had to cut `workers` after OOMs, so that headroom is not theoretical. The
job scripts also run each shard under `uv run --frozen --no-sync`, which
skips project resolution: `pack` simultaneous resolutions would hit the
project dir with exactly the metadata storm `setup_env_cpu.sh` warns about.

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
  job_aif_convergence.sh             # GPU, one task per (iters, damping) shard
  job_door_code_preference.sh        # goal-prior sweeps (GPU, aif only)
  job_rock_sample_preference.sh
  job_factored_tiger_preference.sh
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
  submit_aif_convergence.sh          # double-loops iterations × damping
  submit_door_code_preference.sh     # double-loops the 2-D logit grid
  submit_rock_sample_preference.sh
  submit_factored_tiger_preference.sh
  submit_cpu_door_code.sh            # CPU siblings of the above
  submit_cpu_minigrid_door_code.sh
  submit_cpu_rock_sample.sh
  submit_cpu_rock_sample_kscaling.sh
  submit_cpu_factored_tiger.sh
  job_cpu_aif_timing.sh              # matched-hardware AIF timing anchor
  submit_cpu_aif_timing.sh
  submit_all.sh                      # everything (GPU + CPU + ablations)
  submit_pilot.sh                    # calibration pilot -> scripts/price_slate.py
  preflight.py                       # stage-coverage gate for submit_all
  job_aggregate.sh                   # SBATCH wrapper, chained via afterany
  aggregate_all.sh                   # manual GPU aggregation fallback
  aggregate_cpu_all.sh               # manual CPU aggregation fallback
  README.md                          # you are here
```
