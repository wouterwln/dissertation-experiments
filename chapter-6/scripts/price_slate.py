"""Price the cluster slate in SBU from measured per-seed planning time.

    uv run python scripts/price_slate.py            # from output/
    uv run python scripts/price_slate.py \
        --results-dir output/_pilot --results-dir output   # after the pilot

The consumer of `cluster/submit_pilot.sh`: the pilot measures one
reduced-scale shard per (benchmark, planner) cell on the real partitions,
and this turns those measurements into a slate total.

Rates. gpu_a100 bills a quarter node (1 GPU + 18 cores); genoa bills per
requested core. Charged on ELAPSED, so a generous --time costs queue
priority but no credits. Cross-checks against the two figures already in the
repo: preflight.py's "aif_cpu_timing <=44 SBU (10 genoa tasks x 2 cores x
the 2 h cap)" is exactly 10*2*2*1.1, and this model puts RockSample AIF at
~6.1 h/shard against the 6.05 h job_rock_sample.sh records as measured.

Per-seed cost is the MEDIAN across seeds of a seed's total
`planning_time_s`, and the per-shard fixed cost is that pool's maximum minus
its median. Median because the first seed in each process pays every XLA
compile and Numba JIT and no other seed does: the Jun-2026 RockSample run
took 129.4 s in seed 0 episode 0 against a steady 86.8, and 129.2 in seed
25, the next shard's first seed. So max-minus-median IS the fixed cost, and
both fall out of one run without knowing where the shard boundaries were.

Planning time is a lower bound on elapsed (env stepping, the filter update
and JSON writing are outside the timed region), which STARTUP_H covers.
Compare against `sacct` Elapsed after the pilot to close that gap.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

SBU_PER_GPU_HOUR = 128.0
SBU_PER_CORE_HOUR = 1.0
STARTUP_H = 1.5 / 60          # uv start, imports, and the untimed part of a step

# Both rates are MEASURED, from the calibration pilot's sacct AllocTRES:
# gpu_a100 reported billing=128 for a quarter node (1 GPU + 18 cores), and
# genoa reported billing=24 for a cpu=24,mem=3G job — so 1.0 SBU/core-h, not
# the 1.1 this script first assumed.
#
# genoa bills max(cpus-per-task, ceil(mem_MiB / 1792), 24) per JOB. The 24 is
# a 1/8-node floor: a 192-core node is shared by at most 8 jobs. Every CPU row
# in the slate lands exactly on it — the biggest asks are rock_sample at 21
# cores / 36G (ceil(36864/1792) = 21) and rock_sample_depth at 5 cores / 20G
# (12) — so billed cores is a constant here and the only thing that moves the
# CPU bill is the number of jobs. That is what `PACK` below is for.
GENOA_BILLED_CORES = 24

# Shards per SLURM job, for the stages whose job script runs a BLOCK of them
# concurrently in one 24-core allocation (params.yaml cluster.cpu.<stage>.pack;
# see cluster/job_cpu_door_code.sh). Packing leaves n_shards and every
# shard_index untouched, so it is bit-identical to running them one per job —
# it just stops paying the 24-core floor 1575 times. Stages absent here run
# one shard per job, either because they already use --workers (rock_sample,
# factored_tiger) or because isolation is the point (aif_timing).
PACK = {
    "door_code CPU": 10,
    "minigrid CPU": 10,
    "door_code_depth": 9,
    "door_code_rollout": 9,
    "simscaling 1k": 9,
    "simscaling 10k": 9,
    "simscaling 40k": 9,
}

# Speedup of the IDS information term from the 1e-6 state-mass tolerance,
# MEASURED on the canonical RockSample cell (12.947 -> 0.262 s/decision) and
# assumed equal elsewhere — which is why the pilot runs IDS on all four.
IDS_PRUNE_SPEEDUP = 49.4
TREE = {"bamcp", "boss", "pomcp", "factored_bamcp"}
# Read runner.horizon, which the matched-depth change did not move, and have
# no information term, so a pre-fix measurement of these needs no projection.
DEPTH_INVARIANT = {"aif", "vbp", "psrl", "oracle", "random", "belief_q",
                   "persistent_oracle", "persistent_oracle_vbp"}
# Arms with no measurement of their own yet, priced off the arm they wrap.
PROXY = {"persistent_oracle": "oracle", "persistent_oracle_vbp": "oracle"}

# horizon, max_steps, and the bamcp_depth every JSON under output/ was run
# at (params.yaml before 32e0518 matched depth to runner.horizon).
GEOMETRY = {
    "door_code":          (8,  16, 6),
    "minigrid_door_code": (25, 35, 15),
    "rock_sample":        (15, 30, 6),
    "factored_tiger":     (10, 25, 6),
}

GPU4 = ["aif", "vbp", "psrl", "oracle"]
CPU_DC = ["belief_q", "boss", "bamcp", "pomcp", "ids", "random"]
CPU_RS = ["bamcp", "boss", "factored_bamcp", "ids", "random"]

# What submit_all.sh submits. Planner slates come from the submit scripts'
# {GPU,CPU}_PLANNERS arrays, shard/worker counts and cell counts from
# params.yaml's cluster and sweep blocks. `dir` is where this row's
# measurement is read from, relative to --results-dir.
#   label, dir, benchmark, planners, cells, shards, seeds/shard, part, cores, workers
SLATE = [
    ("door_code GPU",        "door_code", "door_code", GPU4, 1, 5, 20, "gpu", 18, 1),
    ("door_code CPU",        "door_code", "door_code", CPU_DC, 1, 100, 1, "cpu", 2, 1),
    ("minigrid GPU",         "minigrid_door_code", "minigrid_door_code", GPU4, 1, 5, 20, "gpu", 18, 1),
    ("minigrid CPU",         "minigrid_door_code", "minigrid_door_code", CPU_DC, 1, 100, 1, "cpu", 3, 1),
    ("rock_sample GPU",      "rock_sample", "rock_sample",
     GPU4 + ["persistent_oracle", "persistent_oracle_vbp"], 1, 2, 25, "gpu", 18, 1),
    ("rock_sample CPU",      "rock_sample", "rock_sample", CPU_RS, 1, 10, 5, "cpu", 21, 4),
    ("factored_tiger GPU",   "factored_tiger", "factored_tiger", GPU4, 1, 1, 100, "gpu", 18, 1),
    ("factored_tiger CPU",   "factored_tiger", "factored_tiger", CPU_RS, 1, 20, 5, "cpu", 7, 2),
    # AIF-only sweeps: the swept knob changes no tensor shape, so each cell
    # costs what the canonical cell costs.
    ("aif_convergence",      "door_code", "door_code", ["aif"], 20, 5, 5, "gpu", 18, 1),
    ("door_code_preference", "door_code", "door_code", ["aif"], 16, 5, 5, "gpu", 18, 1),
    ("rock_sample_pref",     "rock_sample", "rock_sample", ["aif"], 6, 2, 13, "gpu", 18, 1),
    ("ft_preference",        "factored_tiger", "factored_tiger", ["aif"], 9, 1, 25, "gpu", 18, 1),
    ("ft_listen_cost",       "factored_tiger", "factored_tiger", ["aif"], 1, 1, 25, "gpu", 18, 1),
    ("door_code_rollout",    "door_code", "door_code", ["bamcp", "boss"], 2, 25, 1, "cpu", 2, 1),
    # Sweeps with their own per-cell measurements.
    ("kscaling K2 GPU", "rock_sample_kscaling/K2", "rock_sample", GPU4, 1, 2, 25, "gpu", 18, 1),
    ("kscaling K2 CPU", "rock_sample_kscaling/K2", "rock_sample", CPU_RS, 1, 10, 5, "cpu", 21, 4),
    ("kscaling K3 GPU", "rock_sample_kscaling/K3", "rock_sample", GPU4, 1, 2, 25, "gpu", 18, 1),
    ("kscaling K3 CPU", "rock_sample_kscaling/K3", "rock_sample", CPU_RS, 1, 10, 5, "cpu", 21, 4),
    ("kscaling K4 GPU", "rock_sample_kscaling/K4", "rock_sample", GPU4, 1, 2, 25, "gpu", 18, 1),
    ("kscaling K4 CPU", "rock_sample_kscaling/K4", "rock_sample", CPU_RS, 1, 10, 5, "cpu", 21, 4),
    ("simscaling 1k",  "door_code_simscaling/sims1000", "door_code", ["bamcp"], 1, 25, 1, "cpu", 2, 1),
    ("simscaling 10k", "door_code_simscaling/sims10000", "door_code", ["bamcp"], 1, 25, 1, "cpu", 2, 1),
    ("simscaling 40k", "door_code_simscaling/sims40000", "door_code", ["bamcp"], 1, 25, 1, "cpu", 2, 1),
    # Depth sweeps: priced at the canonical depth, which is the top of each
    # sweep and therefore an upper bound on the cells below it.
    ("door_code_depth",   "door_code", "door_code", ["bamcp", "ids"], 4, 25, 1, "cpu", 2, 1),
    ("rock_sample_depth", "rock_sample", "rock_sample", ["bamcp"], 4, 5, 5, "cpu", 5, 4),
]


def per_seed_hours(path: Path) -> tuple[float, float, bool]:
    """(median per-seed hours, per-shard fixed hours, written by current code).

    A JSON without `config.invocation` predates the matched-depth and
    IDS-pruning changes, so it needs projecting forward.
    """
    data = json.loads(path.read_text())
    totals = sorted(sum(v for v in row if v is not None)
                    for row in data["per_episode"]["planning_time_s"])
    n = len(totals)
    median = (totals[n // 2] if n % 2
              else 0.5 * (totals[n // 2 - 1] + totals[n // 2]))
    return (median / 3600, max(0.0, (totals[-1] - median) / 3600),
            "invocation" in data.get("config", {}))


def find(roots: list[Path], subdir: str, benchmark: str, planner: str):
    """(path, arm, root) for the most specific measurement available.

    `roots` are searched in order, so `--results-dir output/_pilot
    --results-dir output` prices every cell the pilot measured from the
    pilot and falls back to the older run for the rest. Without that
    fallback the ~10 cells the pilot deliberately skips as immaterial would
    drop out of the total rather than being priced.

    Specificity outranks root order: a `subdir` hit in a LATER root beats a
    `benchmark` hit in an earlier one. Otherwise a pilot that measured only
    the canonical cell would price the whole K-sweep at the canonical K —
    `output/_pilot/rock_sample` shadowing `output/rock_sample_kscaling/K2`
    made K2, K3 and K4 come out identical, which they are not.
    """
    for kind in (subdir, benchmark):
        for root in roots:
            for arm in (planner, PROXY.get(planner)):
                if arm is None:
                    continue
                d = root / kind
                for name in (f"{arm}.json", f"shards/{arm}.shard00.json"):
                    if (d / name).exists():
                        return d / name, arm, root
    return None, None, None


def effective_depth(depth: int, horizon: int, max_steps: int) -> float:
    """Mean over an episode of the lookahead a simulation actually gets.

    The tree planners cap at `min(depth, max(horizon, 1))` and `clip_horizon`
    shrinks the horizon to the steps remaining, so the nominal depth
    over-states late-episode lookahead — pricing on the nominal ratio
    over-charges RockSample by 20%.
    """
    return sum(min(depth, min(horizon, max_steps - t))
               for t in range(max_steps)) / max_steps


def project(hours, benchmark, planner, new_depth, tree_hours):
    """Carry a pre-fix measurement forward to the current code."""
    if planner in DEPTH_INVARIANT:
        return hours, "measured"
    horizon, max_steps, old_depth = GEOMETRY[benchmark]
    ratio = (effective_depth(new_depth, horizon, max_steps)
             / effective_depth(old_depth, horizon, max_steps))
    if planner in TREE:
        return hours * ratio, "depth"
    if planner == "ids":
        # The tree part scales with depth; the information term is
        # depth-invariant and now pruned.
        tree = min(tree_hours, hours)
        return tree * ratio + (hours - tree) / IDS_PRUNE_SPEEDUP, "depth+prune"
    raise ValueError(f"no projection rule for {planner!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, action="append",
                    help="searched in order; repeat to fall back "
                         "(e.g. --results-dir output/_pilot --results-dir output)")
    args = ap.parse_args()
    roots = args.results_dir or [REPO / "output"]

    params = yaml.safe_load((REPO / "params.yaml").read_text())
    depth_of = {
        "door_code": params["planner"]["bamcp_depth"],
        "minigrid_door_code": params["minigrid"]["planner"]["bamcp_depth"],
        "rock_sample": params["rock_sample"]["planner"]["bamcp_depth"],
        "factored_tiger": params["factored_tiger"]["planner"]["bamcp_depth"],
    }

    by_label, by_basis, totals, missing = {}, {}, {"gpu": 0.0, "cpu": 0.0}, []
    by_root = {}
    longest = {}
    for (label, subdir, bench, planners, cells, shards, sps, part,
         cores, workers) in SLATE:
        for planner in planners:
            path, arm, root = find(roots, subdir, bench, planner)
            if path is None:
                missing.append(f"{label}/{planner}")
                continue
            hours, fixed, current = per_seed_hours(path)
            basis = "measured"
            if not current:
                tree, _, _ = find(roots, subdir, bench, "bamcp")
                hours, basis = project(
                    hours, bench, planner, depth_of[bench],
                    per_seed_hours(tree)[0] if tree else hours)
            if arm != planner:
                basis += f", proxy {arm}"
            shard_h = hours * math.ceil(sps / workers) + fixed + STARTUP_H
            # A packed job's shards run concurrently, so its elapsed is the
            # worst shard's rather than their sum — which is why dividing the
            # job count by `pack` is the whole saving. GPU rows are never
            # packed (one process owns the device).
            pack = 1 if part == "gpu" else PACK.get(label, 1)
            n_jobs = cells * math.ceil(shards / pack)
            rate = (SBU_PER_GPU_HOUR if part == "gpu"
                    else SBU_PER_CORE_HOUR * GENOA_BILLED_CORES)
            sbu = n_jobs * shard_h * rate
            totals[part] += sbu
            by_label[label] = by_label.get(label, 0.0) + sbu
            by_basis[basis] = by_basis.get(basis, 0.0) + sbu
            by_root[root] = by_root.get(root, 0.0) + sbu
            longest[label] = max(longest.get(label, 0.0), shard_h)

    total = sum(totals.values())
    print("results from " + ", ".join(str(r) for r in roots) + "\n")
    print(f"{'group':22s} {'SBU':>8s}   longest shard")
    for label, sbu in sorted(by_label.items(), key=lambda kv: -kv[1]):
        print(f"{label:22s} {sbu:8.0f}   {longest[label]:8.2f} h")
    print(f"\nGPU {totals['gpu']:8.0f} SBU  ({totals['gpu'] / SBU_PER_GPU_HOUR:5.0f} A100-h)")
    print(f"CPU {totals['cpu']:8.0f} SBU  "
          f"({totals['cpu'] / SBU_PER_CORE_HOUR:5.0f} BILLED core-h, at "
          f"{GENOA_BILLED_CORES} per job whatever the job asks for)")
    print(f"    {total:8.0f} SBU  SLATE")
    if len(by_root) > 1:
        print("\nSBU by source of the measurement:")
        for r, v in sorted(by_root.items(), key=lambda kv: -kv[1]):
            print(f"  {str(r):34s} {v:8.0f} SBU  ({100 * v / total:5.1f}%)")
    print("\nbasis of the per-seed cost:")
    for basis, sbu in sorted(by_basis.items(), key=lambda kv: -kv[1]):
        print(f"  {basis:22s} {sbu:8.0f} SBU  ({100 * sbu / total:5.1f}%)")
    print(f"\nmeasured rates (pilot sacct AllocTRES): {SBU_PER_GPU_HOUR:.0f} "
          f"SBU/GPU-h, {SBU_PER_CORE_HOUR} SBU/core-h with a "
          f"{GENOA_BILLED_CORES}-core floor per genoa job.")
    print(f"assumed: {STARTUP_H * 60:.1f} min startup per shard.")
    packed = ", ".join(f"{k}x{v}" for k, v in sorted(PACK.items()))
    print(f"shards packed per job: {packed}")
    for line in sorted(set(missing)):
        print(f"  MISSING  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
