"""Regenerate the committed golden slices of `output/` under tests/golden/.

The drift detector in `tests/test_report.py` compares the paper's
hand-typed table cells against the numbers the emitter computes. Pointed
at `output/` alone it skipped everywhere `output/` is absent -- CI, fresh
clones, every git worktree -- which is most places, so the "nothing has
drifted" guarantee was enforced nowhere automatic. These slices carry the
only fields those tests read, so the check always runs; where a real
`output/` *is* present the same tests additionally run against it, which
is what catches the slices themselves going stale.

Regenerate after a rerun that legitimately changes the published numbers,
from a machine holding the canonical output/:

    uv run python tests/golden/make_fixture.py
    uv run python tests/golden/make_fixture.py --input-dir /path/to/output

(the second form for a git worktree, whose own output/ does not exist)

then update PUBLISHED_SCALARS / PUBLISHED_TIMING in the same commit and
say in the message which run produced them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

# Only these per_episode arrays are read by the golden tests; the full
# JSONs are ~9.5 MB each and mostly per-step trajectories.
FIELDS = ("planning_time_s", "n_steps")
PRECISION = 6


def _round(values: list) -> list:
    return [
        [
            v if isinstance(v, int) or v is None else round(float(v), PRECISION)
            for v in row
        ]
        for row in values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=REPO_ROOT / "output",
        help="canonical result tree to slice (default: <repo>/output)",
    )
    args = parser.parse_args()

    from test_report import PUBLISHED_SCALARS, PUBLISHED_TIMING  # noqa: PLC0415

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import make_tables  # noqa: PLC0415

    for bench in make_tables.BENCHMARKS:
        planners = sorted(
            set(PUBLISHED_SCALARS.get(bench.dirname, {}))
            | set(PUBLISHED_TIMING.get(bench.dirname, {}))
        )
        runs: dict[str, dict] = {}
        for planner in planners:
            path = args.input_dir / bench.dirname / f"{planner}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is published but absent; the slice would be "
                    f"incomplete"
                )
            run = json.loads(path.read_text())
            runs[planner] = {
                "n_seeds": int(run["n_seeds"]),
                "n_episodes": int(run["n_episodes"]),
                "per_episode": {
                    key: _round(run["per_episode"][key])
                    for key in (bench.metric, bench.mass_key, *FIELDS)
                },
            }
        out = GOLDEN_DIR / f"slice_{bench.dirname}.json"
        out.write_text(json.dumps(
            {"source": f"output/{bench.dirname}", "runs": runs},
            separators=(",", ":"), sort_keys=True,
        ) + "\n")
        print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
