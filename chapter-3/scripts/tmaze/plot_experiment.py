#!/usr/bin/env python3
"""Render the T-Maze episode video / TikZ frames from a finished experiment run.

This is the *post-hoc visualization* half of the T-Maze pipeline. The experiment
(`experiment.py`) writes the full per-episode record — including each step's
planning snapshot — to `results.json`; this script reloads the last episode from
that JSON and re-renders `episode.mp4` and the `frames/` TikZ output.

Keeping it a separate DVC stage means editing the visualization code re-renders the
figures without re-running the (expensive) experiment: the run only depends on its
own code, not on `src/visualization/`.

Usage:
    # Render video + TikZ frames into the run's own directory:
    python scripts/tmaze/plot_experiment.py --input-dir data/tmaze/marginal

    # Write the standalone reference T-maze figure (run once):
    python scripts/tmaze/plot_experiment.py --reference data/tmaze/tmaze.tex
"""

import argparse
import json
import sys
from pathlib import Path

# Project root (for `src.*`) and this dir (for `experiment`).
_HERE = Path(__file__).parent
project_root = _HERE.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(_HERE))

# Reuse the experiment's own dataclasses + render helpers so the output is
# identical to what an inline `experiment.py` run would have produced.
from experiment import (  # noqa: E402
    EpisodeResult,
    PlanningSnapshot,
    create_last_episode_video,
    create_last_episode_tikz_frames,
)
from src.visualization import save_tmaze_reference  # noqa: E402


def load_last_episode(results_path: Path) -> tuple[EpisodeResult, str]:
    """Reconstruct the last episode's EpisodeResult from a results.json file."""
    with open(results_path) as f:
        data = json.load(f)

    episode = dict(data["episodes"][-1])
    snapshots = episode.get("planning_history")
    if snapshots is not None:
        episode["planning_history"] = [PlanningSnapshot(**s) for s in snapshots]

    result = EpisodeResult(**episode)
    inference_tag = data.get("metadata", {}).get("inference_type", "")
    return result, inference_tag


def main():
    parser = argparse.ArgumentParser(
        description="Render T-Maze episode video / TikZ frames from results.json"
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Run directory containing results.json; video/frames are written here.",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Write the reference T-maze figure (.tex) to this path and exit.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip episode.mp4")
    parser.add_argument("--no-tikz", action="store_true", help="Skip TikZ frames/")
    args = parser.parse_args()

    if args.reference:
        ref_path = save_tmaze_reference(Path(args.reference))
        print(f"Reference figure saved to: {ref_path}")
        return

    if not args.input_dir:
        parser.error("either --input-dir or --reference is required")

    input_dir = Path(args.input_dir)
    results_path = input_dir / "results.json"
    if not results_path.exists():
        parser.error(f"no results.json in {input_dir}")

    result, inference_tag = load_last_episode(results_path)

    if not args.no_video:
        video_path = create_last_episode_video(result, input_dir, inference_tag)
        print(f"Video saved to: {video_path}")

    if not args.no_tikz:
        frames_dir = create_last_episode_tikz_frames(result, input_dir)
        print(f"TikZ frames saved to: {frames_dir}")


if __name__ == "__main__":
    main()
