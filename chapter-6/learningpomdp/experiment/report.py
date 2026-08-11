"""Programmatic window metrics for the paper's scalar tables.

The early/final window numbers the paper reports used to exist only as
hand-typed LaTeX literals, so they could drift from `output/` silently.
This module computes them from the raw `per_episode` arrays a runner
writes, and `scripts/make_tables.py` renders them.

Aggregation convention (per-seed, then across seeds)
----------------------------------------------------
For a window of `k` episodes, average the window **within each block**
(a block = one outer seed = one draw of theta_true), giving one scalar
per block, then report the across-block mean with a half-width of
`1.96 * nanstd(ddof=1) / sqrt(n_blocks)`.

Three conventions were candidates and they do not agree:

  (A) per-seed, then across seeds        <- chosen
  (B) mean of the across-seed per-episode curve over the window, with
      the interval taken from the per-episode SEMs
  (C) pooled over every (seed, episode) cell in the window, with
      `n = n_seeds * k`

(A) is chosen for three reasons:

  1. The block is the unit of randomization. theta_true is drawn once
     per block and every episode in that block shares it, so episodes
     within a block are not independent. (C) treats them as if they
     were and understates the interval accordingly.
  2. It is the window generalization of the estimator the publication
     figure renderer already uses per episode
     (`scripts/plot_paper_figures.py:105-113`): both aggregate along the
     seed axis with `ddof=1` and `n` = number of finite seeds.
  3. It reproduces the published tables. On RockSample AIF-MP's early
     window, (A) gives 0.554 -> "0.55" as published, while (B) gives
     0.542 and (C) 0.543, both of which round to "0.54".

NaN handling matters because `reward_normalised` is undefined (NaN) on
episodes where no positive reward was obtainable. Under (A) such cells
drop out of their own block mean and a block is dropped entirely only
when its whole window is NaN, which is why the aggregation is nan-safe
throughout.
"""

from __future__ import annotations

import numpy as np

# Half-width multiplier for a 95% normal-approximation CI of the mean.
# Matches `scripts/plot_paper_figures.py` and every main-figure band.
CI_Z = 1.96


def mean_ci(values: np.ndarray) -> tuple[float, float, int]:
    """Mean, 95% CI half-width, and finite sample size over a 1-D sample.

    NaN-safe: NaN entries are dropped and `n` is the finite count. With
    fewer than two finite entries the half-width is 0.0 (the sample
    standard deviation is undefined).

    `n` is returned rather than left implicit because it is *not* the
    block count in the run header whenever a block's whole window is
    NaN, and a caption that quotes the header count next to an interval
    computed over fewer blocks is wrong. Callers must report the `n`
    they were handed.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    if finite.size < 2:
        return float(finite[0]), 0.0, 1
    sem = float(np.std(finite, ddof=1) / np.sqrt(finite.size))
    return float(finite.mean()), CI_Z * sem, int(finite.size)


def _window(per_episode: np.ndarray, k: int, side: str) -> np.ndarray:
    arr = np.asarray(per_episode, dtype=float)
    n_episodes = arr.shape[1]
    # `arr[:, -0:]` is the whole array and `arr[:, :0]` is empty, so a
    # non-positive k silently degenerates rather than failing; a k past
    # the end silently shortens the window to the run length.
    if k <= 0:
        raise ValueError(f"window length must be positive, got k={k}")
    if k > n_episodes:
        raise ValueError(
            f"window length k={k} exceeds the {n_episodes} episodes on record"
        )
    if side == "early":
        return arr[:, :k]
    if side == "final":
        return arr[:, -k:]
    raise ValueError(f"side must be 'early' or 'final', got {side!r}")


def window_overlap(n_episodes: int, n_early: int, n_final: int) -> int:
    """Episodes shared by the early and final windows; 0 when disjoint.

    The early/final pair is read as a before/after-learning contrast,
    which holds only while the windows are disjoint. They are not
    disjoint whenever `n_early + n_final > n_episodes`, and nothing
    downstream notices: `_window` returns two overlapping slices and the
    aggregation is happy to average the same episode into both. Callers
    must disclose a non-zero overlap rather than reporting the contrast
    as if the windows were separate data.
    """
    if n_early <= 0 or n_final <= 0:
        raise ValueError(
            f"window lengths must be positive, got "
            f"n_early={n_early}, n_final={n_final}"
        )
    return max(0, int(n_early) + int(n_final) - int(n_episodes))


def _row_nanmean(window: np.ndarray) -> np.ndarray:
    """Per-row mean ignoring NaN, NaN for all-NaN rows, no warnings."""
    finite = np.isfinite(window)
    counts = finite.sum(axis=1)
    sums = np.where(finite, window, 0.0).sum(axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def window_mean_ci(
    per_episode: np.ndarray, k: int, side: str,
) -> tuple[float, float, int]:
    """Mean, 95% CI half-width and block count of the first / last `k`.

    Args:
        per_episode: (n_seeds, n_episodes) raw per-episode values.
        k: window length in episodes.
        side: "early" for the first `k` episodes, "final" for the last.

    The third element is the number of blocks that actually entered the
    mean, which is below `per_episode.shape[0]` whenever a block's whole
    window is NaN.
    """
    return mean_ci(_row_nanmean(_window(per_episode, k, side)))


def final_episode_mean_ci(per_episode: np.ndarray) -> tuple[float, float, int]:
    """Mean, 95% CI half-width and block count of the last episode.

    Used for the "final mass" column, which the paper defines on the
    final episode rather than on a window.
    """
    arr = np.asarray(per_episode, dtype=float)
    return mean_ci(arr[:, -1])


def per_decision_planning_time(
    planning_time_s: np.ndarray, n_steps: np.ndarray,
) -> float:
    """Seconds of planning per decision, over all decisions of all blocks.

    `per_episode.planning_time_s` is the **episode total**, not a
    per-decision cost, so it must be divided by the episode's step count.
    This returns `sum(time) / sum(steps)`, i.e. the mean over all
    decisions of all blocks -- the estimator the paper's timing-table
    caption describes. Note `scripts/plot_door_code.py:168-183` instead
    reports the *median* per-episode ratio for its figure panel, which is
    a different (more outlier-robust) estimator and gives different
    numbers.
    """
    time = np.asarray(planning_time_s, dtype=float)
    steps = np.asarray(n_steps, dtype=float)
    mask = np.isfinite(time) & np.isfinite(steps) & (steps > 0)
    if not mask.any():
        return float("nan")
    return float(time[mask].sum() / steps[mask].sum())


def theta_pairing(theta_a: list[int], theta_b: list[int]) -> dict:
    """How much of the theta_true draw two runs share, by seed index.

    theta_true is drawn from a dedicated RNG seeded per shard
    (`door_code_runner.py:593`, `run_door_code.py:256`), so two runs
    agree seed-for-seed only if they used the same shard layout. Runs
    submitted to different SLURM partitions with different `n_shards`
    do not, which makes paired differences invalid between them.

    Returns the shared-index count, the comparable length, and whether
    the sequences are identical over that length.
    """
    n = min(len(theta_a), len(theta_b))
    shared = sum(1 for i in range(n) if theta_a[i] == theta_b[i])
    return {
        "n_compared": n,
        "n_shared_theta": shared,
        "paired": bool(n > 0 and shared == n and len(theta_a) == len(theta_b)),
    }


def paired_window_difference(
    per_episode_a: np.ndarray,
    per_episode_b: np.ndarray,
    k: int,
    side: str,
) -> tuple[float, float, int]:
    """Mean, 95% CI half-width and pair count of the difference a - b.

    Only meaningful when the two runs share theta_true seed-for-seed --
    check with `theta_pairing` first. Environment noise is *not*
    step-matched even when theta is shared (probe bit-flips are drawn
    lazily, `environments/door_code.py:655`), so this is a
    theta-paired difference, not a common-random-numbers difference.
    """
    window_a = _window(per_episode_a, k, side)
    window_b = _window(per_episode_b, k, side)
    if window_a.shape[0] != window_b.shape[0]:
        raise ValueError(
            f"seed counts differ: {window_a.shape[0]} vs {window_b.shape[0]}"
        )
    return mean_ci(_row_nanmean(window_a) - _row_nanmean(window_b))
