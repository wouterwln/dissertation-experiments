#!/usr/bin/env python
"""Plot convergence sweep results: damping curves, method comparisons, heatmaps.

Usage:
    uv run python plot_convergence_sweep.py --environment frozen-lake
    uv run python plot_convergence_sweep.py --environment all
    uv run python plot_convergence_sweep.py --environment frozen-lake --format png --tex
"""

import sys
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent))

# --- Repo-root dissertation theme (imported at runtime, not a DVC dep) -------
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402

import numpy as np

ALL_ENVIRONMENTS = ["frozen-lake", "wumpus-world", "rocksample", "minigrid"]

# Canonical cross-figure method order: BP - VBP - RM-MP - Direct-prior MP -
# AIF-MP (auxiliary variants trail).
ALL_METHODS = [
    "loopy", "vbp-channel", "dyn-channel", "nuijten", "active-inference",
    "region-extended", "precise-info-seeking",
]

METHODS_WITH_DAMPING = {"region-extended", "dyn-channel", "vbp-channel", "precise-info-seeking", "active-inference"}

METHOD_LABELS = {
    "loopy": "Loopy BP",
    "region-extended": "Region-Extended",
    "dyn-channel": "Dyn-Channel",
    "nuijten": "Direct-prior MP",
    "vbp-channel": "VBP Channel",
    "precise-info-seeking": "Precise Info-Seeking",
    "active-inference": "Active Inference",
}

ENV_LABELS = {
    "frozen-lake": "Frozen Lake",
    "wumpus-world": "Wumpus World",
    "rocksample": "RockSample",
    "minigrid": "MiniGrid DoorKey",
}


# =============================================================================
# Data loading
# =============================================================================


def load_results(input_dir, env_name):
    """Load all per-run JSON results for an environment.

    Returns dict: {(method, damping, seed): data_dict}
    """
    env_dir = Path(input_dir) / env_name
    results = {}

    if not env_dir.exists():
        print(f"Warning: {env_dir} does not exist, skipping")
        return results

    for method_dir in sorted(env_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "plots":
            continue
        method = method_dir.name
        for damping_dir in sorted(method_dir.iterdir()):
            if not damping_dir.is_dir():
                continue
            damping_str = damping_dir.name.replace("damping_", "")
            damping = float(damping_str)
            for json_file in sorted(damping_dir.glob("seed_*.json")):
                seed = int(json_file.stem.replace("seed_", ""))
                with open(json_file) as f:
                    data = json.load(f)
                results[(method, damping, seed)] = data

    return results


def collect_vfe_traces(results, method, damping):
    """Collect VFE traces for a (method, damping) pair across seeds.

    Returns list of arrays, one per seed.
    """
    traces = []
    for (m, d, s), data in sorted(results.items()):
        if m == method and d == damping:
            traces.append(np.array(data["results"]["vfe_trace"]))
    return traces


def get_methods_and_dampings(results):
    """Get sorted unique methods and damping values from results."""
    methods = sorted(set(m for m, d, s in results.keys()),
                     key=lambda m: ALL_METHODS.index(m) if m in ALL_METHODS else 99)
    dampings = sorted(set(d for m, d, s in results.keys()))
    return methods, dampings


# =============================================================================
# Plot 1: Per-method damping sweep
# =============================================================================


def plot_damping_sweep(results, env_name, output_dir, fmt, generate_tex):
    """One plot per (env, method): VFE vs iteration, one line per damping."""
    import matplotlib.pyplot as plt

    methods, dampings = get_methods_and_dampings(results)
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        method_dampings = sorted(set(
            d for (m, d, s) in results.keys() if m == method
        ))
        if not method_dampings:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))

        # Damping is an ordered parameter, so it wears the theme's ordinal teal
        # ramp (light = weak damping, dark = strong) rather than distinct hues.
        damping_colors = theme.ordinal_colors(len(method_dampings))
        for damping, dcolor in zip(method_dampings, damping_colors):
            traces = collect_vfe_traces(results, method, damping)
            if not traces:
                continue

            # Pad to same length
            max_len = max(len(t) for t in traces)
            padded = np.full((len(traces), max_len), np.nan)
            for i, t in enumerate(traces):
                padded[i, :len(t)] = t

            median = np.nanmedian(padded, axis=0)
            q25 = np.nanpercentile(padded, 25, axis=0)
            q75 = np.nanpercentile(padded, 75, axis=0)
            iters = np.arange(max_len)

            label = f"d={damping}"
            line, = ax.plot(iters, median, linewidth=1.2, label=label, color=dcolor)
            ax.fill_between(iters, q25, q75, alpha=0.15, color=dcolor)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("VFE")
        ax.set_title(f"{ENV_LABELS.get(env_name, env_name)}: {METHOD_LABELS.get(method, method)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _set_robust_ylim(ax)
        fig.tight_layout()

        for ext in _extensions(fmt):
            path = plot_dir / f"{method}_damping.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if generate_tex:
            _write_damping_tex(plot_dir, method, method_dampings, traces, env_name)


def _write_damping_tex(plot_dir, method, dampings, all_traces, env_name):
    """Generate pgfplots .tex for damping sweep."""
    from matplotlib.colors import to_hex

    tex_path = plot_dir / f"{method}_damping.tex"
    # Same ordinal teal ramp as the matplotlib version, emitted as \definecolor.
    ramp = [to_hex(c).lstrip("#").upper() for c in theme.ordinal_colors(len(dampings))]
    color_defs = "\n".join(
        f"\\definecolor{{damping{i}}}{{HTML}}{{{hexval}}}" for i, hexval in enumerate(ramp)
    )

    plot_lines = []
    legend_entries = []
    for i, d in enumerate(dampings):
        plot_lines.append(
            f"    \\addplot[damping{i}, thick] table[col sep=comma, "
            f"x=iteration, y=d{str(d).replace('.', '_')}] "
            f"{{{method}_damping.csv}};"
        )
        legend_entries.append(f"$\\lambda = {d}$")

    tex = f"""\
\\documentclass[tikz]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=newest}}
{color_defs}
\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=10cm, height=7cm,
    xlabel={{Iteration}}, ylabel={{VFE}},
    title={{{METHOD_LABELS.get(method, method)} ({ENV_LABELS.get(env_name, env_name)})}},
    grid=major, grid style={{gray!30}},
    legend pos=south east, legend cell align={{left}},
]
{chr(10).join(plot_lines)}
    \\legend{{{", ".join(legend_entries)}}}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
    tex_path.write_text(tex)


# =============================================================================
# Plot 2: Cross-method comparison
# =============================================================================


def plot_method_comparison(results, env_name, output_dir, fmt):
    """All methods on one plot, each at best-converging damping."""
    import matplotlib.pyplot as plt

    methods, dampings = get_methods_and_dampings(results)
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in methods:
        # Find best damping: lowest median final delta
        method_dampings = sorted(set(
            d for (m, d, s) in results.keys() if m == method
        ))
        best_damping = None
        best_med_delta = float("inf")
        for d in method_dampings:
            deltas = [
                abs(data["results"]["vfe_trace"][-1] - data["results"]["vfe_trace"][-2])
                for (m, dd, s), data in results.items()
                if m == method and dd == d and len(data["results"]["vfe_trace"]) >= 2
            ]
            if deltas:
                med = np.median(deltas)
                if med < best_med_delta:
                    best_med_delta = med
                    best_damping = d

        if best_damping is None:
            continue

        traces = collect_vfe_traces(results, method, best_damping)
        if not traces:
            continue

        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, :len(t)] = t

        median = np.nanmedian(padded, axis=0)
        q25 = np.nanpercentile(padded, 25, axis=0)
        q75 = np.nanpercentile(padded, 75, axis=0)
        iters = np.arange(max_len)

        label = f"{METHOD_LABELS.get(method, method)}"
        if best_damping != 1.0:
            label += f" (d={best_damping})"
        mcolor = theme.method_color(method)
        # Sparse markers double as a non-color encoding (CVD / greyscale print).
        ax.plot(iters, median, linewidth=1.4, label=label, color=mcolor,
                marker=theme.method_marker(method), markersize=4,
                markevery=max(len(iters) // 12, 1))
        ax.fill_between(iters, q25, q75, alpha=0.12, color=mcolor)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("VFE")
    ax.set_title(f"{ENV_LABELS.get(env_name, env_name)}: Method Comparison (best damping)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    _set_robust_ylim(ax)
    fig.tight_layout()

    for ext in _extensions(fmt):
        path = plot_dir / f"method_comparison.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plot 3: Convergence heatmap
# =============================================================================


def plot_convergence_heatmap(results, env_name, output_dir, fmt):
    """Heatmap: rows=methods, cols=damping, color=fraction converged."""
    import matplotlib.pyplot as plt

    methods, dampings = get_methods_and_dampings(results)
    if not methods or not dampings:
        return

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_methods = len(methods)
    n_dampings = len(dampings)

    conv_frac = np.full((n_methods, n_dampings), np.nan)
    med_iters = np.full((n_methods, n_dampings), np.nan)

    for mi, method in enumerate(methods):
        for di, damping in enumerate(dampings):
            runs = [
                data for (m, d, s), data in results.items()
                if m == method and d == damping
            ]
            if not runs:
                continue
            n_conv = sum(1 for r in runs if r["results"]["converged"])
            conv_frac[mi, di] = n_conv / len(runs)
            conv_iters = [
                r["results"]["convergence_iteration"]
                for r in runs if r["results"]["convergence_iteration"] >= 0
            ]
            if conv_iters:
                med_iters[mi, di] = np.median(conv_iters)

    fig, ax = plt.subplots(figsize=(max(4, n_dampings * 1.2), max(3, n_methods * 0.6)))

    # Thesis diverging colormap: terracotta (diverges) → neutral → teal (converges).
    cmap = theme.diverging_cmap()
    im = ax.imshow(conv_frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Cell annotations: paper ink on the dark pole cells, charcoal elsewhere.
    for mi in range(n_methods):
        for di in range(n_dampings):
            if np.isnan(conv_frac[mi, di]):
                ax.text(di, mi, "—", ha="center", va="center", fontsize=8,
                        color=theme.PALETTE["muted"])
            else:
                frac = conv_frac[mi, di]
                iter_str = f"\n({int(med_iters[mi, di])})" if not np.isnan(med_iters[mi, di]) else ""
                color = theme.PALETTE["cell"] if (frac < 0.25 or frac > 0.75) else theme.PALETTE["ink"]
                ax.text(di, mi, f"{frac:.0%}{iter_str}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(n_dampings))
    ax.set_xticklabels([str(d) for d in dampings], fontsize=8)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8)
    ax.set_xlabel("Damping")
    ax.set_title(f"{ENV_LABELS.get(env_name, env_name)}: Convergence Rate")

    fig.colorbar(im, ax=ax, label="Fraction converged", shrink=0.8)
    fig.tight_layout()

    for ext in _extensions(fmt):
        path = plot_dir / f"convergence_heatmap.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plot 4: Final VFE bar chart
# =============================================================================


def plot_final_vfe(results, env_name, output_dir, fmt):
    """Grouped bar chart: final VFE per method × damping."""
    import matplotlib.pyplot as plt

    methods, dampings = get_methods_and_dampings(results)
    if not methods:
        return

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 4))

    bar_width = 0.8 / max(len(dampings), 1)
    x = np.arange(len(methods))

    # Damping is ordered, so the bars wear the ordinal teal ramp.
    damping_colors = theme.ordinal_colors(len(dampings))
    for di, damping in enumerate(dampings):
        means = []
        stds = []
        for method in methods:
            vfes = [
                data["results"]["final_vfe"]
                for (m, d, s), data in results.items()
                if m == method and d == damping
            ]
            if vfes:
                means.append(np.mean(vfes))
                stds.append(np.std(vfes))
            else:
                means.append(np.nan)
                stds.append(0)

        offset = (di - len(dampings) / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
               label=f"d={damping}", capsize=2, alpha=0.85,
               color=damping_colors[di])

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Final VFE")
    ax.set_title(f"{ENV_LABELS.get(env_name, env_name)}: Final VFE by Method & Damping")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    _set_robust_ylim(ax)
    fig.tight_layout()

    for ext in _extensions(fmt):
        path = plot_dir / f"final_vfe.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Helpers
# =============================================================================


def _extensions(fmt):
    if fmt == "both":
        return ["pdf", "png"]
    return [fmt]


def _set_robust_ylim(ax, margin=0.05):
    """Set y-axis limits based on 5th/95th percentile to exclude outliers."""
    all_y = []
    for line in ax.get_lines():
        all_y.extend(line.get_ydata())
    for patch in ax.patches:
        if patch is None:
            continue
        h = patch.get_height()
        if np.isfinite(h):
            all_y.append(h)
    all_y = np.array([v for v in all_y if np.isfinite(v)])
    if len(all_y) < 2:
        return
    lo, hi = np.percentile(all_y, [5, 95])
    pad = max((hi - lo) * margin, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence sweep results")
    parser.add_argument("--environment", type=str, default="all",
                        choices=ALL_ENVIRONMENTS + ["all"])
    parser.add_argument("--input-dir", type=str, default="data/convergence_sweep")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override plot output dir (default: <input-dir>/plots)")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "both"])
    parser.add_argument("--tex", action="store_true",
                        help="Also generate pgfplots .tex files")

    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        theme.apply_rcparams()  # shared serif/sizing across the dissertation
        matplotlib.rcParams.update({
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.usetex": False,
            "figure.figsize": (6, 4),
        })
    except ImportError:
        print("matplotlib not available — cannot generate plots")
        sys.exit(1)

    envs = ALL_ENVIRONMENTS if args.environment == "all" else [args.environment]

    for env_name in envs:
        print(f"Processing {env_name}...")
        results = load_results(args.input_dir, env_name)
        if not results:
            print(f"  No results found, skipping")
            continue

        output_dir = args.output_dir or str(Path(args.input_dir) / "plots")
        env_plot_dir = Path(output_dir) / env_name

        print(f"  Loaded {len(results)} runs")
        print(f"  Output: {env_plot_dir}")

        plot_damping_sweep(results, env_name, env_plot_dir, args.format, args.tex)
        plot_method_comparison(results, env_name, env_plot_dir, args.format)
        plot_convergence_heatmap(results, env_name, env_plot_dir, args.format)
        plot_final_vfe(results, env_name, env_plot_dir, args.format)

        print(f"  Done")

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
