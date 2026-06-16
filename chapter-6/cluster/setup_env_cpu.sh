#!/usr/bin/env bash
# Snellius CPU-partition (genoa/rome) environment setup. Source this from
# any CPU job script — and run it once on the login node first to build
# the venv (compute nodes lack internet, so the install must happen
# before sbatch).
#
# Usage:
#   source cluster/setup_env_cpu.sh
#
# This is the CPU sibling of cluster/setup_env.sh: it installs the base
# project (no --extra gpu), into a separate .venv-cpu so the cuda jaxlib
# from the GPU venv is not on PATH. The CPU planners (bamcp, boss, pomcp,
# ids, belief_q, factored_bamcp) are pure NumPy/Numba and do not need
# CUDA.
#
# NOTE: this file is meant to be *sourced*, so it deliberately does not
# `set -euo pipefail` at the top — that would kill the caller's shell on
# any failure. Job scripts that source it already enable strict mode
# themselves; on the login node, individual failing commands are
# surfaced via explicit `|| return 1` below.

PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge || return 1
module load 2023 || return 1
module load "$PYTHON_MODULE" || return 1

# uv is the project's mandated package manager (CLAUDE.md). Bootstrap a
# standalone install into ~/.local/bin on first use; idempotent thereafter.
if ! command -v uv &>/dev/null; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing uv into ~/.local/bin (login-node only) ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh \
            || { echo "uv install failed" >&2; return 1; }
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv-cpu"

# Build the venv from the lockfile (no --extra gpu — base jax/jaxlib pins
# resolve to the CPU build). Only sync when the venv is absent or a refresh is
# explicitly requested via FORCE_UV_SYNC: a compute job re-sourcing this on
# every array task would otherwise hammer the project dir with metadata I/O at
# startup, and when that dir is on a degraded GPFS mount the simultaneous storm
# across an array trips node-health checks and the jobs get killed. Build or
# refresh once on the login node with `FORCE_UV_SYNC=1 source ...`.
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ] || [ -n "${FORCE_UV_SYNC:-}" ]; then
    echo "Syncing $UV_PROJECT_ENVIRONMENT (CPU-only) ..."
    uv sync --project "$PROJECT_DIR" \
        || { echo "uv sync (CPU) failed" >&2; return 1; }
else
    echo "Using existing venv $UV_PROJECT_ENVIRONMENT (skipping uv sync; set FORCE_UV_SYNC=1 to refresh)"
fi

# Surface the venv tools on PATH so plain `python` / `uv run` work.
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
