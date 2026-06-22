#!/usr/bin/env bash
#
# reproduce-all.sh — regenerate every chapter's results from scratch.
#
# Each chapter is an independent uv environment + DVC pipeline. This script
# walks them in dissertation order and, for each one, syncs the environment
# and runs `dvc repro`. DVC itself is incremental: stages whose code and
# params are unchanged are skipped, so re-running this is cheap once results
# exist. In particular, because every chapter keeps post-hoc plotting in its
# own DVC stage(s), editing a plot script and re-running only re-renders the
# figure — the expensive experiment runs stay cached.
#
# Usage:
#   ./reproduce-all.sh                 # sync + repro every chapter, in order
#   ./reproduce-all.sh chapter-5       # only the named chapter(s)
#   ./reproduce-all.sh --cuda          # `uv sync --extra gpu` (GPU jaxlib)
#   ./reproduce-all.sh --no-sync       # skip `uv sync` (env already built)
#   ./reproduce-all.sh --push          # `dvc push` after each successful repro
#   ./reproduce-all.sh -- --force      # pass extra args through to `dvc repro`
#
# Anything after a literal `--` is forwarded verbatim to each `dvc repro`
# call (e.g. `-- --force`, `-- -s tmaze_experiment_marginal`, `-- --dry`).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_CHAPTERS=(chapter-3 chapter-4 chapter-5 chapter-6)

SYNC=1
PUSH=0
SYNC_ARGS=()
SELECTED=()
DVC_ARGS=()

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sync) SYNC=0; shift ;;
    --cuda)    SYNC_ARGS+=(--extra gpu); shift ;;
    --push)    PUSH=1; shift ;;
    --)        shift; DVC_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    chapter-*) SELECTED+=("$1"); shift ;;
    *)
      echo "error: unknown argument '$1' (chapters must be named like 'chapter-5')" >&2
      exit 2 ;;
  esac
done

if [[ ${#SELECTED[@]} -gt 0 ]]; then
  CHAPTERS=("${SELECTED[@]}")
else
  CHAPTERS=("${ALL_CHAPTERS[@]}")
fi

# --- run --------------------------------------------------------------------
failed=()
for ch in "${CHAPTERS[@]}"; do
  dir="$ROOT/$ch"
  if [[ ! -f "$dir/dvc.yaml" ]]; then
    echo "!! skipping $ch — no dvc.yaml found" >&2
    failed+=("$ch (no pipeline)")
    continue
  fi

  echo
  echo "==================================================================="
  echo ">>> $ch"
  echo "==================================================================="

  (
    cd "$dir"
    if [[ "$SYNC" -eq 1 ]]; then
      echo "--- uv sync ${SYNC_ARGS[*]:-}"
      if [[ ${#SYNC_ARGS[@]} -gt 0 ]]; then
        uv sync "${SYNC_ARGS[@]}"
      else
        uv sync
      fi
    fi
    if [[ ${#DVC_ARGS[@]} -gt 0 ]]; then
      echo "--- uv run dvc repro ${DVC_ARGS[*]}"
      uv run dvc repro "${DVC_ARGS[@]}"
    else
      echo "--- uv run dvc repro"
      uv run dvc repro
    fi
    if [[ "$PUSH" -eq 1 ]]; then
      echo "--- uv run dvc push"
      uv run dvc push
    fi
  ) || { failed+=("$ch"); echo "!! $ch FAILED" >&2; }
done

# --- summary ----------------------------------------------------------------
echo
echo "==================================================================="
if [[ ${#failed[@]} -eq 0 ]]; then
  echo ">>> all chapters reproduced successfully"
else
  echo ">>> FAILURES: ${failed[*]}"
  exit 1
fi
