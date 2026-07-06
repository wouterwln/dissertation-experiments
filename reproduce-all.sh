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
#   ./reproduce-all.sh --commit        # git-commit each chapter's dvc.lock after repro
#   ./reproduce-all.sh -- --force      # pass extra args through to `dvc repro`
#
# `--push` uploads the regenerated data to the DVC remote; `--commit` records
# the matching dvc.lock (+ .gitignore) in git so the commit points at exactly
# that pushed data, then `git push`es once at the end if anything was
# committed. Use them together (`--push --commit`) to close the loop:
# repro -> upload data -> commit the pointers -> push. (`--commit` only stages
# the DVC lockfiles, never source code; cache:false metric files are left
# untouched.)
#
# Anything after a literal `--` is forwarded verbatim to each `dvc repro`
# call (e.g. `-- --force`, `-- -s tmaze_experiment_marginal`, `-- --dry`).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_CHAPTERS=(chapter-3 chapter-4 chapter-5 chapter-6)

SYNC=1
PUSH=0
COMMIT=0
SYNC_ARGS=()
SELECTED=()
DVC_ARGS=()

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sync) SYNC=0; shift ;;
    --cuda)    SYNC_ARGS+=(--extra gpu); shift ;;
    --push)    PUSH=1; shift ;;
    --commit)  COMMIT=1; shift ;;
    --)        shift; DVC_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

if [[ "$COMMIT" -eq 1 && "$PUSH" -eq 0 ]]; then
  echo "note: --commit without --push — dvc.lock will be committed but the data" \
       "it points to is only in the local cache, not on the remote." >&2
fi

# Git-commit a chapter's DVC lockfiles (dvc.lock + the dvc-managed .gitignore)
# so the commit points at exactly the data just regenerated/pushed. Scoped to
# those files only — never stages source code or cache:false metric files, and
# commits nothing if the lockfile is unchanged.
commit_lock() {
  local ch="$1"
  local paths=()
  [[ -f "$ROOT/$ch/dvc.lock" ]]   && paths+=("$ch/dvc.lock")
  [[ -f "$ROOT/$ch/.gitignore" ]] && paths+=("$ch/.gitignore")
  [[ ${#paths[@]} -eq 0 ]] && return 0
  git -C "$ROOT" add -- "${paths[@]}"
  if git -C "$ROOT" diff --cached --quiet -- "${paths[@]}"; then
    echo "--- $ch: dvc.lock unchanged, nothing to commit"
  else
    git -C "$ROOT" commit -q -m "$ch: update dvc.lock (reproduce-all)" -- "${paths[@]}"
    echo "--- $ch: committed ${paths[*]}"
    committed_any=1
  fi
}

# --- run --------------------------------------------------------------------
failed=()
committed_any=0
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

  if (
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
  ); then
    [[ "$COMMIT" -eq 1 ]] && commit_lock "$ch"
  else
    failed+=("$ch"); echo "!! $ch FAILED" >&2
  fi
done

if [[ "$COMMIT" -eq 1 && "$committed_any" -eq 1 ]]; then
  echo
  echo "--- git push"
  git -C "$ROOT" push
fi

# --- summary ----------------------------------------------------------------
echo
echo "==================================================================="
if [[ ${#failed[@]} -eq 0 ]]; then
  echo ">>> all chapters reproduced successfully"
else
  echo ">>> FAILURES: ${failed[*]}"
  exit 1
fi
