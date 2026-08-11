#!/usr/bin/env bash
# Read params.yaml once into $PARAMS_TSV, and define read_param().
#
# Source this from a job script after cluster/setup_env_cpu.sh:
#
#   source cluster/_read_params.sh
#   K="$(read_param "['env']['K']")"
#
# read_param takes the same Python subscript chain the job scripts have
# always passed, and formats values identically (bools lowercased), so it
# is a drop-in for the per-value `uv run python -c` it replaces. The point
# is one interpreter per SLURM task instead of one per CLI flag: a packed
# block builds COMMON_ARGS from ~24 of these, and at pack=10 with several
# planners' arrays in flight the old form put thousands of `uv run`
# startups against `.venv-cpu` on GPFS at once. That is the metadata storm
# setup_env_cpu.sh warns about, and it is what was getting genoa jobs
# killed the moment they left the queue.
#
# Lookups go through awk rather than a bash associative array so this file
# runs under bash 3.2 as well as the cluster's 4.4+. That is not
# portability for its own sake: `declare -A` cannot be exercised on a
# stock macOS shell, and the equivalence of old and new read_param is
# exactly the property that has to be checkable off the cluster. An awk
# fork per value is still ~24 forks, but of a system binary, and the cost
# being removed here is walking a 15k-file venv on GPFS — not the fork.
#
# Field-exact match, not a substring one: `['env']['K']` is a suffix of
# `['minigrid']['env']['K']`, so a grep would silently return the wrong
# benchmark's value.
#
# Meant to be *sourced*, so no `set -euo pipefail` here — that would apply
# to the caller. Callers already enable strict mode themselves.

PARAMS_TSV="$(uv run --frozen --no-sync python cluster/_dump_params.py)"

if [ -z "$PARAMS_TSV" ]; then
    echo "cluster/_dump_params.py produced no values — params.yaml unreadable?" >&2
    exit 1
fi

# A miss returns empty rather than failing, because every call site is a
# `$(read_param ...)` command substitution and an `exit` there would only
# leave the subshell — the caller would carry on with an empty argument.
# Job scripts guard the assembled array instead; see require_all_set.
read_param() {
    awk -F'\t' -v key="$1" '$1 == key { print $2; exit }' <<< "$PARAMS_TSV"
}

# Fail on an empty element of the assembled argument list. That is what a
# mistyped key path looks like by the time it reaches the runner: argparse
# would silently take the *next* flag as the value ("--K --lock-length"),
# so the run would start and quietly measure the wrong configuration.
require_all_set() {
    local i
    for i in "$@"; do
        if [ -z "$i" ]; then
            echo "empty value in the assembled argument list — a read_param" \
                 "key path does not name a params.yaml leaf" >&2
            exit 1
        fi
    done
}
