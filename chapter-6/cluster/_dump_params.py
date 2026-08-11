"""Print every `params.yaml` leaf as `<key-path>\tvalue`, for the job scripts.

`cluster/_read_params.sh` reads this into a bash associative array once per
SLURM task, so that `read_param "['env']['K']"` becomes a lookup rather than
its own `uv run python -c`. The key-path syntax is the Python subscript chain
the job scripts already pass, so no call site changes.

Why it exists: the packed CPU jobs read ~24 values to build `COMMON_ARGS`,
and each one used to be its own interpreter. At `pack=10` with several
planners' arrays in flight that is thousands of `uv run` startups walking
`.venv-cpu` on GPFS at once — the metadata storm `setup_env_cpu.sh` warns
about, which is what was getting these jobs killed on genoa.

Values are formatted exactly as the old inline `python -c` formatted them:
`str(v).lower()` for bools (so `False` reaches argparse as `false`), plain
`str(v)` otherwise. Non-dict leaves are not descended into, so a list prints
as its repr — again matching the old behaviour.

Usage:

    uv run --frozen --no-sync python cluster/_dump_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

import yaml


def walk(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield `(key_path, formatted_value)` for every non-dict leaf."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, f"{path}['{key}']")
    else:
        yield path, str(node).lower() if isinstance(node, bool) else str(node)


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    for key_path, value in walk(params):
        # Tab-separated, so a value carrying one would silently split into a
        # wrong key/value pair in the shell rather than fail. params.yaml is
        # hand-edited, so this is worth catching here instead of downstream
        # as an unexplained CLI argument.
        if any(c in value for c in "\t\n\r"):
            sys.exit(f"params.yaml {key_path} contains a tab or newline: {value!r}")
        print(f"{key_path}\t{value}")


if __name__ == "__main__":
    main()
