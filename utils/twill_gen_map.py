# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

"""
twill_gen_map.py — build a TWILL_MAP file (rank -> group id) for the TWILL
alltoall algorithms in libpico.

The map file has one integer per line; line i is the group id of MPI rank i.
Group ids may be arbitrary (libpico densifies them); this script emits dense
ids 0..G-1 in sorted-key order so the file is human-readable and stable.

The "group" of a rank is derived from its hostname:
  * with --regex, the concatenation of the regex capture groups is the group
    key (e.g. extract the dragonfly group field from a site nodename);
  * without --regex, the whole hostname is the key (node = group), which is the
    generic fallback and matches libpico's own TWILL_GROUP=node default.

Per-rank hostnames come from exactly one source:
  --alloc-csv FILE   PICO alloc_<n>.csv ("rank,hostname[,xname]" lines) — the
                     most reliable, since it reflects the actual placement.
  --hostfile FILE    one hostname per line, already in rank order; or one node
                     per line if --tasks-per-node is given (block expansion).
  --from-slurm       expand $SLURM_JOB_NODELIST via `scontrol show hostnames`,
                     then block-expand by --tasks-per-node.

Examples:
  # node = group, from a PICO allocation file
  python3 utils/twill_gen_map.py --alloc-csv results/.../alloc_512.csv --out twill.map

  # dragonfly group = digits after the 'g' in nodenames like nid-g03-c1
  python3 utils/twill_gen_map.py --from-slurm --tasks-per-node 4 \\
          --regex 'g(\\d+)' --out twill.map

Then:  TWILL_MAP=twill.map mpirun ... pico_core ... twill_group_over ...
"""

import argparse
import os
import re
import subprocess
import sys
from typing import List, Optional


def parse_alloc_csv(path: str) -> List[str]:
    """Read a PICO alloc CSV ('rank,hostname[,xname]') into rank-ordered hosts."""
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # Skip a possible header row.
            if not parts[0].lstrip("-").isdigit():
                continue
            rows.append((int(parts[0]), parts[1]))
    rows.sort(key=lambda r: r[0])
    if [r[0] for r in rows] != list(range(len(rows))):
        sys.exit(f"error: alloc csv ranks are not a dense 0..N-1 range: {path}")
    return [host for _, host in rows]


def read_hostfile(path: str, tasks_per_node: Optional[int]) -> List[str]:
    """Read a hostfile: per-rank if no tasks_per_node, else per-node block-expanded."""
    with open(path) as f:
        hosts = [ln.strip() for ln in f if ln.strip()]
    if tasks_per_node:
        return [h for h in hosts for _ in range(tasks_per_node)]
    return hosts


def slurm_hostnames(tasks_per_node: int) -> List[str]:
    """Expand $SLURM_JOB_NODELIST via scontrol, block-expanded by tasks_per_node."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        sys.exit("error: --from-slurm needs $SLURM_JOB_NODELIST in the environment")
    try:
        out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"error: `scontrol show hostnames` failed: {e}")
    nodes = [ln.strip() for ln in out.decode().splitlines() if ln.strip()]
    if tasks_per_node < 1:
        sys.exit("error: --tasks-per-node must be >= 1 with --from-slurm")
    return [n for n in nodes for _ in range(tasks_per_node)]


def group_key(host: str, regex: Optional[re.Pattern]) -> str:
    """Group key for a host: joined regex captures, or the whole hostname."""
    if regex is None:
        return host
    m = regex.search(host)
    if not m:
        return host  # unmatched hosts fall back to node=group
    groups = [g for g in m.groups() if g is not None]
    return "|".join(groups) if groups else m.group(0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a TWILL_MAP (rank->group) file.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--alloc-csv", help="PICO alloc_<n>.csv (rank,hostname[,xname])")
    src.add_argument("--hostfile", help="hostnames in rank order (or per node with --tasks-per-node)")
    src.add_argument("--from-slurm", action="store_true", help="expand $SLURM_JOB_NODELIST")
    ap.add_argument("--tasks-per-node", type=int, default=None,
                    help="ranks per node for block expansion (--hostfile/--from-slurm)")
    ap.add_argument("--regex", default=None,
                    help="regex whose capture groups form the group key (default: whole hostname)")
    ap.add_argument("--out", required=True, help="output TWILL_MAP path")
    args = ap.parse_args()

    if args.alloc_csv:
        hosts = parse_alloc_csv(args.alloc_csv)
    elif args.hostfile:
        hosts = read_hostfile(args.hostfile, args.tasks_per_node)
    else:
        hosts = slurm_hostnames(args.tasks_per_node or 1)

    if not hosts:
        sys.exit("error: no hostnames resolved")

    regex = None
    if args.regex is not None:
        try:
            regex = re.compile(args.regex)
        except re.error as e:
            sys.exit(f"error: invalid --regex: {e}")

    keys = [group_key(h, regex) for h in hosts]
    # Dense ids 0..G-1 in sorted-key order (stable, readable).
    order = {k: i for i, k in enumerate(sorted(set(keys)))}
    ids = [order[k] for k in keys]

    with open(args.out, "w") as f:
        for gid in ids:
            f.write(f"{gid}\n")

    n_groups = len(order)
    sizes = [ids.count(g) for g in range(n_groups)]
    print(f"wrote {args.out}: {len(ids)} ranks, {n_groups} groups, "
          f"sizes min/max={min(sizes)}/{max(sizes)}", file=sys.stderr)
    if args.regex and all(k == h for k, h in zip(keys, hosts)):
        print("warning: --regex matched no hostname; fell back to node=group",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
