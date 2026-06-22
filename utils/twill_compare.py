# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

"""
twill_compare.py — quick text comparison of PICO alltoall results (no plotting).

Reads PICO per-test result CSVs (the `highest` column = max-across-ranks time per
iteration, in ns) from result directories OR `.tar.gz` archives (the run configs
compress+delete by default, so archives are the normal case), summarizes each
(algorithm, message size), and prints a table. Runs are labelled by the NOTES
column of results/<system>_metadata.csv (e.g. "twill W=8 group=node cache=1"),
so window/cache/group sweep points are distinguishable.

Stdlib only — runs anywhere (incl. a login node with no pip).

Examples:
  # newest run under results/* (auto-discovered):
  python3 utils/twill_compare.py
  # specific runs (dirs or tarballs); size x algorithm table each:
  python3 utils/twill_compare.py results/leonardo/2026_*_W8_*.tar.gz
  # how one algorithm varies across the sweep (rows = run/NOTES, cols = size):
  python3 utils/twill_compare.py --cross twill_group_over results/leonardo/*.tar.gz
  # add speedup vs a baseline column-set:
  python3 utils/twill_compare.py --baseline default_ompi <run>
"""

import argparse
import csv
import glob
import os
import statistics
import sys
import tarfile
from typing import Dict, List, Optional, Tuple

TYPE_BYTES = {"int8": 1, "int16": 2, "int32": 4, "int64": 8,
              "int": 4, "float": 4, "double": 8, "char": 1}

# Short display labels for known algorithms (full name used as fallback).
SHORT = {
    "default_ompi": "ompi_def", "linear_ompi": "ompi_lin",
    "pairwise_ompi": "ompi_pair", "modified_bruck_ompi": "ompi_bruck",
    "linear_sync_ompi": "ompi_lsync", "bine_over": "bine",
    "pairwise_ompi_over": "pico_pair", "twill_shift_over": "t_shift",
    "twill_random_over": "t_random", "twill_group_over": "t_group",
}


def fmt_size(count: int, comm_sz: int, tbytes: Optional[int]) -> str:
    """Per-pair size label if we know comm_sz and the element size, else total count."""
    if comm_sz and tbytes:
        per_pair = (count // comm_sz) * tbytes
        # simple human-readable
        v = per_pair
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if v < 1024 or unit == "TiB":
                return f"{v:.0f} {unit}" if unit != "B" else f"{int(v)} B"
            v /= 1024.0
    return f"count={count}"


def parse_csv_text(text: str) -> Tuple[int, List[int]]:
    """Return (comm_sz, list of 'highest' ns values) from a PICO result CSV string."""
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return 0, []
    header = lines[0].split(",")
    comm_sz = len(header) - 1 if header[0].strip() == "highest" else 0
    vals = []
    for ln in lines[1:]:
        first = ln.split(",", 1)[0].strip()
        try:
            vals.append(int(first))
        except ValueError:
            pass
    return comm_sz, vals


def parse_name(fname: str) -> Optional[Tuple[int, str, str]]:
    """Parse '<count>_<algo>_<type>.csv' -> (count, algo, type); None if not a result CSV."""
    base = os.path.basename(fname)
    if not base.endswith(".csv") or base.endswith("_instrument.csv"):
        return None
    stem = base[:-4]
    toks = stem.split("_")
    if len(toks) < 3 or not toks[0].isdigit():
        return None  # skips alloc_*.csv etc.
    return int(toks[0]), "_".join(toks[1:-1]), toks[-1]


# data[algo][count] = (comm_sz, type, [highest_ns...])
RunData = Dict[str, Dict[int, Tuple[int, str, List[int]]]]


def _ingest(data: RunData, parsed, comm_sz: int, vals: List[int]):
    count, algo, typ = parsed
    if not vals:
        return
    slot = data.setdefault(algo, {})
    prev = slot.get(count)
    merged = (prev[2] + vals) if prev else vals
    slot[count] = (comm_sz or (prev[0] if prev else 0), typ, merged)


def read_run(path: str) -> RunData:
    data: RunData = {}
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                p = parse_name(f)
                if p is None:
                    continue
                with open(os.path.join(root, f)) as fh:
                    comm_sz, vals = parse_csv_text(fh.read())
                _ingest(data, p, comm_sz, vals)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                p = parse_name(m.name)
                if p is None:
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                comm_sz, vals = parse_csv_text(fh.read().decode("utf-8", "replace"))
                _ingest(data, p, comm_sz, vals)
    else:
        sys.stderr.write(f"warning: not a dir or tarball, skipping: {path}\n")
    return data


def run_meta(path: str) -> Tuple[str, str]:
    """(timestamp, NOTES) for a run, reading results/<system>_metadata.csv best-effort."""
    ap = os.path.abspath(path.rstrip("/"))
    base = os.path.basename(ap)
    ts = base[:-7] if base.endswith(".tar.gz") else base
    system_dir = os.path.dirname(ap)
    meta = os.path.join(os.path.dirname(system_dir), f"{os.path.basename(system_dir)}_metadata.csv")
    note = ""
    try:
        with open(meta, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("timestamp") == ts and row.get("notes") not in (None, "", "null"):
                    note = row["notes"]
                    break
    except (OSError, csv.Error):
        pass
    return ts, note


def run_label(path: str) -> str:
    """timestamp + NOTES (from results/<system>_metadata.csv), best-effort."""
    ts, note = run_meta(path)
    return f"{ts}  [{note}]" if note else ts


def summary(vals: List[int], stat: str) -> float:
    """ns -> us for the chosen statistic."""
    if stat == "min":
        return min(vals) / 1000.0
    if stat == "mean":
        return statistics.fmean(vals) / 1000.0
    return statistics.median(vals) / 1000.0


def discover() -> List[str]:
    runs = []
    for entry in sorted(glob.glob("results/*/*")):
        if entry.endswith("_metadata.csv"):
            continue
        if os.path.isdir(entry) or entry.endswith(".tar.gz"):
            runs.append(entry)
    return runs


def all_counts(data: RunData) -> List[int]:
    cs = set()
    for slot in data.values():
        cs.update(slot.keys())
    return sorted(cs)


def col(s: str, w: int) -> str:
    return s[:w].rjust(w)


def print_run_table(path: str, data: RunData, stat: str, baseline: Optional[str],
                    base_data: Optional[RunData] = None):
    # Merge an optional baseline run (its knob-invariant algos) under this run's
    # data; the run's own algos win for any shared name (e.g. the twill variants).
    merged: RunData = {}
    from_base = set()
    if base_data:
        for a, slot in base_data.items():
            merged[a] = slot
            from_base.add(a)
    for a, slot in data.items():
        merged[a] = slot
        from_base.discard(a)
    if not merged:
        print(f"\n=== {run_label(path)} ===\n  (no result CSVs found)")
        return
    algos = sorted(merged.keys(), key=lambda a: (a not in SHORT, a))
    counts = all_counts(merged)
    comm_sz, tbytes = 0, None
    for slot in merged.values():
        for (cs, typ, _) in slot.values():
            comm_sz = comm_sz or cs
            tbytes = tbytes or TYPE_BYTES.get(typ)
    w = 11

    def head(a: str) -> str:
        return SHORT.get(a, a) + ("^" if a in from_base else "")

    print(f"\n=== {run_label(path)} ===")
    print(f"  comm_sz={comm_sz or '?'}  stat={stat}  unit=us  (* = fastest in row"
          + ("; ^ = from --with-baseline run)" if from_base else ")"))
    print("  " + "size/pair".ljust(12) + "".join(col(head(a), w) for a in algos))
    for count in counts:
        times = {a: (summary(merged[a][count][2], stat) if count in merged[a] else None) for a in algos}
        best = min((t for t in times.values() if t is not None), default=None)
        cells = []
        for a in algos:
            t = times[a]
            if t is None:
                cells.append(col("-", w))
            else:
                mark = "*" if best is not None and abs(t - best) < 1e-9 else " "
                cells.append(col(f"{t:.2f}{mark}", w))
        print("  " + fmt_size(count, comm_sz, tbytes).ljust(12) + "".join(cells))

    if baseline and baseline in merged:
        print(f"  -- speedup vs {SHORT.get(baseline, baseline)} (>1 = faster) --")
        for count in counts:
            if count not in merged[baseline]:
                continue
            base_t = summary(merged[baseline][count][2], stat)
            cells = [col(f"{base_t / summary(merged[a][count][2], stat):.2f}x", w)
                     if count in merged[a] else col("-", w) for a in algos]
            print("  " + fmt_size(count, comm_sz, tbytes).ljust(12) + "".join(cells))


def best_across_runs(runs: List[RunData], stat: str) -> RunData:
    """Merge runs keeping, per (algo, count), the vals from the run with the lowest stat."""
    algos = set(a for d in runs for a in d)
    best: RunData = {}
    for algo in algos:
        for d in runs:
            if algo not in d:
                continue
            for count, (cs, typ, vals) in d[algo].items():
                t = summary(vals, stat)
                prev = best.get(algo, {}).get(count)
                if prev is None or t < summary(prev[2], stat):
                    best.setdefault(algo, {})[count] = (cs, typ, vals)
    return best


def print_cross(paths: List[str], runs: List[RunData], algo: str, stat: str):
    """rows = run (label), cols = size, cell = `algo` time (us)."""
    # union of counts across runs (for this algo)
    counts = sorted({c for d in runs for c in d.get(algo, {})})
    if not counts:
        print(f"(no data for algorithm '{algo}' in the given runs)")
        return
    comm_sz, tbytes = 0, None
    for d in runs:
        for (cs, typ, _) in d.get(algo, {}).values():
            comm_sz = comm_sz or cs
            tbytes = tbytes or TYPE_BYTES.get(typ)
    w = 11
    # label each run by its NOTES (the knob differentiator) when available, else timestamp
    labels = []
    for p in paths:
        ts, note = run_meta(p)
        labels.append(note if note else ts)
    lw = max((len(x) for x in labels), default=10)
    lw = min(max(lw, 10), 60)
    print(f"\n=== cross-run: {SHORT.get(algo, algo)}  (stat={stat}, unit=us) ===")
    print("  " + "run \\ size".ljust(lw) + "".join(col(fmt_size(c, comm_sz, tbytes), w) for c in counts))
    for label, d in zip(labels, runs):
        row = [col(f"{summary(d[algo][c][2], stat):.2f}", w) if c in d.get(algo, {}) else col("-", w)
               for c in counts]
        print("  " + label.ljust(lw)[:lw] + "".join(row))


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick text comparison of PICO alltoall results.")
    ap.add_argument("paths", nargs="*", help="result dirs or .tar.gz (default: auto-discover under results/*/)")
    ap.add_argument("--stat", choices=["median", "min", "mean"], default="median")
    ap.add_argument("--baseline", default=None, help="algorithm to show speedups against (per-run table)")
    ap.add_argument("--cross", default=None, metavar="ALGO",
                    help="pivot: rows=run, cols=size, cell=ALGO time (good for the sweep)")
    ap.add_argument("--with-baseline", default=None, metavar="RUN",
                    help="overlay knob-invariant algos (vendor/bine) from a separate full-field "
                         "run, marked '^'. Caveat: that run is a different allocation, so its "
                         "absolute times aren't strictly comparable to the swept twill ones.")
    ap.add_argument("--latest", action="store_true", help="only the most recent discovered run")
    ap.add_argument("--best", action="store_true",
                    help="merge all runs into one table using the best time per (algo, size) "
                         "across all provided runs — e.g. t_group shows its optimal W/G at "
                         "every size rather than one fixed combo")
    args = ap.parse_args()

    paths = args.paths or discover()
    if not paths:
        sys.exit("no result dirs/tarballs found (looked under results/*/). Pass paths explicitly.")
    if args.latest and not args.paths:
        paths = [max(paths, key=lambda p: os.path.getmtime(p))]

    runs = [read_run(p) for p in paths]
    base_data = read_run(args.with_baseline) if args.with_baseline else None

    if args.cross:
        print_cross(paths, runs, args.cross, args.stat)
    elif args.best:
        merged = best_across_runs(runs, args.stat)
        label = f"best across {len(runs)} run(s)"
        print_run_table(label, merged, args.stat, args.baseline, base_data)
    else:
        for p, d in zip(paths, runs):
            print_run_table(p, d, args.stat, args.baseline, base_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
