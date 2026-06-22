#!/usr/bin/env python3

#
# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License
#

import argparse
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# --- Which MPI calls count as collectives ---
COLLECTIVES = {
    # "MPI_Barrier", 
    "MPI_Bcast",
    "MPI_Allreduce", 
    "MPI_Iallreduce", 
    "MPI_Alltoall", 
    "MPI_Alltoallv", 
    "MPI_Allgather", 
    "MPI_Allgatherv", 
    "MPI_Gather", 
    "MPI_Gatherv", 
    "MPI_Scatter", 
    "MPI_Scatterv", 
    "MPI_Reduce", 
    "MPI_Scan", 
    "MPI_Exscan",
}

RANK_RE = re.compile(r"rank-(\d+)", re.IGNORECASE)

def is_trace_line(s: str) -> bool:
    return s.startswith("MPI_")

def parse_start_end(tokens):
    def to_float(x):
        if x in (None, "-", ""):
            return None
        try:
            return float(x)
        except ValueError:
            return None
    start = to_float(tokens[1]) if len(tokens) > 1 else None
    end   = to_float(tokens[-1]) if len(tokens) > 1 else None
    return start, end

def detect_unit_scale(earliest, latest) -> float:
    """
    Heuristic: if overall span is large (>= 1e5), assume microseconds and convert to seconds (1e-6).
    Otherwise assume seconds.
    """
    if earliest is None or latest is None:
        return 1.0
    span = latest - earliest
    return 1e-6 if span >= 1e5 else 1.0

def analyze_trace_file(path: Path):
    total_ops = coll_ops = 0
    coll_time_raw = 0.0
    earliest = latest = None

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not is_trace_line(line):
                    continue
                parts = line.split(":")
                if not parts:
                    continue
                func = parts[0]
                start, end = parse_start_end(parts)

                if start is not None:
                    earliest = start if earliest is None else min(earliest, start)
                cand_latest = end if end is not None else start
                if cand_latest is not None:
                    latest = cand_latest if latest is None else max(latest, cand_latest)

                total_ops += 1

                if func in COLLECTIVES and start is not None and end is not None and end >= start:
                    coll_ops += 1
                    coll_time_raw += (end - start)

        total_time_raw = (latest - earliest) if (earliest is not None and latest is not None) else None
        scale = detect_unit_scale(earliest, latest)

        rank = int(RANK_RE.search(path.name).group(1)) if RANK_RE.search(path.name) else None
        coll_time_s = coll_time_raw * scale
        total_time_s = total_time_raw * scale if total_time_raw is not None else None
        coll_time_pct = (coll_time_s / total_time_s * 100.0) if total_time_s and total_time_s > 0 else 0.0

        return dict(
            path=str(path),
            rank=rank,
            total_ops=total_ops,
            coll_ops=coll_ops,
            coll_ops_pct=(coll_ops / total_ops * 100.0) if total_ops else 0.0,
            coll_time_s=coll_time_s,
            total_time_s=total_time_s,
            coll_time_pct=coll_time_pct,
        )
    except Exception as e:
        return dict(
            path=str(path), rank=None, error=str(e),
            total_ops=0, coll_ops=0, coll_ops_pct=0.0,
            coll_time_s=0.0, total_time_s=None, coll_time_pct=0.0,
        )

def breakdown_collectives_for_file(path: Path):
    """
    Returns:
      counts: Counter(collective -> count)
      times_s: dict(collective -> time in seconds)
      total_count, total_time_s
    """
    counts = Counter()
    times_raw = defaultdict(float)
    earliest = latest = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not is_trace_line(line):
                continue
            parts = line.split(":")
            if not parts:
                continue
            func = parts[0]
            start, end = parse_start_end(parts)

            # track bounds to pick unit
            if start is not None:
                earliest = start if earliest is None else min(earliest, start)
            cand_latest = end if end is not None else start
            if cand_latest is not None:
                latest = cand_latest if latest is None else max(latest, cand_latest)

            if func not in COLLECTIVES:
                continue

            counts[func] += 1
            if start is not None and end is not None and end >= start:
                times_raw[func] += (end - start)

    scale = detect_unit_scale(earliest, latest)
    times_s = {k: v * scale for k, v in times_raw.items()}
    total_count = sum(counts.values())
    total_time_s = sum(times_s.values())
    return counts, times_s, total_count, total_time_s


def collect_allreduce_dimensions(path: Path):
    """
    Build a histogram of MPI_Allreduce message 'dimensions', i.e., element
    counts and datatype byte sizes. Returns (hist Counter, total_allreduces, errors).
    """
    hist = Counter()
    total = 0
    errors = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("MPI_Allreduce:"):
                continue
            parts = line.split(":")
            if len(parts) < 6:
                errors += 1
                continue
            try:
                elem_count = int(parts[4])
            except ValueError:
                errors += 1
                continue
            dtype_token = parts[5]
            dtype_fields = dtype_token.split(",")
            dtype_bytes = None
            if len(dtype_fields) >= 2:
                try:
                    dtype_bytes = int(dtype_fields[1])
                except ValueError:
                    dtype_bytes = None
            key = (elem_count, dtype_bytes, dtype_token)
            hist[key] += 1
            total += 1
    return hist, total, errors

def human_s(x):
    if x is None:
        return "-"
    return f"{x:8.4f}s"

def main():
    ap = argparse.ArgumentParser(
        description="Per-run summary (Avg/Min/Max % collective time) + top-N collective breakdowns."
    )
    ap.add_argument("root", type=str, help="Root directory (searched recursively).")
    ap.add_argument("--name-glob", default="pmpi-trace-rank-*.txt",
                    help="Filename glob for per-rank traces (default: pmpi-trace-rank-*.txt).")
    ap.add_argument("--mpi-traces-dir", default="mpi_traces",
                    help="Directory that holds rank traces (default: mpi_traces).")
    ap.add_argument("--top", type=int, default=3, help="Top-N runs to break down (default: 3).")
    ap.add_argument("--show-summary", action="store_true",
                    help="Print the per-run summary table (Avg/Min/Max).")
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional CSV file path to save the breakdown rows.")
    ap.add_argument(
        "--show-allreduce-dims",
        action="store_true",
        help="Print the MPI_Allreduce element/datatype distribution for each top run.",
    )
    ap.add_argument(
        "--allreduce-dims-limit",
        type=int,
        default=10,
        help="Maximum rows to show per run when --show-allreduce-dims is enabled (0 = unlimited).",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Discover all per-rank files grouped by run (parent of mpi_traces)
    per_run = defaultdict(list)
    for p in root.rglob(args.name_glob):
        if not p.is_file() or p.parent.name != args.mpi_traces_dir:
            continue
        run_dir = p.parent.parent
        stats = analyze_trace_file(p)
        per_run[str(run_dir)].append(stats)

    if not per_run:
        print("No trace files found.")
        sys.exit(0)

    # Build summary (avg/min/max % collective time across ranks) and representatives
    summary_rows = []
    representatives = []
    for run, lst in per_run.items():
        vals = [s["coll_time_pct"] for s in lst if not s.get("error")]
        nvals = len(vals)
        if nvals:
            vals_arr = np.array(vals)
            avgv = float(vals_arr.mean())
            minv = float(vals_arr.min())
            maxv = float(vals_arr.max())
            q1v = float(np.percentile(vals_arr, 25))
            q3v = float(np.percentile(vals_arr, 75))
            median = float(np.median(vals_arr))

            # Add Q1 and Q3 to summary
            summary_rows.append(
                (Path(run).name, avgv, minv, median, q1v, q3v, maxv, nvals)
            )
        # representative rank = highest % collective time
        best = None
        for s in lst:
            if s.get("error"):
                continue
            if best is None or s["coll_time_pct"] > best["coll_time_pct"]:
                best = s
        if best:
            representatives.append({
                "run": Path(run).name,
                "rank": best["rank"],
                "path": best["path"],
                "coll_time_pct": best["coll_time_pct"],
                "coll_ops_pct": best["coll_ops_pct"],
                "coll_time_s": best["coll_time_s"],
                "total_time_s": best["total_time_s"],
                "coll_ops": best["coll_ops"],
                "total_ops": best["total_ops"],
            })


    # Rank runs by representative % collective time and pick top-N
    representatives.sort(key=lambda r: r["coll_time_pct"], reverse=True)

    # Show ranking (like earlier)
    print("\nRepresentative-rank ranking (by % collective time):")
    print(f"{'#':>4}  {'%CollTime':>10}  {'%CollOps':>9}  {'CollTime':>10}  "
          f"{'TotalTime':>10}  {'CollOps':>8}  {'TotOps':>8}  {'Rank':>5}  {'Run':<20}")
    print("-" * 90)
    for i, r in enumerate(representatives, 1):
        rep_rank_str = f"{r['rank']:>5d}" if r['rank'] is not None else f"{-1:>5d}"
        print(f"{i:>4d}  {r['coll_time_pct']:10.3f}%  {r['coll_ops_pct']:9.3f}%  "
              f"{human_s(r['coll_time_s']):>10}  {human_s(r['total_time_s']):>10}  "
              f"{r['coll_ops']:8d}  {r['total_ops']:8d}  {rep_rank_str}  {r['run']:<20}")

    # Breakdown for top-N
    topN = representatives[: max(0, args.top)]
    csv_rows = []

    # Print per-run summary (Avg/Min/Max) if requested
    print("\n" + "=" * 90 + "\n"+ "+" * 90 + "\n" + "=" * 90)
    if args.show_summary:
        print("\nPer-run summary (Avg / Min / Median / Q1 / Q3 / Max collective %):")
        print(f"{'Run':<20} {'Avg':>8} {'Min':>8} {'Median':>8} {'Q1':>8} {'Q3':>8} {'Max':>8} {'Ranks':>7}")
        print("-" * 85)
        for run, avgv, minv, median, q1v, q3v, maxv, nvals in sorted(summary_rows):
            print(f"{run:<20} {avgv:8.2f} {minv: 8.2f} {median:8.2f} {q1v:8.2f} {q3v:8.2f} {maxv:8.2f} {nvals:7d}")


    print("\n" + "=" * 90 + "\n"+ "+" * 90 + "\n" + "=" * 90)
    print(f"\nCollective breakdown for top runs (representative rank only):")
    print("-" * 75)
    for idx, r in enumerate(topN, 1):
        path = Path(r["path"])
        counts, times_s, total_count, total_time_s = breakdown_collectives_for_file(path)

        print(f"[{idx}] Run: {r['run']}   Representative rank: {r['rank']}   File: {path.name}")
        print(f"{'Collective':<16} {'Count':>8} {'% of Colls':>12} {'Time (s)':>12} {'% Coll Time':>14}")
        print("-" * 75)

        # sort by % collective time descending
        rows = []
        for coll in sorted(counts.keys()):
            c = counts[coll]
            t = times_s.get(coll, 0.0)
            pct_c = (c / total_count * 100.0) if total_count else 0.0
            pct_t = (t / total_time_s * 100.0) if total_time_s else 0.0
            rows.append((coll, c, pct_c, t, pct_t))
        rows.sort(key=lambda x: x[4], reverse=True)

        for coll, c, pct_c, t, pct_t in rows:
            print(f"{coll:<16} {c:8d} {pct_c:12.2f}% {t:12.6f} {pct_t:14.2f}%")
            if args.csv:
                csv_rows.append({
                    "run": r["run"], "rep_rank": r["rank"], "file": path.name,
                    "collective": coll, "count": c,
                    "pct_of_collectives": round(pct_c, 4),
                    "time_s": round(t, 6),
                    "pct_of_collective_time": round(pct_t, 4)
                })

        print("-" * 75)
        print(f"{'TOTAL':<16} {total_count:8d} {100.00:12.2f}% {total_time_s:12.6f} {100.00:14.2f}%\n")

        if args.show_allreduce_dims:
            try:
                hist, total_allr, dim_errors = collect_allreduce_dimensions(path)
            except OSError as exc:
                print(f"    Failed to read MPI_Allreduce dimensions: {exc}")
                continue

            print("    MPI_Allreduce dimension distribution:")
            if total_allr == 0:
                print("      (no MPI_Allreduce calls in this trace)")
            else:
                rows = []
                for (elem_count, dtype_bytes, dtype_token), freq in hist.items():
                    msg_bytes = (
                        elem_count * dtype_bytes
                        if (elem_count is not None and dtype_bytes is not None)
                        else None
                    )
                    pct = freq / total_allr * 100.0 if total_allr else 0.0
                    rows.append(
                        {
                            "elem_count": elem_count,
                            "dtype_bytes": dtype_bytes,
                            "msg_bytes": msg_bytes,
                            "freq": freq,
                            "pct": pct,
                            "dtype_token": dtype_token,
                        }
                    )
                rows.sort(key=lambda r: (-r["freq"], r["elem_count"]))

                limit = args.allreduce_dims_limit or 0
                display_rows = rows if limit <= 0 else rows[:limit]

                header = (
                    f"{'Elems':>10} {'DTypeB':>8} {'MsgB':>10} "
                    f"{'Hits':>8} {'Pct':>7}  {'Datatype token'}"
                )
                print("      " + header)
                print("      " + "-" * (len(header) + 5))
                for row in display_rows:
                    dtype_b = row["dtype_bytes"] if row["dtype_bytes"] is not None else "-"
                    msg_b = row["msg_bytes"] if row["msg_bytes"] is not None else "-"
                    print(
                        f"      {row['elem_count']:10d} {dtype_b:>8} "
                        f"{msg_b:>10} {row['freq']:8d} {row['pct']:7.2f}%  "
                        f"{row['dtype_token']}"
                    )
                if limit > 0 and len(rows) > limit:
                    hidden = len(rows) - limit
                    print(f"      ... ({hidden} more combinations)")
            if dim_errors:
                print(f"      Skipped {dim_errors} malformed MPI_Allreduce entries.")
            print()

        if args.csv and total_count:
            csv_rows.append({
                "run": r["run"], "rep_rank": r["rank"], "file": path.name,
                "collective": "TOTAL", "count": total_count,
                "pct_of_collectives": 100.0, "time_s": round(total_time_s, 6),
                "pct_of_collective_time": 100.0
            })


    # Optional CSV export
    if args.csv:
        out = Path(args.csv)
        with out.open("w", encoding="utf-8") as f:
            f.write("run,rep_rank,file,collective,count,pct_of_collectives,time_s,pct_of_collective_time\n")
            for row in csv_rows:
                f.write(f"{row['run']},{row['rep_rank']},{row['file']},{row['collective']},"
                        f"{row['count']},{row['pct_of_collectives']},{row['time_s']},"
                        f"{row['pct_of_collective_time']}\n")
        print(f"\nSaved CSV to: {out}")

if __name__ == "__main__":
    main()
