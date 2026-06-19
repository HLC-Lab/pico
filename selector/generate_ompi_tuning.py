#!/usr/bin/env python3
"""
Generate Open MPI tuning rules from benchmark results.

For each collective / communicator size / message size tuple, selects the
best-performing OMPI algorithm from the benchmark data in results/<system>/.
Produces a dynamic rules file consumable by
OMPI_MCA_coll_tuned_dynamic_rules_filename.

filtering:
  Each run in the metadata CSV has a 'notes' column with free-text tags
  (e.g. "UCX_MAX_RNDV_RAILS=1", "rail 4", "vsnccl").  You can include or
  exclude entire timestamps based on regex matches against these notes.

Examples:
  # Only runs tagged with a specific rail config
   python selector/generate_ompi_tuning.py leonardo --include-notes "rail 4"

    # Exclude experimental/test runs
    python selector/generate_ompi_tuning.py leonardo --exclude-notes "testing|fork"

   # Exclude a range of timestamps
   python selector/generate_ompi_tuning.py leonardo -x 2025_04_04___18_21_29-2025_04_06___20_44_58

  # Discover available notes values
  python selector/generate_ompi_tuning.py leonardo --list-notes

   # Single-rail runs with custom output path
   python selector/generate_ompi_tuning.py mare_nostrum \\
       --include-notes "UCX_MAX_RNDV_RAILS=1" \\
       -o custom_rules.txt

   # Select by latency instead of bandwidth (minimize execution time directly)
   python selector/generate_ompi_tuning.py leonardo --criterion latency

   # Select by latency using median (default metric for both criteria)
   python selector/generate_ompi_tuning.py leonardo --criterion latency --metric median
"""

import os
import sys
import json
import argparse
import re
import tarfile
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np


COLLECTIVE_IDS = {
    "ALLGATHER": 0,
    "ALLREDUCE": 2,
    "ALLTOALL": 3,
    "BCAST": 7,
    "GATHER": 9,
    "REDUCE": 11,
    "REDUCE_SCATTER": 12,
    "SCATTER": 15,
}

OMPI_ALGO_DIR = "config/algorithms/MPI/Open-MPI"
SUMMARY_SCRIPT = "plot/summarize_data.py"


def regex_type(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"invalid regex '{pattern}': {e}")


def comma_separated(val: str) -> list[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def format_bytes(n: int) -> str:
    try:
        x = float(n)
    except (ValueError, TypeError):
        return str(n)

    if x >= 1024**2:
        return f"{x / 1024**2:.0f} MiB"
    if x >= 1024:
        return f"{x / 1024:.0f} KiB"
    return f"{x:.0f} B"


def parse_timestamp_ranges(specs: list[str], all_timestamps: list[str]) -> list[str]:
    to_exclude = []
    for spec in specs:
        if "-" in spec:
            parts = spec.split("-", 1)
            start, end = parts[0], parts[1]
            if not start or not end:
                print(f"  Warning: malformed range spec '{spec}' — skipping", file=sys.stderr)
                continue
            if start > end:
                start, end = end, start
            to_exclude.extend(ts for ts in all_timestamps if start <= ts <= end)
        else:
            to_exclude.append(spec)
    return to_exclude


def find_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent.parent
    if (script_dir / "config").is_dir() and (script_dir / "plot").is_dir():
        return script_dir
    cwd = Path.cwd()
    if (cwd / "config").is_dir() and (cwd / "plot").is_dir():
        return cwd
    print("Error: cannot find repo root (run from repo root or scripts/)", file=sys.stderr)
    sys.exit(1)


def load_metadata(results_dir: Path, system: str) -> pd.DataFrame:
    metadata_path = results_dir / f"{system}_metadata.csv"
    if not metadata_path.is_file():
        print(f"Error: metadata file not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    metadata = pd.read_csv(metadata_path)
    ompi_mask = metadata["mpi_lib"].str.strip().str.upper() == "OMPI"
    ompi_metadata = metadata[ompi_mask].copy()

    if ompi_metadata.empty:
        print(f"Warning: no OMPI tests found in {metadata_path}", file=sys.stderr)
        sys.exit(0)

    return ompi_metadata


def list_unique_notes(metadata: pd.DataFrame) -> None:
    unique_notes = metadata["notes"].dropna().unique()
    unique_notes = sorted(set(n.strip() for n in unique_notes if isinstance(n, str) and n.strip()))
    print("Unique 'notes' values in OMPI metadata:")
    for n in unique_notes:
        print(f"  {n}")


def filter_by_notes(
    metadata: pd.DataFrame,
    include_re: re.Pattern | None,
    exclude_re: re.Pattern | None,
    verbose: bool = False,
) -> pd.DataFrame:
    if include_re is None and exclude_re is None:
        return metadata

    def _notes_str(val) -> str:
        return str(val) if pd.notna(val) else ""

    timestamps_to_keep = []

    for ts, group in metadata.groupby("timestamp"):
        if include_re is not None:
            all_match = all(include_re.search(_notes_str(n)) for n in group["notes"])
            if not all_match:
                if verbose:
                    notes_vals = [_notes_str(n) for n in group["notes"].unique()]
                    print(f"  Excluding {ts} (include_notes: notes={notes_vals} do not all match /{include_re.pattern}/)")
                continue

        if exclude_re is not None:
            any_match = any(exclude_re.search(_notes_str(n)) for n in group["notes"])
            if any_match:
                if verbose:
                    notes_vals = [_notes_str(n) for n in group["notes"].unique()]
                    print(f"  Excluding {ts} (exclude_notes: notes={notes_vals} match /{exclude_re.pattern}/)")
                continue

        timestamps_to_keep.append(ts)

    result = metadata[metadata["timestamp"].isin(timestamps_to_keep)]
    return result


def load_ompi_algorithms(repo_root: Path) -> dict:
    algo_dir = repo_root / OMPI_ALGO_DIR
    if not algo_dir.is_dir():
        print(f"Error: OMPI algorithm directory not found: {algo_dir}", file=sys.stderr)
        sys.exit(1)
    algorithms = {}
    for fname in os.listdir(algo_dir):
        if fname.endswith(".json"):
            coll = fname[:-5].upper()
            with open(algo_dir / fname) as f:
                algorithms[coll] = json.load(f)
    return algorithms


def ensure_summarized(repo_root: Path, results_dir: Path, system: str, timestamp: str) -> None:
    ts_dir = results_dir / system / timestamp
    tar_path = results_dir / system / f"{timestamp}.tar.gz"
    summary_path = ts_dir / "aggregated_results_summary.csv"

    if summary_path.is_file():
        return

    if not ts_dir.is_dir():
        if tar_path.is_file():
            print(f"  Extracting {tar_path}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=results_dir / system)
        else:
            print(f"  Warning: {ts_dir} not found and no {tar_path}", file=sys.stderr)
            return

    if not ts_dir.is_dir():
        return

    print(f"  Running summarization for {timestamp}...")
    result = subprocess.run(
        [sys.executable, SUMMARY_SCRIPT, "--result-dir", str(ts_dir)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    if result.returncode != 0:
        print(f"  Warning: summarization failed for {timestamp}: {result.stderr}", file=sys.stderr)


def generate_tuning_rules(best: pd.DataFrame, add_comments: bool = False) -> str:
    lines = []
    active = best["collective_type"].unique()
    active_sorted = sorted(active, key=lambda c: COLLECTIVE_IDS.get(c, 99))
    lines.append(f"{len(active_sorted)}")
    lines.append("")

    for coll in active_sorted:
        coll_id = COLLECTIVE_IDS[coll]
        lines.append(f"# {coll} ({coll_id})")
        lines.append(f"{coll_id}")

        coll_data = best[best["collective_type"] == coll]
        comm_sizes = sorted(coll_data["comm_size"].unique())
        lines.append(f"{len(comm_sizes)}")

        for cs in comm_sizes:
            cs_data = coll_data[coll_data["comm_size"] == cs].sort_values("array_dim")
            lines.append(f"# comm_size={cs}")
            lines.append(f"{cs}")

            prev_algo_num = None
            rules = []
            first = True
            for _, row in cs_data.iterrows():
                algo_num = int(row["selection"])
                if first:
                    if add_comments:
                        rules.append((0, algo_num, 0, 0, row["algo_name"], 0))
                    else:
                        rules.append((0, algo_num, 0, 0))
                    prev_algo_num = algo_num
                    first = False
                elif algo_num != prev_algo_num:
                    min_bytes = int(row["buffer_size"])
                    if add_comments:
                        rules.append((min_bytes, algo_num, 0, 0, row["algo_name"], min_bytes))
                    else:
                        rules.append((min_bytes, algo_num, 0, 0))
                    prev_algo_num = algo_num

            lines.append(f"{len(rules)}")
            if add_comments:
                for r in rules:
                    min_b, alg, seg, rad, aname, tbytes = r
                    if tbytes == 0:
                        lines.append(f"# 0 bytes (default) — {aname}")
                    else:
                        lines.append(f"# {format_bytes(tbytes)} — {aname}")
                    lines.append(f"{min_b} {alg} {seg} {rad}")
            else:
                for min_b, alg, seg, rad in rules:
                    lines.append(f"{min_b} {alg} {seg} {rad}")
            lines.append("")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Open MPI tuning rules from benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "filtering:\n"
            "  Each run in the metadata CSV has a 'notes' column with free-text tags\n"
            "  (e.g. \"UCX_MAX_RNDV_RAILS=1\", \"rail 4\", \"vsnccl\").  You can include or\n"
            "  exclude entire timestamps based on regex matches against these notes.\n"
            "\n"
            "selection:\n"
            "  --criterion bandwidth  maximizes bandwidth (default)\n"
            "  --criterion latency    minimizes execution time directly\n"
            "  --metric sets the latency statistic to use (mean/median/percentile_90, default median)\n"
            "\n"
            "Examples:\n"
            "  python selector/generate_ompi_tuning.py leonardo --include-notes \"rail 4\"\n"
            "  python selector/generate_ompi_tuning.py leonardo --exclude-notes \"testing|fork\"\n"
            "  python selector/generate_ompi_tuning.py leonardo --list-notes\n"
            "  python selector/generate_ompi_tuning.py leonardo --criterion latency\n"
            "  python selector/generate_ompi_tuning.py mare_nostrum \\\n"
            "      --include-notes \"UCX_MAX_RNDV_RAILS=1\" -o custom_rules.txt"
        ),
    )
    parser.add_argument("system", help="System name (e.g. mare_nostrum, leonardo)")
    parser.add_argument("--results-dir", default="results",
                        help="Path to results directory (default: results/)")
    parser.add_argument("--output", "-o",
                        help="Output file path (default: results/<system>/ompi_tuning_rules_<system>.txt)")

    notes_group = parser.add_argument_group("filtering")
    notes_group.add_argument("--include-notes", type=regex_type, metavar="REGEX",
                             help="Only include timestamps where EVERY row's notes match the given regex")
    notes_group.add_argument("--exclude-notes", type=regex_type, metavar="REGEX",
                             help="Exclude timestamps where ANY row's notes match the given regex")
    notes_group.add_argument("--include-timestamps", type=comma_separated, default=[], metavar="TS1,TS2-TS4,...",
                             help="Only keep these timestamps (name or range, e.g. TS1,TS2-TS4)")
    notes_group.add_argument("--exclude-timestamps", "-x", type=comma_separated, default=[], metavar="TS1,TS2-TS4,...",
                             help="Exclude timestamps by name or range (e.g. -x 2025_04_04___18_21_29,2025_04_05___10_00_00-2025_04_06___20_44_58)")
    notes_group.add_argument("--list-notes", action="store_true",
                             help="Print all unique 'notes' values found in the OMPI metadata and exit")

    selection_group = parser.add_argument_group("selection")
    selection_group.add_argument("--criterion", choices=["bandwidth", "latency"],
                                 default="bandwidth",
                                 help="Selection criterion: 'bandwidth' maximizes bandwidth, "
                                      "'latency' minimizes execution time (default: bandwidth)")
    selection_group.add_argument("--metric", choices=["mean", "median", "percentile_90"],
                                 default="median",
                                 help="Latency metric used for algorithm selection (default: median)")

    verb_group = parser.add_argument_group("verbosity")
    verb_group.add_argument("--quiet", "-q", action="store_true",
                            help="Suppress informational output (warnings/errors only)")
    verb_group.add_argument("--verbose", "-v", action="store_true",
                            help="Show detailed filtering decisions")
    verb_group.add_argument("--annotate", action="store_true",
                            help="Add human-readable comments (message size + algorithm name) to tuning rules")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    repo_root = find_repo_root()
    results_dir = repo_root / args.results_dir
    system = args.system

    quiet = args.quiet
    verbose = args.verbose

    ompi_metadata = load_metadata(results_dir, system)

    if args.list_notes:
        list_unique_notes(ompi_metadata)
        return

    ompi_metadata = filter_by_notes(ompi_metadata, args.include_notes, args.exclude_notes, verbose)

    if args.include_timestamps:
        all_ts = sorted(ompi_metadata["timestamp"].unique())
        to_keep = parse_timestamp_ranges(args.include_timestamps, all_ts)
        before = ompi_metadata["timestamp"].nunique()
        ompi_metadata = ompi_metadata[ompi_metadata["timestamp"].isin(to_keep)]
        after = ompi_metadata["timestamp"].nunique()
        if verbose and before != after:
            print(f"  Kept {after} of {before} timestamp(s) via --include-timestamps")

    if args.exclude_timestamps:
        all_ts = sorted(ompi_metadata["timestamp"].unique())
        to_exclude = parse_timestamp_ranges(args.exclude_timestamps, all_ts)
        before = ompi_metadata["timestamp"].nunique()
        ompi_metadata = ompi_metadata[~ompi_metadata["timestamp"].isin(to_exclude)]
        after = ompi_metadata["timestamp"].nunique()
        if verbose and before != after:
            print(f"  Excluded {before - after} timestamp(s) via --exclude-timestamps")

    timestamps = sorted(ompi_metadata["timestamp"].unique())
    if not timestamps:
        print("Error: no timestamps remaining after filtering", file=sys.stderr)
        sys.exit(1)

    if not quiet:
        print(f"System: {system}")
        print(f"Timestamps: {len(timestamps)}")
        print(f"Collectives: {sorted(ompi_metadata['collective_type'].unique())}")
        print(f"Node counts: {sorted(ompi_metadata['nnodes'].unique())}")

    ompi_algos = load_ompi_algorithms(repo_root)

    for ts in timestamps:
        ensure_summarized(repo_root, results_dir, system, ts)

    all_results = []
    for ts in timestamps:
        summary_path = results_dir / system / ts / "aggregated_results_summary.csv"
        if summary_path.is_file():
            try:
                df = pd.read_csv(summary_path)
            except pd.errors.EmptyDataError:
                if not quiet:
                    print(f"  Warning: empty aggregated results for {ts}", file=sys.stderr)
                continue
            if df.empty:
                continue
            all_results.append(df)

    if not all_results:
        print("Error: no aggregated results found", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    if not quiet:
        print(f"Aggregated measurements loaded: {len(combined)} total")

    filtered_parts = []
    for coll in combined["collective_type"].unique():
        coll_upper = coll.upper()
        if coll_upper not in ompi_algos:
            if not quiet:
                print(f"  Skipping {coll}: no OMPI algorithm definitions")
            continue

        ompi_algo_keys = set(ompi_algos[coll_upper].keys()) - {"default_ompi"}
        coll_data = combined[combined["collective_type"] == coll].copy()
        before = len(coll_data)
        coll_data = coll_data[coll_data["algo_name"].isin(ompi_algo_keys)]
        if coll_data.empty:
            if not quiet:
                print(f"  {coll}: {before} measurements, 0 match OMPI algorithms (all filtered out)")
            continue

        coll_data["selection"] = coll_data["algo_name"].apply(
            lambda x: ompi_algos[coll_upper][x].get("selection", 0)
        )
        if not quiet:
            print(f"  {coll}: {before} -> {len(coll_data)} OMPI measurements")
        filtered_parts.append(coll_data)

    if not filtered_parts:
        print("Error: no OMPI algorithm data remaining after filtering", file=sys.stderr)
        sys.exit(1)

    final = pd.concat(filtered_parts, ignore_index=True)
    final["comm_size"] = final["nnodes"] * final["tasks_per_node"]

    if args.criterion == "bandwidth":
        final["bandwidth"] = ((final["buffer_size"].astype(float) * 8.0) / 1e9) / (
            final[args.metric].astype(float) / 1e9
        )
        best = final.loc[
            final.groupby(["collective_type", "comm_size", "array_dim"])["bandwidth"].idxmax()
        ].copy()
    else:
        best = final.loc[
            final.groupby(["collective_type", "comm_size", "array_dim"])[args.metric].idxmin()
        ].copy()

    if not quiet:
        print(f"\nBest algorithm selection summary (criterion: {args.criterion}, metric: {args.metric}):")
        for coll in sorted(best["collective_type"].unique(), key=lambda c: COLLECTIVE_IDS.get(c, 99)):
            cd = best[best["collective_type"] == coll]
            n_comms = cd["comm_size"].nunique()
            n_dims = cd["array_dim"].nunique()
            n_algos = cd["algo_name"].nunique()
            print(f"  {coll}: {n_comms} comm_sizes, {n_dims} array_dims, {n_algos} unique algorithms selected")

    tuning = generate_tuning_rules(best, add_comments=args.annotate)

    if args.output:
        output_path = repo_root / args.output
    else:
        output_path = results_dir / system / f"ompi_tuning_rules_{system}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tuning)

    if not quiet:
        print(f"\nTuning file written to: {output_path}")
        print(f"  Collectives: {best['collective_type'].nunique()}")
        print(f"  Total rules: {len(best)}")


if __name__ == "__main__":
    main()
