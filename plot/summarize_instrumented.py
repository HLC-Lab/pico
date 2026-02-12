#!/usr/bin/env python3
"""
Build a per-file summary DataFrame from benchmark CSVs.

Expected layout (typical):
  <root>/
    0/   (cpu)
      <vec_size>_<algo>_[<buffer_size>_]over_<datatype>_instrument.csv
    1/   (gpu)
      (same)

Output columns:
  array_dim, buffer_size, algo_name, datatype, test_type, <csv_col>_mean...

Means are computed after discarding the first 20% of iterations (rows).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import pandas as pd


# ----------------------------
# Filename parsing
# ----------------------------

@dataclass(frozen=True)
class FileMeta:
    array_dim: int
    buffer_size: Optional[int]
    algo_name: str
    datatype: str


def parse_benchmark_filename(filename: str) -> FileMeta:
    """
    Robustly parse filenames such as:
      134217728_recursive_distance_doubling_hierarchical_v4_over_int32_instrument.csv
      1048576_my_algo_4096_over_float32_instrument.csv
      1048576_my_algo_4096_int32_instrument.csv   (also supported)

    Rules:
      - Must end with "_instrument.csv"
      - array_dim is the leading integer before first underscore
      - datatype is the last token before "_instrument.csv"
      - an optional literal token "over" may appear just before datatype
      - buffer_size is optional: if the last token of algo part is purely digits, treat it as buffer_size
    """
    if not filename.endswith("_instrument.csv"):
        raise ValueError(f"Not an instrument file: {filename}")

    base = filename[: -len("_instrument.csv")]  # strip suffix
    if "_" not in base:
        raise ValueError(f"Unexpected filename format (no underscores): {filename}")

    array_str, tail = base.split("_", 1)
    if not array_str.isdigit():
        raise ValueError(f"Filename does not start with array_dim integer: {filename}")
    array_dim = int(array_str)

    parts = tail.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename tail: {filename}")

    datatype = parts[-1]
    # drop optional "over"
    if len(parts) >= 2 and parts[-2].lower() == "over":
        algo_parts = parts[:-2]
    else:
        algo_parts = parts[:-1]

    if not algo_parts:
        raise ValueError(f"Could not parse algo name from: {filename}")

    buffer_size: Optional[int] = None
    if algo_parts and algo_parts[-1].isdigit():
        buffer_size = int(algo_parts[-1])
        algo_parts = algo_parts[:-1]

    algo_name = "_".join(algo_parts).strip()
    if not algo_name:
        raise ValueError(f"Algo name resolved empty for: {filename}")

    return FileMeta(
        array_dim=array_dim,
        buffer_size=buffer_size,
        algo_name=algo_name,
        datatype=datatype,
    )


# ----------------------------
# CPU/GPU inference from path
# ----------------------------

def infer_test_type(root_dir: str, filepath: str) -> str:
    """
    Infer cpu/gpu from the relative path.
      - If any path component == "0" -> cpu
      - If any path component == "1" -> gpu
      - If any component contains "cpu"/"gpu" (case-insensitive) -> cpu/gpu
      - Else "unknown"
    """
    rel = os.path.relpath(filepath, root_dir)
    comps = [c.lower() for c in rel.split(os.sep)]

    if "0" in comps:
        return "gpu"
    if "1" in comps:
        return "cpu"
    if any("cpu" in c for c in comps):
        return "cpu"
    if any("gpu" in c for c in comps):
        return "gpu"
    return "unknown"


# ----------------------------
# Efficient mean after dropping first 20%
# ----------------------------

def count_data_rows(filepath: str, bufsize: int = 1024 * 1024) -> int:
    """
    Count number of data rows (excluding header) quickly by counting newlines.
    Assumes a single header line.
    """
    n_newlines = 0
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            n_newlines += chunk.count(b"\n")

    # subtract header
    return max(n_newlines - 1, 0)


def summarize_csv_means(
    filepath: str,
    discard_frac: float = 0.2,
    chunksize: int = 200_000,
) -> Dict[str, float]:
    """
    Compute per-column means after discarding the first discard_frac of rows.
    Uses chunked reading; does not load the full CSV into memory.

    Returns: dict of { "<col>_mean": mean_value } for numeric columns.
    """
    total_rows = count_data_rows(filepath)
    discard = int(math.floor(discard_frac * total_rows))

    sums = None
    counts = None
    processed_total = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        orig_len = len(chunk)

        # Trim rows belonging to the initial discard section
        if processed_total < discard:
            drop = min(orig_len, discard - processed_total)
            chunk = chunk.iloc[drop:]

        processed_total += orig_len

        if chunk.empty:
            continue

        # Keep only numeric values; coerce non-numeric to NaN then ignore in sums/counts
        num = chunk.apply(pd.to_numeric, errors="coerce")

        chunk_sums = num.sum(axis=0, skipna=True)
        chunk_counts = num.count(axis=0)

        if sums is None:
            sums = chunk_sums
            counts = chunk_counts
        else:
            sums = sums.add(chunk_sums, fill_value=0)
            counts = counts.add(chunk_counts, fill_value=0)

    if sums is None or counts is None:
        return {}

    means = (sums / counts).to_dict()
    return {f"{k}_mean": float(v) for k, v in means.items() if pd.notna(v)}


# ----------------------------
# Main collection logic
# ----------------------------

def iter_instrument_csvs(root_dir: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith("_instrument.csv"):
                yield os.path.join(dirpath, fn)


def build_summary_dataframe(
    root_dir: str,
    discard_frac: float = 0.2,
    chunksize: int = 200_000,
    strict_parse: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for fp in iter_instrument_csvs(root_dir):
        fn = os.path.basename(fp)

        try:
            meta = parse_benchmark_filename(fn)
        except Exception as e:
            if strict_parse:
                raise
            # Skip unparseable files in non-strict mode
            print(f"[WARN] Skipping {fn}: {e}")
            continue

        test_type = infer_test_type(root_dir, fp)

        means = summarize_csv_means(fp, discard_frac=discard_frac, chunksize=chunksize)

        row: Dict[str, object] = {
            "array_dim": meta.array_dim,
            "buffer_size": meta.buffer_size,
            "algo_name": meta.algo_name,
            "datatype": meta.datatype,
            "test_type": test_type,
            "source_file": os.path.relpath(fp, root_dir),
        }
        row.update(means)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Nice-to-have: stable ordering if present
    sort_cols = [c for c in ["algo_name", "datatype", "buffer_size", "array_dim", "test_type"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", help="Path to the folder that contains the benchmark data (e.g., with 0/ and 1/).")
    ap.add_argument("-o", "--output", default="", help="Optional path to write output CSV (e.g., summary.csv).")
    ap.add_argument("--discard-frac", type=float, default=0.2, help="Fraction of initial rows to discard (default: 0.2).")
    ap.add_argument("--chunksize", type=int, default=200_000, help="CSV read chunksize (default: 200000).")
    ap.add_argument("--strict-parse", action="store_true", help="Fail if any file name cannot be parsed.")
    args = ap.parse_args()

    df = build_summary_dataframe(
        args.root_dir,
        discard_frac=args.discard_frac,
        chunksize=args.chunksize,
        strict_parse=args.strict_parse,
    )

    print(df.head(20).to_string(index=False))
    print(f"\nRows: {len(df)}  Columns: {len(df.columns)}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
