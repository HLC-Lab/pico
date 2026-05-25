# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import numpy as np

from .utils import PlotMetadata


class SummaryEmptyError(RuntimeError):
    """Raised when filtering removes every row from a summary."""


def read_summary(path: str) -> pd.DataFrame:
    """
    Load an aggregated summary CSV produced by ``summarize_data.py``.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise SummaryEmptyError(f"Summary file {path} is empty.")
    return df


def extract_metadata(df: pd.DataFrame) -> PlotMetadata:
    """
    Extract the invariant metadata fields from a summary dataframe.
    """
    row = df.iloc[0]
    tasks_per_node = row.get("tasks_per_node")
    return PlotMetadata(
        system=row["system"],
        timestamp=row["timestamp"],
        mpi_lib=row["mpi_lib"],
        nnodes=str(row["nnodes"]),
        tasks_per_node=int(tasks_per_node) if pd.notna(tasks_per_node) else 1,
        gpu_lib=row.get("gpu_lib", "CPU"),
    )


def filter_summary(
    df: pd.DataFrame,
    *,
    collective: str | None = None,
    datatype: str | None = None,
    algorithm: Iterable[str] | None = None,
    filter_by: Iterable[str] | None = None,
    filter_out: Iterable[str] | None = None,
    min_dim: int | None = None,
    max_dim: int | None = None,
) -> pd.DataFrame:
    """
    Apply filtering operations that mirror the legacy CLI flags.
    """
    filtered = df.copy()

    if collective:
        filtered = filtered[filtered["collective_type"] == collective]
    if datatype:
        filtered = filtered[filtered["datatype"] == datatype]
    if algorithm:
        algorithms = list(algorithm)
        filtered = filtered[filtered["algo_name"].isin(algorithms)]
    if filter_by:
        pattern = "|".join(filter_by)
        filtered = filtered[filtered["algo_name"].str.contains(pattern, case=False, na=False)]
    if filter_out:
        pattern = "|".join(filter_out)
        filtered = filtered[~filtered["algo_name"].str.contains(pattern, case=False, na=False)]
    if min_dim is not None:
        filtered = filtered[filtered["buffer_size"] >= int(min_dim)]
    if max_dim is not None:
        filtered = filtered[filtered["buffer_size"] <= int(max_dim)]

    if filtered.empty:
        raise SummaryEmptyError("Filtered data is empty.")
    return filtered


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove metadata columns that do not vary across the dataframe.
    """
    always_drop = ["array_dim"]
    conditional_drop = [
        "nnodes",
        "system",
        "timestamp",
        "test_id",
        "MPI_Op",
        "notes",
        "mpi_lib",
        "mpi_lib_version",
        "gpu_lib",
        "gpu_lib_version",
        "libpico_version",
    ]

    cleaned = df.drop(columns=[col for col in always_drop if col in df.columns], errors="ignore")

    for col in conditional_drop:
        if col not in cleaned.columns:
            continue
        uniques = cleaned[col].dropna().unique()
        if len(uniques) <= 1:
            cleaned = cleaned.drop(columns=col)

    return cleaned


def normalize_dataset(
    data: pd.DataFrame,
    *,
    mpi_lib: str,
    gpu_lib: str,
    base: str | None = None,
    group_key: str = "buffer_size",
    metric_col: str = "mean",
    se_col: str = "standard_error",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    corr: float = 0.0,   # rho in [-1, 1] for pairing; 0 is conservative default
) -> pd.DataFrame:
    """
    Normalize the dataset by dividing by a reference algorithm within each group_key.

    Outputs:
      - normalized_mean (ratio of means)
      - normalized_se   (propagated SE of the ratio using delta method)
      - normalized_ci_lower / normalized_ci_upper (normalized CI if present)
    """
    df = data.copy()

    chosen_base = base
    if chosen_base is None:
        if mpi_lib in {"OMPI", "OMPI_BINE"}:
            chosen_base = "allreduce_nccl_pat" if gpu_lib == "CUDA" else "default_ompi"
        elif mpi_lib in {"MPICH", "CRAY_MPICH"}:
            chosen_base = "default_mpich"
        elif mpi_lib in {"NCCL"}:
            chosen_base = "ring_nccl_simple"

    if chosen_base is None:
        raise ValueError("Could not determine a baseline algorithm; pass base=... explicitly.")

    # Prepare output columns
    norm_mean = pd.Series(index=df.index, dtype=float)
    norm_se = pd.Series(index=df.index, dtype=float)
    norm_ci_lo = pd.Series(index=df.index, dtype=float)
    norm_ci_hi = pd.Series(index=df.index, dtype=float)

    has_se = se_col in df.columns
    has_ci = (ci_lower_col in df.columns) and (ci_upper_col in df.columns)

    for key, group in df.groupby(group_key):
        base_row = group.loc[group["algo_name"] == chosen_base]
        if base_row.empty:
            continue  # no baseline for this group; leave NaNs, filled later

        mu_b = float(base_row[metric_col].iloc[0])
        if mu_b == 0.0 or not np.isfinite(mu_b):
            continue

        # Point estimate ratio
        r = group[metric_col] / mu_b
        norm_mean.loc[group.index] = r

        # Normalize CI bounds directly (CI for the mean -> divide by same baseline mean)
        if has_ci:
            norm_ci_lo.loc[group.index] = group[ci_lower_col] / mu_b
            norm_ci_hi.loc[group.index] = group[ci_upper_col] / mu_b

        # Propagate SE for ratio (preferred for errorbars)
        if has_se:
            se = group[se_col].astype(float)
            mu = group[metric_col].astype(float)

            se_b = float(base_row[se_col].iloc[0]) if se_col in base_row else np.nan
            if not np.isfinite(se_b):
                # If baseline SE missing, fall back to only numerator contribution (still better than std)
                rel_var = (se / mu) ** 2
            else:
                rel_t = se / mu
                rel_b = se_b / mu_b
                rel_var = rel_t**2 + rel_b**2 - 2.0 * corr * rel_t * rel_b

            rel_var = np.maximum(rel_var, 0.0)  # numerical safety
            norm_se.loc[group.index] = np.abs(r) * np.sqrt(rel_var)

        # Convention: baseline is exactly 1 with zero error bar (since it's the reference)
        base_idx = base_row.index
        norm_mean.loc[base_idx] = 1.0

        # Baseline "error" shown as relative uncertainty of baseline mean (scale factor uncertainty)
        if has_se:
            se_b = float(base_row[se_col].iloc[0])
            mu_b = float(base_row[metric_col].iloc[0])
            norm_se.loc[base_idx] = (se_b / mu_b) if (mu_b != 0 and np.isfinite(se_b) and np.isfinite(mu_b)) else 0.0

        if has_ci:
            lb = float(base_row[ci_lower_col].iloc[0])
            ub = float(base_row[ci_upper_col].iloc[0])
            mu_b = float(base_row[metric_col].iloc[0])
            if mu_b != 0 and np.isfinite(lb) and np.isfinite(ub) and np.isfinite(mu_b):
                norm_ci_lo.loc[base_idx] = lb / mu_b
                norm_ci_hi.loc[base_idx] = ub / mu_b
            else:
                norm_ci_lo.loc[base_idx] = 1.0
                norm_ci_hi.loc[base_idx] = 1.0

    df["normalized_mean"] = norm_mean.fillna(1.0)

    if has_se:
        df["normalized_se"] = norm_se.fillna(0.0)
    else:
        df["normalized_se"] = 0.0

    if has_ci:
        df["normalized_ci_lower"] = norm_ci_lo.fillna(df["normalized_mean"])
        df["normalized_ci_upper"] = norm_ci_hi.fillna(df["normalized_mean"])
    else:
        df["normalized_ci_lower"] = df["normalized_mean"]
        df["normalized_ci_upper"] = df["normalized_mean"]

    return df
