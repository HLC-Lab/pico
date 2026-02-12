# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from ..utils import (
    PlotMetadata,
    apply_adaptive_legend,
    build_tab10_palette,
    ensure_dir,
    format_time_units_ns,
    format_bytes,
    sort_key,
    style_axes,
)

sns.set_palette("tab10")
sns.set_style("whitegrid")
sns.set_context("paper")


def _resolve_output_dir(system: str, output_dir: str | Path | None) -> Path:
    if output_dir:
        return ensure_dir(output_dir)
    return ensure_dir(Path("plot") / system)


def _collapse_preaggregated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicated rows (same algorithm/buffer size) by recomputing summary
    statistics using the provided aggregates.  This prevents seaborn from
    collapsing already-aggregated data with a second estimator pass.
    """
    key_cols = ["algo_name", "buffer_size"]
    if not set(key_cols).issubset(df.columns):
        return df

    duplicates = df.duplicated(key_cols, keep=False)
    if not duplicates.any():
        return df

    collapsed_rows: list[pd.Series] = []
    for _, group in df.groupby(key_cols, sort=False):
        collapsed_rows.append(_merge_group(group))

    collapsed = pd.DataFrame(collapsed_rows)

    # Ensure all original columns are present and in the same order
    for col in df.columns:
        if col not in collapsed.columns:
            collapsed[col] = np.nan

    return collapsed[df.columns]


def _merge_group(group: pd.DataFrame) -> pd.Series:
    base = group.iloc[0].copy()

    if "n_iter" in group.columns:
        weights = group["n_iter"].astype(float).fillna(0.0)
    else:
        weights = pd.Series(np.ones(len(group)), index=group.index, dtype=float)

    total_weight = float(weights.sum())
    if total_weight <= 0:
        weights = pd.Series(np.ones(len(group)), index=group.index, dtype=float)
        total_weight = float(weights.sum())

    weight_array = weights.to_numpy(dtype=float)

    means = group["mean"].astype(float).to_numpy()
    stds = (
        group["std"].astype(float).fillna(0.0).to_numpy()
        if "std" in group.columns
        else np.zeros_like(means)
    )

    sum_x = float(np.dot(weight_array, means))
    sum_x2 = float(np.dot(weight_array, stds**2 + means**2))

    combined_mean = sum_x / total_weight
    combined_var = max(sum_x2 / total_weight - combined_mean**2, 0.0)
    combined_std = float(np.sqrt(combined_var))

    base["mean"] = combined_mean
    if "std" in base.index:
        base["std"] = combined_std
    if "n_iter" in base.index:
        base["n_iter"] = int(round(total_weight))

    def _weighted_average(column: str) -> float | None:
        if column not in group.columns:
            return None
        values = group[column].astype(float)
        if np.all(np.isnan(values)):
            return np.nan
        return float(np.average(values, weights=weight_array))

    for column in ("median", "percentile_10", "percentile_25", "percentile_75", "percentile_90", "iqr"):
        if column in base.index:
            value = _weighted_average(column)
            if value is not None:
                base[column] = value

    if "min" in base.index:
        base["min"] = float(group["min"].min())
    if "max" in base.index:
        base["max"] = float(group["max"].max())
    if "num_outliers" in base.index:
        base["num_outliers"] = int(group["num_outliers"].fillna(0).sum())

    if "standard_error" in base.index:
        if total_weight > 1:
            se = float(combined_std / np.sqrt(total_weight))
            base["standard_error"] = se
            if "ci_lower" in base.index:
                base["ci_lower"] = combined_mean - 1.96 * se
            if "ci_upper" in base.index:
                base["ci_upper"] = combined_mean + 1.96 * se
        else:
            base["standard_error"] = np.nan
            if "ci_lower" in base.index:
                base["ci_lower"] = np.nan
            if "ci_upper" in base.index:
                base["ci_upper"] = np.nan

    return base


def generate_line_plot(
    data: pd.DataFrame,
    *,
    metadata: PlotMetadata,
    collective: str,
    datatype: str,
    error_col: str | None,
    error_mode: str = "band",
    output_dir: str | Path | None = None,
) -> Path:
    """
    Render the latency line plot (log-log) for a single ``collective`` and
    ``datatype`` pair.  ``data`` must already be filtered to contain only rows
    for the selected pair.
    """
    if data.empty:
        raise ValueError("No data available for generate_line_plot.")

    sorted_algos = sorted(data["algo_name"].unique().tolist(), key=sort_key)
    palette = build_tab10_palette(sorted_algos)
    df = _collapse_preaggregated(data).sort_values("buffer_size").copy()

    err_series = None
    if error_col == "std" and {"std", "n_iter"}.issubset(df.columns):
        err_series = df["std"]
    elif error_col == "ci" and {"ci_lower", "ci_upper"}.issubset(df.columns):
        err_series = (df["ci_upper"] - df["ci_lower"]) / 2.0
    elif error_col == "se" and "standard_error" in df.columns:
        err_series = df["standard_error"]
    elif error_col == "iqr" and {"iqr"}.issubset(df.columns):
        err_series = df["iqr"] / 2.0
    elif error_col == "percentiles" and {"percentile_10", "percentile_90"}.issubset(df.columns):
        err_series = (df["percentile_90"] - df["percentile_10"]) / 2.0
    else:
        print(f"Warning: could not find valid error columns for '{error_col}'; skipping error bars.")
        err_series = None

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df,
        x="buffer_size",
        y="mean",
        hue="algo_name",
        style="algo_name",
        hue_order=sorted_algos,
        palette=palette,
        markers=True,
        markersize=11,
        linewidth=3,
        estimator=None,          # data already aggregated; plot it as-is
        errorbar=None            # ← turn off CI bands
    )

    if err_series is not None:
        df = df.assign(__err=err_series.astype(float))
        if error_mode == "band":
            for algo, group in df.groupby("algo_name"):
                g = group.sort_values("buffer_size")
                x = g["buffer_size"].values
                y = g["mean"].values
                e = g["__err"].values
                ax.fill_between(x, y - e, y + e, alpha=0.15, zorder=2, color=palette.get(algo))
        else:
            for algo, group in df.groupby("algo_name"):
                g = group.sort_values("buffer_size")
                ax.errorbar(
                    g["buffer_size"].values,
                    g["mean"].values,
                    yerr=g["__err"].values,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1,
                    capsize=3,
                    zorder=4,
                )

    plt.xscale("log")
    plt.yscale("log")

    xticks = sorted(pd.unique(df["buffer_size"]).astype(float))
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_bytes(x) for x in xticks])
    ax.yaxis.set_major_formatter(format_time_units_ns)
    # ax.tick_params(axis="both", labelsize=18)

    if metadata.total_nodes == metadata.mpi_tasks:
        title = f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, {metadata.nnodes} nodes"
    else:
        title = (
            f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, "
            f"{metadata.nnodes} nodes ({metadata.mpi_tasks} tasks)"
        )

    plt.title(title, fontsize=18)
    ax.set_xlabel("Message Size", fontsize=15)
    ax.set_ylabel("Execution Time", fontsize=15)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        " ".join(w.capitalize() for w in label.replace("_", " ").split()) #if w not in {"over", "ompi", "distance"})
        for label in labels
    ]
    if "Binomial Doubling (open Mpi 4.1.6)" in new_labels:
        idx = new_labels.index("Binomial Doubling (open Mpi 4.1.6)")
        new_labels[idx] = "Binomial Doubling (Open MPI 4.1.6)"
    apply_adaptive_legend(ax, handles=handles, labels=new_labels, loc="upper left", frameon=True)
    style_axes(ax)
    plt.tight_layout()

    target_dir = _resolve_output_dir(metadata.system, output_dir)
    suffix = f"{error_col}_lineplot" if error_col else "lineplot"
    name = f"{collective.lower()}_{metadata.nnodes}_{datatype}_{metadata.timestamp}_{suffix}.pdf"
    name_pdf = name.replace(".png", ".pdf")
    full_path = target_dir / name
    full_path_pdf = target_dir / name_pdf
    plt.savefig(full_path, dpi=300)
    plt.savefig(full_path_pdf, dpi=300)
    plt.close()
    return full_path
