#!/usr/bin/env python3
"""
Grouped bar chart of normalized "execution time" with error bars.

Baseline (first bar): Rabenseifner 2 rails, normalized to 1.0
Second bar: Rabenseifner 4 rails, normalized to baseline, with correct direction.

Key point:
- If mean is TIME (lower is better): normalized = mean_4 / mean_2
- If mean is THROUGHPUT (higher is better) but you want normalized TIME: normalized = mean_2 / mean_4

Usage:
  python plot_norm.py results.csv -o normalized.png --metric throughput
"""

import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from .utils import apply_adaptive_legend, style_axes
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils import apply_adaptive_legend, style_axes

BASELINE_ALGO = "rabenseifner_(2_rails)_ompi"
COMPARE_ALGO  = "rabenseifner_(4_rails)_ompi"


def bytes_to_human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        v /= 1024.0
        if v < 1024.0:
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))} {u}"
            return f"{v:.1f} {u}"
    return f"{v:.1f} PiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV with columns: buffer_size, algo_name, mean, standard_error (or your choice)")
    ap.add_argument("-o", "--out", default="", help="Output image path (e.g., plot.png). If omitted, shows window.")
    ap.add_argument("--error-col", default="standard_error",
                    help="Column to use for error bars (e.g., standard_error, std). Default: standard_error")
    ap.add_argument("--metric", choices=["time", "throughput"], default="throughput",
                    help="How to interpret 'mean'. If throughput, we plot normalized *time* via baseline/compare.")
    ap.add_argument(
        "--propagate-baseline",
        action="store_true",
        help=(
            "Propagate baseline uncertainty into the normalized error for the 4-rails series (delta method).\n"
            "Recommended if both series' uncertainty matters."
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {"buffer_size", "algo_name", "mean", args.error_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    base = df[df["algo_name"] == BASELINE_ALGO].copy().set_index("buffer_size").sort_index()
    comp = df[df["algo_name"] == COMPARE_ALGO].copy().set_index("buffer_size").sort_index()

    if base.empty or comp.empty:
        raise ValueError(
            "Could not find both algorithms in the CSV.\n"
            f"Expected algo_name values:\n  - {BASELINE_ALGO}\n  - {COMPARE_ALGO}"
        )

    common = base.index.intersection(comp.index)
    if len(common) == 0:
        raise ValueError("No overlapping buffer_size values between the two algorithms.")

    base = base.loc[common]
    comp = comp.loc[common]

    m2 = base["mean"].astype(float).to_numpy()
    e2 = base[args.error_col].astype(float).to_numpy()

    m4 = comp["mean"].astype(float).to_numpy()
    e4 = comp[args.error_col].astype(float).to_numpy()

    # Normalized baseline (always 1)
    y2 = np.ones_like(m2, dtype=float)
    y2_err = e2 / m2  # relative error on baseline mean

    if args.metric == "time":
        # normalized time = t4 / t2
        y4 = m4 / m2

        if args.propagate_baseline:
            # se(y) ≈ y*sqrt((se4/m4)^2 + (se2/m2)^2)
            y4_err = y4 * np.sqrt((e4 / m4) ** 2 + (e2 / m2) ** 2)
        else:
            # treat baseline as fixed: se(t4/t2) ≈ se4 / t2
            y4_err = e4 / m2

    else:
        # mean is throughput; want "time-like" normalization (lower is better):
        # normalized time ∝ thr2 / thr4
        y4 = m2 / m4

        if args.propagate_baseline:
            # se(y) ≈ y*sqrt((se2/m2)^2 + (se4/m4)^2)
            y4_err = y4 * np.sqrt((e2 / m2) ** 2 + (e4 / m4) ** 2)
        else:
            # treat baseline as fixed: y = m2/m4; dy/dm4 = -m2/m4^2
            y4_err = (m2 / (m4 ** 2)) * e4

    # X labels
    labels = [bytes_to_human(int(b)) for b in common.to_numpy()]
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.38

    cmap = plt.get_cmap("tab10")
    c0, c1 = cmap(0), cmap(1)

    ax.bar(x - width/2, y2, width, yerr=y2_err, capsize=4,
           label="Rabenseifner (2 Rails)", color=c0, edgecolor="black")
    ax.bar(x + width/2, y4, width, yerr=y4_err, capsize=4,
           label="Rabenseifner (4 Rails)", color=c1, edgecolor="black")

    ax.set_ylabel("Normalized Execution Time", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    apply_adaptive_legend(ax, loc="lower left", frameon=True)

    ymax = float(np.max(np.r_[y2 + y2_err, y4 + y4_err]))
    ax.set_ylim(0.0, max(1.05, math.ceil((ymax + 0.05) * 10) / 10))

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
