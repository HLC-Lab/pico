# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from ..utils import ensure_dir, human_readable_size
from .comparison_heatmap import BIG_FONT_SIZE, SMALL_FONT_SIZE

METRICS = ("mean", "median", "percentile_90")
SUPPORTED_COLLECTIVES = {"ALLGATHER", "REDUCE_SCATTER"}

DEFAULT_RUNS_BY_SYSTEM: dict[str, list[tuple[str, int]]] = {
    "leonardo": [
        ("2025_04_05___23_20_55", 256),
        ("2025_04_06___13_24_12", 128),
        ("2025_04_06___13_24_31", 64),
        ("2025_04_06___13_24_51", 32),
        ("2025_04_06___13_25_12", 16),
        ("2025_04_06___13_25_25", 8),
        ("2025_04_06___13_25_39", 4),
    ],
    "mare_nostrum": [
        ("2025_04_14___03_44_45", 64),
        ("2025_04_12___14_04_47", 32),
        ("2025_04_10___19_19_35", 16),
        ("2025_04_10___19_18_47", 8),
        ("2025_04_10___17_40_37", 4),
    ],
    "lumi": [
        ("2025_04_07___18_42_37", 8),
        ("2025_04_10___16_25_22", 16),
        ("2025_04_10___16_24_38", 32),
        ("2025_04_09___00_15_52", 64),
        ("2025_04_10___14_26_48", 128),
        ("2025_04_10___15_46_57", 256),
        ("2025_04_10___18_47_49", 512),
        ("2025_04_09___16_00_41", 1024),
    ],
}

BASELINE_DEFAULTS = {
    "ALLGATHER": "recursive_doubling_ompi",
    "REDUCE_SCATTER": "recursive_halving_ompi",
}

BASELINE_OVERRIDES = {
    "lumi": {
        "ALLGATHER": "recursive_doubling_mpich",
        "REDUCE_SCATTER": "recursive_halving_mpich",
    }
}


@dataclass(slots=True)
class BestBineHeatmapConfig:
    system: str
    collective: str
    metric: str = "mean"
    runs: Iterable[str] | None = None
    output: str | Path | None = None


def _default_runs(system: str) -> list[tuple[str, int]]:
    return DEFAULT_RUNS_BY_SYSTEM.get(system.lower(), DEFAULT_RUNS_BY_SYSTEM["leonardo"])


def _parse_runs(arg_runs: Iterable[str] | None, system: str) -> list[tuple[str, int]]:
    if not arg_runs:
        return _default_runs(system)

    out: list[tuple[str, int]] = []
    for item in arg_runs:
        if ":" not in item:
            raise ValueError(f"Invalid run entry '{item}'. Expected TIMESTAMP:NODES.")
        timestamp, nodes_s = item.split(":", 1)
        if not timestamp:
            raise ValueError(f"Invalid run entry '{item}'. Missing timestamp.")
        try:
            nodes = int(nodes_s)
        except ValueError as exc:
            raise ValueError(f"Invalid run entry '{item}'. Nodes must be an integer.") from exc
        out.append((timestamp, nodes))
    return out


def _resolve_baseline(system: str, collective: str) -> str:
    override = BASELINE_OVERRIDES.get(system.lower(), {}).get(collective)
    if override:
        return override
    return BASELINE_DEFAULTS[collective]


def _classify_algo(name: str, collective: str) -> str | None:
    lower = name.lower()
    if "bine_permute_remap" in lower:
        return "permute"
    if "bine_send_remap" in lower:
        return "send"
    if "bine_block_by_block" in lower:
        return "block"
    if collective == "ALLGATHER" and "bine_2_blocks" in lower:
        return "two"
    return None


def generate_best_bine_heatmap(cfg: BestBineHeatmapConfig) -> Path:
    collective = cfg.collective.upper()
    if collective not in SUPPORTED_COLLECTIVES:
        raise ValueError(
            f"Unsupported collective '{cfg.collective}'. "
            f"Expected one of {sorted(SUPPORTED_COLLECTIVES)}."
        )
    metric = cfg.metric.lower()
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric '{cfg.metric}'. Expected one of {METRICS}.")

    runs = _parse_runs(cfg.runs, cfg.system)
    baseline = _resolve_baseline(cfg.system, collective)

    bine_patterns = ["bine_block_by_block", "bine_permute_remap", "bine_send_remap"]
    if collective == "ALLGATHER":
        bine_patterns.append("bine_2_blocks")
    patterns = tuple(bine_patterns + [baseline])

    frames: list[pd.DataFrame] = []
    for timestamp, nodes in runs:
        summary_path = Path("results") / cfg.system / timestamp / "aggregated_results_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file {summary_path} not found.")

        df = pd.read_csv(summary_path)
        subset = df[df["collective_type"].str.lower() == collective.lower()].copy()
        subset["Nodes"] = str(nodes)
        mask = subset["algo_name"].str.contains("|".join(patterns), case=False, na=False)
        subset = subset[mask]
        subset = subset[~subset["algo_name"].str.contains("dtype", case=False, na=False)]
        if subset.empty:
            continue

        subset[f"bandwidth_{metric}"] = (
            ((subset["buffer_size"] * 8.0) / 1e9) / (subset[metric].astype(float) / 1e9)
        )
        frames.append(subset[["buffer_size", "Nodes", "algo_name", f"bandwidth_{metric}"]])

    if not frames:
        raise RuntimeError("No matching data found for the requested runs/collective.")

    bandwidth_df = pd.concat(frames, ignore_index=True)
    ordered_nodes = sorted(bandwidth_df["Nodes"].unique(), key=lambda n: int(str(n)))

    records: list[dict[str, object]] = []
    ratio_map: dict[tuple[int, str], float] = {}

    for (buffer_size, node), group in bandwidth_df.groupby(["buffer_size", "Nodes"]):
        bine_group = group[group["algo_name"].str.contains("bine", case=False, na=False)]
        target_rows = group[group["algo_name"].str.contains(baseline, case=False, na=False)]
        if bine_group.empty or target_rows.empty:
            continue

        best_row = bine_group.loc[bine_group[f"bandwidth_{metric}"].idxmax()]
        category = _classify_algo(str(best_row["algo_name"]), collective)
        if category is None:
            continue

        target_value = target_rows[f"bandwidth_{metric}"].max()
        if not np.isfinite(target_value) or target_value <= 0:
            continue

        ratio_map[(int(buffer_size), str(node))] = best_row[f"bandwidth_{metric}"] / target_value
        records.append(
            {
                "buffer_size": int(buffer_size),
                "Nodes": str(node),
                "category": category,
            }
        )

    if not records:
        raise RuntimeError("Unable to determine a winning Bine algorithm for any cell.")

    letter_map = {"permute": "P", "send": "S", "block": "B"}
    code_map = {"permute": 0, "send": 1, "block": 2}
    if collective == "ALLGATHER":
        letter_map["two"] = "T"
        code_map["two"] = 3

    matrix_df = pd.DataFrame.from_records(records)
    matrix_df["code"] = matrix_df["category"].map(code_map)
    category_map = {
        (row.buffer_size, row.Nodes): row.category for row in matrix_df.itertuples(index=False)
    }

    heatmap_data = matrix_df.pivot(index="buffer_size", columns="Nodes", values="code")
    heatmap_data = heatmap_data.reindex(index=sorted(heatmap_data.index), columns=ordered_nodes)

    tab10 = plt.get_cmap("tab10")
    allowed_indices = [i for i in range(10) if i != 3]
    colors = [tab10(i) for i in allowed_indices[: len(code_map)]]
    cmap = ListedColormap(colors)
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        vmin=-0.5,
        vmax=(len(code_map) - 1) + 0.5,
        cbar=False,
        annot=False,
        ax=ax,
    )

    for i, buffer_size in enumerate(heatmap_data.index):
        for j, node in enumerate(heatmap_data.columns):
            code = heatmap_data.iloc[i, j]
            if pd.isna(code):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=SMALL_FONT_SIZE,
                )
                continue

            category = category_map.get((int(buffer_size), str(node)))
            if category is None:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "?",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=BIG_FONT_SIZE,
                    weight="bold",
                )
                continue

            ratio = ratio_map.get((int(buffer_size), str(node)))
            ax.text(
                j + 0.5,
                i + 0.38,
                letter_map[category],
                ha="center",
                va="center",
                color="white",
                fontsize=BIG_FONT_SIZE,
                weight="bold",
            )
            if ratio is not None and np.isfinite(ratio):
                ax.text(
                    j + 0.5,
                    i + 0.74,
                    f"{ratio:.2f}×",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=SMALL_FONT_SIZE - 1,
                )

    buffer_labels = [human_readable_size(int(size)) for size in heatmap_data.index]
    ax.set_yticks(np.arange(len(buffer_labels)) + 0.5)
    ax.set_yticklabels(buffer_labels, fontsize=SMALL_FONT_SIZE)
    ax.set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
    ax.set_xticklabels(heatmap_data.columns, fontsize=SMALL_FONT_SIZE)
    ax.set_xlabel("# Nodes", fontsize=BIG_FONT_SIZE)
    ax.set_ylabel("Vector Size", fontsize=BIG_FONT_SIZE)

    fig.tight_layout()

    if cfg.output is not None:
        outfile = Path(cfg.output)
        ensure_dir(outfile.parent)
    else:
        outdir = ensure_dir(Path("plot") / cfg.system / "heatmaps" / collective.lower())
        outfile = outdir / f"{cfg.system}_{collective.lower()}_best_bine_variant.pdf"

    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile
