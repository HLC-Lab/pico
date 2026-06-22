# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import build_ratio_colormap, ensure_dir, human_readable_size

BIG_FONT_SIZE = 18
SMALL_FONT_SIZE = 15
FMT = ".2f"
METRICS = ["mean", "median", "percentile_90"]


@dataclass(slots=True)
class ComparisonHeatmapConfig:
    system: str
    collective: str
    nnodes: Iterable[str]
    target_algo: str = "ring_ompi"
    tasks_per_node: int = 1
    notes: str | None = None
    exclude: str | None = None
    metric: str = "mean"
    show_names: bool = False
    output_dir: str | Path | None = None

    def sorted_nodes(self) -> list[str]:
        return [str(n) for n in self.nnodes]


def _metadata_filter(metadata: pd.DataFrame, cfg: ComparisonHeatmapConfig, nodes: str) -> pd.DataFrame:
    if "tasks_per_node" in metadata.columns:
        filtered = metadata[
            (metadata["collective_type"].str.lower() == cfg.collective.lower())
            & (metadata["nnodes"].astype(str) == str(nodes))
            & (metadata["tasks_per_node"].astype(int) == cfg.tasks_per_node)
        ]
    else:
        filtered = metadata[
            (metadata["collective_type"].str.lower() == cfg.collective.lower())
            & (metadata["nnodes"].astype(str) == str(nodes))
        ]
    if cfg.notes:
        filtered = filtered[filtered["notes"].str.strip() == cfg.notes.strip()]
    else:
        filtered = filtered[filtered["notes"].isnull()]
    return filtered


def _discover_summaries(cfg: ComparisonHeatmapConfig) -> dict[str, str]:
    metadata_file = f"results/{cfg.system}_metadata.csv"
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} not found.")

    metadata = pd.read_csv(metadata_file)
    summaries: dict[str, str] = {}

    for nodes in cfg.sorted_nodes():
        filtered = _metadata_filter(metadata, cfg, nodes)
        if filtered.empty:
            raise RuntimeError(f"Metadata file {metadata_file} does not contain the requested data.")
        last_entry = filtered.iloc[-1]
        summaries[nodes] = f"results/{cfg.system}/{last_entry['timestamp']}/"
    return summaries


def _load_summaries(cfg: ComparisonHeatmapConfig) -> pd.DataFrame:
    summaries = _discover_summaries(cfg)
    frames: list[pd.DataFrame] = []

    for nodes, summary_dir in summaries.items():
        summary_path = os.path.join(summary_dir, "aggregated_results_summary.csv")
        if not os.path.exists(summary_path):
            subprocess.run(
                [
                    "python3",
                    "./plot/summarize_data.py",
                    "--result-dir",
                    summary_dir,
                ],
                stdout=subprocess.DEVNULL,
                check=False,
            )
        df = pd.read_csv(summary_path)
        df = df[df["collective_type"].str.lower() == cfg.collective.lower()]
        df = df[df["buffer_size"] != 4]
        df["Nodes"] = nodes
        frames.append(df)

    if not frames:
        raise RuntimeError("No data found for the requested configuration.")
    return pd.concat(frames, ignore_index=True)


def _augment_dataframe(df: pd.DataFrame, cfg: ComparisonHeatmapConfig) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[df.groupby(["buffer_size", "Nodes", "algo_name"])[f"bandwidth_{cfg.metric}"].idxmax()]

    new_rows = []
    for (buffer_size, nodes), group in df.groupby(["buffer_size", "Nodes"]):
        best_row = group.loc[group[f"bandwidth_{cfg.metric}"].idxmax()]
        best_algo = best_row["algo_name"]

        target_row = group[group["algo_name"] == cfg.target_algo]
        if target_row.empty:
            continue
        target_value = target_row[f"bandwidth_{cfg.metric}"].values[0]

        if cfg.show_names:
            cell = best_algo
        else:
            tmp = group[group["algo_name"] != best_algo][f"bandwidth_{cfg.metric}"]
            if tmp.empty:
                continue
            second_best = group.loc[tmp.idxmax()]
            if best_algo == cfg.target_algo:
                cell = best_row[f"bandwidth_{cfg.metric}"] / second_best[f"bandwidth_{cfg.metric}"]
            else:
                ratio = target_value / best_row[f"bandwidth_{cfg.metric}"]
                cell = round(ratio, 1)

        new_rows.append(
            {
                "buffer_size": buffer_size,
                "Nodes": nodes,
                "cell": cell,
            }
        )

    return pd.DataFrame(new_rows)


def _algo_to_short(algo_name: str) -> str:
    mapping = {
        "allreduce_nccl_ring": "N-R",
        "allreduce_nccl_tree": "N-T",
        "allreduce_nccl_pat": "N-P",
        "allreduce_nccl_collnet": "N-C",
        "allreduce_nccl_nvls": "N-N",
        "allreduce_nccl_nvlstree": "N-E",
        "allreduce_nccl_collnetdirect": "N-D",
        "allreduce_nccl_collnetdirectchain": "N-H",
    }
    return mapping.get(algo_name, algo_name)


def generate_comparison_heatmap(cfg: ComparisonHeatmapConfig) -> Path:
    df = _load_summaries(cfg)
    df = df[["buffer_size", "Nodes", "algo_name", "mean", "median", "percentile_90"]]

    if cfg.exclude:
        df = df[~df["algo_name"].str.contains(cfg.exclude, case=False)]

    target_algo_rows = df[df["algo_name"] == cfg.target_algo]
    df = df[df["algo_name"] != cfg.target_algo]
    if not target_algo_rows.empty:
        df = pd.concat([df, target_algo_rows], ignore_index=True)

    for metric in METRICS:
        if metric == cfg.metric:
            df[f"bandwidth_{metric}"] = ((df["buffer_size"] * 8.0) / 1e9) / (df[metric].astype(float) / 1e9)
    for metric in METRICS:
        df = df.drop(columns=[metric])

    df = _augment_dataframe(df, cfg)

    numeric_df = df.copy()
    numeric_df["cell"] = pd.to_numeric(df["cell"], errors="coerce")

    heatmap_data = numeric_df.pivot(index="buffer_size", columns="Nodes", values="cell")
    heatmap_data = heatmap_data[cfg.sorted_nodes()]

    text_data = df.pivot(index="buffer_size", columns="Nodes", values="cell")
    text_data = text_data[cfg.sorted_nodes()]

    plt.figure()
    cmap = build_ratio_colormap()
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap=cmap,
        fmt=FMT,
        center=1,
        cbar=True,
        annot_kws={"size": BIG_FONT_SIZE, "weight": "bold"},
        cbar_kws={"orientation": "horizontal", "location": "top", "aspect": 40},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE)

    if cfg.show_names:
        for i in range(text_data.shape[0]):
            for j in range(text_data.shape[1]):
                val = text_data.iloc[i, j]
                if isinstance(val, str):
                    short = _algo_to_short(val)
                    plt.text(j + 0.5, i + 0.5, short, ha="center", va="center", color="black", weight="bold", fontsize=BIG_FONT_SIZE)
                elif pd.isna(val):
                    plt.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center", color="black", weight="bold", fontsize=BIG_FONT_SIZE)

    buffer_sizes = text_data.index.astype(int).tolist()
    buffer_sizes.sort()
    buffer_labels = [human_readable_size(size) for size in buffer_sizes]
    plt.yticks(ticks=np.arange(len(buffer_labels)) + 0.5, labels=buffer_labels, fontsize=SMALL_FONT_SIZE)

    plt.xlabel("# Nodes", fontsize=BIG_FONT_SIZE)
    plt.ylabel("Vector Size", fontsize=BIG_FONT_SIZE)
    plt.xticks(fontsize=SMALL_FONT_SIZE, rotation=0)

    outdir = cfg.output_dir
    if outdir is None:
        outdir = Path("plot") / cfg.system / "heatmaps" / cfg.collective
    target_dir = ensure_dir(outdir)

    args_parts = [
        f"nnodes_{','.join(cfg.sorted_nodes())}",
        f"target_{cfg.target_algo}",
        f"metric_{cfg.metric}",
    ]
    if cfg.exclude:
        args_parts.append(f"exclude_{cfg.exclude.replace('|', '_')}")
    outfile = target_dir / f"{'_'.join(args_parts)}.pdf"
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile
