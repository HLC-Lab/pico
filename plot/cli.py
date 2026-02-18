# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd

from .data import (
    drop_unused_columns,
    extract_metadata,
    filter_summary,
    normalize_dataset,
    read_summary,
    SummaryEmptyError,
)
from .plots import generate_bar_plot, generate_cut_bar_plot, generate_line_plot
from .plots.plot_bine_heatmap import BestBineHeatmapConfig, generate_best_bine_heatmap
from .plots.box_plot import BoxplotConfig, generate_boxplot
from .plots.family_heatmap import FamilyHeatmapConfig, generate_family_heatmap
from .plots.comparison_heatmap import ComparisonHeatmapConfig, generate_comparison_heatmap
from .plots.stacked_latency_plot import generate_stacked_latency_bars
from .plots.refined_loader import RefinedDataset, load_data
from .plots.refined_line_plot import generate_refined_line_plot


def _split_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _load_filtered_dataframe(args) -> tuple[pd.DataFrame, object]:
    df = read_summary(args.summary_file)
    metadata = extract_metadata(df)
    df = drop_unused_columns(df)
    df = filter_summary(
        df,
        collective=args.collective,
        datatype=args.datatype,
        algorithm=_split_list(args.algorithm),
        filter_by=_split_list(args.filter_by),
        filter_out=_split_list(args.filter_out),
        min_dim=args.min_dim,
        max_dim=args.max_dim,
    )
    return df, metadata


def _iter_groups(df: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    for datatype, subdf in df.groupby("datatype"):
        for collective, subgroup in subdf.groupby("collective_type"):
            for gpu_awareness, subsubgroup in subgroup.groupby("gpu_awareness"):
                yield datatype, collective, gpu_awareness, subsubgroup.copy()


def _add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datatype", help="Data type to keep (comma separated for multiple).")
    parser.add_argument("--collective", help="Collective to keep.")
    parser.add_argument("--algorithm", help="Filter algorithms by exact name (comma separated).")
    parser.add_argument("--filter-by", help="Keep algorithms whose name contains any of these substrings.")
    parser.add_argument("--filter-out", help="Discard algorithms whose name contains any of these substrings.")
    parser.add_argument("--min-dim", type=int, help="Minimum buffer size to keep.")
    parser.add_argument("--max-dim", type=int, help="Maximum buffer size to keep.")
    parser.add_argument("--output-dir", help="Directory where figures are written.")


def _line_command(args) -> None:
    df, metadata = _load_filtered_dataframe(args)
    for datatype, collective, gpu_awareness, subset in _iter_groups(df):
        generate_line_plot(
            subset,
            metadata=metadata,
            collective=collective,
            datatype=datatype,
            gpu_awareness=gpu_awareness,
            error_col=args.error_col,
            error_mode=args.error_mode,
            output_dir=args.output_dir,
        )


def _bar_command(args) -> None:
    df, metadata = _load_filtered_dataframe(args)
    for datatype, collective, gpu_awareness, subset in _iter_groups(df):
        normalized = normalize_dataset(
            subset,
            mpi_lib=metadata.mpi_lib,
            gpu_lib=metadata.gpu_lib,
            base=args.normalize_by,
        )
        generate_bar_plot(
            normalized,
            metadata=metadata,
            collective=collective,
            datatype=datatype,
            gpu_awareness=gpu_awareness,
            errorbars=args.errorbars,
            k=args.k,
            threshold=args.std_threshold,
            marker_loc=args.marker_loc,
            output_dir=args.output_dir,
        )


def _cut_command(args) -> None:
    df, metadata = _load_filtered_dataframe(args)
    for datatype, collective, gpu_awareness, subset in _iter_groups(df):
        normalized = normalize_dataset(
            subset,
            mpi_lib=metadata.mpi_lib,
            gpu_lib=metadata.gpu_lib,
            base=args.normalize_by,
        )
        generate_cut_bar_plot(
            normalized,
            metadata=metadata,
            collective=collective,
            datatype=datatype,
            gpu_awareness=gpu_awareness,
            errorbars=args.errorbars,
            k=args.k,
            threshold=args.std_threshold,
            marker_loc=args.marker_loc,
            output_dir=args.output_dir,
        )


def _suite_command(args) -> None:
    df, metadata = _load_filtered_dataframe(args)
    for datatype, collective, gpu_awareness ,subset in _iter_groups(df):
        generate_line_plot(
            subset,
            metadata=metadata,
            collective=collective,
            datatype=datatype,
            gpu_awareness=gpu_awareness,
            error_col=args.error_col,
            error_mode=args.error_mode,
            output_dir=args.output_dir,
        )
        normalized = normalize_dataset(
            subset,
            mpi_lib=metadata.mpi_lib,
            gpu_lib=metadata.gpu_lib,
            base=args.normalize_by,
        )
        generate_bar_plot(
            normalized,
            metadata=metadata,
            collective=collective,
            datatype=datatype,
            gpu_awareness=gpu_awareness,
            output_dir=args.output_dir,
        )
        generate_cut_bar_plot(
            normalized,
            metadata=metadata,
            collective=collective,
            datatype=datatype,
            gpu_awareness=gpu_awareness,
            output_dir=args.output_dir,
        )




def _boxplot_command(args) -> None:
    cfg = BoxplotConfig(
        system=args.system,
        nnodes=_split_list(args.nnodes) or [],
        tasks_per_node=args.tasks_per_node,
        notes=args.notes,
        exclude=args.exclude,
        metric=args.metric,
        output_dir=args.output_dir,
    )
    generate_boxplot(cfg)
def _heatmap_command(args) -> None:
    cfg = FamilyHeatmapConfig(
        system=args.system,
        collective=args.collective,
        nnodes=_split_list(args.nnodes) or [],
        tasks_per_node=args.tasks_per_node,
        notes=args.notes,
        exclude=args.exclude,
        metric=args.metric,
        hide_y_labels=args.hide_y_labels,
        output_dir=args.output_dir,
    )
    generate_family_heatmap(cfg)


def _comparison_heatmap_command(args) -> None:
    cfg = ComparisonHeatmapConfig(
        system=args.system,
        collective=args.collective,
        nnodes=_split_list(args.nnodes) or [],
        target_algo=args.target_algo,
        tasks_per_node=args.tasks_per_node,
        notes=args.notes,
        exclude=args.exclude,
        metric=args.metric,
        show_names=args.show_names,
        output_dir=args.output_dir,
    )
    generate_comparison_heatmap(cfg)


def _bine_heatmap_command(args) -> None:
    cfg = BestBineHeatmapConfig(
        system=args.system,
        collective=args.collective,
        metric=args.metric,
        runs=args.runs,
        output=args.output,
    )
    generate_best_bine_heatmap(cfg)


def _refined_command(args) -> None:
    dataset = RefinedDataset()
    messages = [msg.strip() for msg in args.messages.split(",") if msg.strip()]
    collectives = [c.strip() for c in args.collective.split(",") if c.strip()]

    cluster_paths = {
        "baseline": args.baseline,
        "op_null": args.op_null,
        "no_memcpy": args.no_memcpy,
        "no_memcpy_op_null": args.no_memcpy_op_null,
    }

    for collective in collectives:
        load_data(
            dataset,
            "baseline",
            args.nodes,
            cluster_paths["baseline"],
            messages,
            coll=collective,
            congested=args.congested,
        )
        load_data(
            dataset,
            "op_null",
            args.nodes,
            cluster_paths["op_null"],
            messages,
            coll=collective,
            congested=args.congested,
        )
        load_data(
            dataset,
            "no_memcpy",
            args.nodes,
            cluster_paths["no_memcpy"],
            messages,
            coll=collective,
            congested=args.congested,
        )
        load_data(
            dataset,
            "no_memcpy_op_null",
            args.nodes,
            cluster_paths["no_memcpy_op_null"],
            messages,
            coll=collective,
            congested=args.congested,
        )

        title = args.title or f"{args.label}, {collective.capitalize()}, {args.nodes} nodes"
        generate_refined_line_plot(dataset, title, output_dir=args.output_dir)
        generate_stacked_latency_bars(
            dataset,
            title,
            sizes=tuple(messages),
            output_dir=args.output_dir,
        )
        dataset.reset()


def _add_summary_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--summary-file", required=True, help="Path to aggregated summary CSV.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plotting CLI for pico results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    line_parser = subparsers.add_parser("line", help="Generate latency line plots.")
    _add_summary_argument(line_parser)
    _add_common_filters(line_parser)
    line_parser.add_argument("--error-col", default="se", help="Error column to use (default: se).")
    line_parser.add_argument(
        "--error-mode",
        choices=("band", "bar"),
        default="band",
        help="Render errors as shaded band or bar style.",
    )
    line_parser.set_defaults(func=_line_command)

    bar_parser = subparsers.add_parser("bar", help="Generate normalized bar plots.")
    _add_summary_argument(bar_parser)
    _add_common_filters(bar_parser)
    bar_parser.add_argument("--normalize-by", help="Reference algorithm used for normalization.")
    bar_parser.add_argument( "--errorbars", choices=("none", "se", "ci"), default="se",
        help="Error bar mode for bar plots: none, se (symmetric), or ci (asymmetric).",)
    bar_parser.add_argument( "--k", type=float, default=1.96,
        help="Multiplier for SE error bars (e.g., 1.0 for ±SE, 1.96 for ~95%%).",)
    bar_parser.add_argument( "--std-threshold", type=float, default=0.15,
        help="Error marker threshold (applies to chosen error metric).",)
    bar_parser.add_argument( "--marker-loc", type=float, default=0.05,
        help="Vertical offset for the red marker when threshold is exceeded.",)
    bar_parser.set_defaults(func=_bar_command)

    cut_parser = subparsers.add_parser("cut", help="Generate split normalized bar plots.")
    _add_summary_argument(cut_parser)
    _add_common_filters(cut_parser)
    cut_parser.add_argument("--normalize-by", help="Reference algorithm used for normalization.")
    cut_parser.add_argument( "--errorbars", choices=("none", "se", "ci"), default="se",
        help="Error bar mode for cut bar plots: none, se (symmetric), or ci (asymmetric).",)
    cut_parser.add_argument("--k", type=float, default=1.96, help="Multiplier for SE error bars.")
    cut_parser.add_argument("--std-threshold", type=float, default=0.5, help="Error marker threshold.")
    cut_parser.add_argument("--marker-loc", type=float, default=0.05, help="Marker vertical offset.")
    cut_parser.set_defaults(func=_cut_command)

    suite_parser = subparsers.add_parser("summary", help="Recreate the legacy create_graphs pipeline.")
    _add_summary_argument(suite_parser)
    _add_common_filters(suite_parser)
    suite_parser.add_argument("--normalize-by", help="Reference algorithm used for normalization.")
    suite_parser.add_argument("--std-threshold", type=float, default=0.15, help="Std threshold for bar plot markers.")
    suite_parser.add_argument(
        "--cut-std-threshold",
        type=float,
        default=0.5,
        help="Std threshold for cut bar plot markers.",
    )
    suite_parser.add_argument("--error-col", default="se", help="Error column to use for line plots.")
    suite_parser.add_argument(
        "--error-mode",
        choices=("band", "bar"),
        default="band",
        help="Render errors as shaded band or classic error bars.",
    )
    suite_parser.set_defaults(func=_suite_command)

    box_parser = subparsers.add_parser("boxplot", help="Generate improvement boxplots.")
    box_parser.add_argument("--system", required=True, help="System name (e.g. leonardo).")
    box_parser.add_argument("--nnodes", required=True, help="Comma separated list of node counts.")
    box_parser.add_argument("--tasks-per-node", type=int, default=1, dest="tasks_per_node", help="Tasks per node.")
    box_parser.add_argument("--notes", help="Filter metadata entries by notes column.")
    box_parser.add_argument("--exclude", help="Exclude algorithms that match this substring (regex).")
    box_parser.add_argument(
        "--metric",
        choices=("mean", "median", "percentile_90"),
        default="mean",
        help="Metric to convert into bandwidth for ranking.",
    )
    box_parser.add_argument("--output-dir", help="Target directory for the generated boxplot.")
    box_parser.set_defaults(func=_boxplot_command)

    refined_parser = subparsers.add_parser("refined", help="Generate refined line and latency plots.")
    refined_parser.add_argument("--nodes", type=int, default=8, help="Number of nodes.")
    refined_parser.add_argument("--messages", default="32 B,256 B,2 KiB,16 KiB,128 KiB,1 MiB,8 MiB,64 MiB,512 MiB", help="Comma separated message sizes.")
    refined_parser.add_argument("--collective", default="allreduce", help="Comma separated collectives.")
    refined_parser.add_argument("--baseline", required=True, help="Directory for baseline traces.")
    refined_parser.add_argument("--op-null", required=True, dest="op_null", help="Directory for op_null traces.")
    refined_parser.add_argument("--no-memcpy", required=True, dest="no_memcpy", help="Directory for no_memcpy traces.")
    refined_parser.add_argument("--no-memcpy-op-null", required=True, dest="no_memcpy_op_null", help="Directory for no_memcpy_op_null traces.")
    refined_parser.add_argument("--congested", action="store_true", help="Include congested data files.")
    refined_parser.add_argument("--label", default="Experiment", help="Label prefix used in plot titles.")
    refined_parser.add_argument("--title", help="Override plot title (applied per collective).")
    refined_parser.add_argument("--output-dir", help="Directory where figures are written.")
    refined_parser.set_defaults(func=_refined_command)

    heatmap_parser = subparsers.add_parser("heatmap", help="Generate algorithm family heatmaps.")
    heatmap_parser.add_argument("--system", required=True, help="System name (e.g. leonardo).")
    heatmap_parser.add_argument("--collective", required=True, help="Collective to plot.")
    heatmap_parser.add_argument("--nnodes", required=True, help="Comma separated list of node counts.")
    heatmap_parser.add_argument("--tasks-per-node", type=int, default=1, dest="tasks_per_node", help="Tasks per node.")
    heatmap_parser.add_argument("--notes", help="Filter metadata entries by notes column.")
    heatmap_parser.add_argument("--exclude", help="Exclude algorithms that match this substring (regex).")
    heatmap_parser.add_argument(
        "--metric",
        choices=("mean", "median", "percentile_90"),
        default="mean",
        help="Metric to convert into bandwidth for ranking.",
    )
    heatmap_parser.add_argument("--hide-y-labels", action="store_true", help="Hide y-axis labels.")
    heatmap_parser.add_argument("--output-dir", help="Target directory for the generated heatmap.")
    heatmap_parser.set_defaults(func=_heatmap_command)

    comp_parser = subparsers.add_parser("comparison-heatmap", help="Generate target algorithm ratio heatmaps.")
    comp_parser.add_argument("--system", required=True, help="System name (e.g. leonardo).")
    comp_parser.add_argument("--collective", required=True, help="Collective to plot.")
    comp_parser.add_argument("--nnodes", required=True, help="Comma separated list of node counts.")
    comp_parser.add_argument("--target-algo", default="ring_ompi", help="Algorithm to compare against the best one.")
    comp_parser.add_argument("--tasks-per-node", type=int, default=1, dest="tasks_per_node", help="Tasks per node.")
    comp_parser.add_argument("--notes", help="Filter metadata entries by notes column.")
    comp_parser.add_argument("--exclude", help="Exclude algorithms that match this substring (regex).")
    comp_parser.add_argument(
        "--metric",
        choices=("mean", "median", "percentile_90"),
        default="mean",
        help="Metric to convert into bandwidth for ranking.",
    )
    comp_parser.add_argument("--show-names", action="store_true", help="Annotate cells with algorithm names.")
    comp_parser.add_argument("--output-dir", help="Target directory for the generated heatmap.")
    comp_parser.set_defaults(func=_comparison_heatmap_command)

    bine_parser = subparsers.add_parser("bine-heatmap", help="Generate best-Bine variant heatmaps.")
    bine_parser.add_argument("--system", required=True, help="System name (e.g. leonardo).")
    bine_parser.add_argument(
        "--collective",
        required=True,
        choices=("ALLGATHER", "REDUCE_SCATTER"),
        help="Collective to plot.",
    )
    bine_parser.add_argument(
        "--metric",
        choices=("mean", "median", "percentile_90"),
        default="mean",
        help="Metric used to compute bandwidth.",
    )
    bine_parser.add_argument(
        "--runs",
        nargs="+",
        metavar="TIMESTAMP:NODES",
        help="List of result runs (e.g. 2025_04_06___13_24_31:64).",
    )
    bine_parser.add_argument(
        "--output",
        help="Optional output PDF path. Defaults to plot/<system>/heatmaps/<collective>/<system>_<collective>_best_bine_variant.pdf",
    )
    bine_parser.set_defaults(func=_bine_heatmap_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SummaryEmptyError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())