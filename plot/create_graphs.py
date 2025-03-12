import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def format_bytes(x):
    # Ensure x is a float
    try:
        x = float(x)
    except ValueError:
        return x
    if x >= 1024**2:
        return f"{x/1024**2:.0f} MiB"
    elif x >= 1024:
        return f"{x/1024:.0f} KiB"
    else:
        return f"{x:.0f} B"

# Define a sort key to order the algorithms
def sort_key(algo: str):
    if algo.startswith("default"):
        return (0, algo)
    elif not algo.endswith("over"):
        return (1, algo)
    elif "swing" not in algo:
        return (2, algo)
    else:
        return (3, algo)

# Global error bar drawing function that supports both absolute and relative threshold modes.
def draw_errorbars(ax, data, sorted_algos, std_threshold, threshold_mode='absolute', loc=0.05, top=False, y_min = 2.0):
    """
    Draw error bars or red markers on the bars in the provided axis.
    
    Parameters:
    - ax: matplotlib axis object.
    - data: DataFrame containing the data.
    - sorted_algos: List of algorithm names in the desired order.
    - std_threshold: Threshold value. In 'absolute' mode, it's used directly; in 'relative' mode,
      the threshold is computed as std_threshold * y.
    - threshold_mode: Either 'absolute' (for generate_barplot) or 'relative' (for generate_cut_barplot).
    - loc: Vertical offset for the red marker.
    - top: In relative mode, if True, skip markers for bars below a certain height.
    """
    for i, algo in enumerate(sorted_algos):
        algo_group = data[data['algo_name'] == algo]
        # Retrieve the container for the current hue (assumes same order as sorted_algos)
        container = ax.containers[i]
        for bar, (_, row) in zip(container, algo_group.iterrows()):
            x = bar.get_x() + bar.get_width() / 2.0
            y = bar.get_height()
            std_dev = row['normalized_std']
            if threshold_mode == 'absolute':
                # In absolute mode, if the std deviation is above the threshold, mark with a red dot.
                if std_dev > std_threshold:
                    ax.scatter(x, y + loc, color='red', s=50, zorder=5)
                else:
                    ax.errorbar(x, y, yerr=std_dev, fmt='none', ecolor='black', capsize=3, zorder=4)
            elif threshold_mode == 'relative':
                # In relative mode, compute the threshold as a fraction of y.
                real_threshold = std_threshold * y
                if std_dev <= real_threshold:
                    ax.errorbar(x, y, yerr=std_dev, fmt='none', ecolor='black', capsize=3, zorder=4)
                else:
                    # In the top subplot of the cut barplot, skip markers for bars below a minimum height.
                    if top and y < y_min:
                        continue
                    ax.scatter(x, y + loc, color='red', s=50, zorder=5)
            else:
                raise ValueError("Invalid threshold_mode. Use 'absolute' or 'relative'.")


def normalize_dataset(data: pd.DataFrame, mpi_lib : str, base : str | None = None) -> pd.DataFrame:
    """
    Normalize the dataset by dividing the mean execution time of each algorithm
    by the mean execution time of the base algorithm.
    The base algorithm is the one with the name specified in the 'base' parameter,
    or the default algorithm for the MPI library if not specified.
    The normalized mean is stored in a new column 'normalized_mean' in the DataFrame.

    Parameters:
    - data:     DataFrame containing the data.
    - mpi_lib:  MPI library used in the test.
    - base:     Name of the base algorithm. If None, the default algorithm for the MPI
                library is used.
    """
    if base is None:
        if mpi_lib in ['OMPI', 'OMPI_SWING']:
            base = 'default_ompi'
        elif mpi_lib in ['MPICH', 'CRAY_MPICH']:
            base = 'default_mpich'

    grouped_data = data.groupby('buffer_size')

    normalized_means = pd.Series(index=data.index, dtype=float)
    normalized_stds = pd.Series(index=data.index, dtype=float)

    for buf, group in grouped_data:
        base_row = group[group['algo_name'] == base].copy()

        if base_row.empty:
            continue

        base_mean = base_row['mean'].iloc[0]
        normalized_means.loc[group.index] = group['mean'] / base_mean
        normalized_stds.loc[group.index] = (group['std'] / group['mean']) * normalized_means.loc[group.index]

    data['normalized_mean'] = normalized_means.fillna(1.0)
    data['normalized_std'] = normalized_stds.fillna(0.0)

    return data

def generate_lineplot(data: pd.DataFrame, system, collective, nnodes, datatype, timestamp):
    """
    Create line plots with logarithmic scale on both axis.
    """

    plt.figure(figsize=(12, 8))
    
    # Group data by algorithm and plot each group
    for algo, group in data.groupby('algo_name'):
        sorted_group = group.sort_values('buffer_size')
        if isinstance(sorted_group, pd.DataFrame): # silence warning with isinstance
            plt.plot(sorted_group['buffer_size'], sorted_group['mean'], label=algo, marker='*', markersize=5, linewidth=1)

    plt.title(f'{system}, {collective.lower()}, {nnodes} nodes', fontsize=18)
    plt.xlabel('Array Size (Bytes)', fontsize=15)
    plt.ylabel('Mean Execution Time (ns)', fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, linestyle='--', linewidth=0.25)
    plt.tight_layout()

    plt.legend(fontsize=13)

    name = f'{collective.lower()}_{nnodes}_{datatype}_{timestamp}_lineplot.png'
    dir = f'plot/{system}'
    full_name = os.path.join(dir, name)
    plt.savefig(full_name, dpi=300)
    plt.close()

def generate_barplot(data: pd.DataFrame, system, collective, nnodes, datatype, timestamp, std_threshold: float = 0.35):
    # Get the sorted list of unique algorithm names
    sorted_algos = sorted(data['algo_name'].unique().tolist(), key=sort_key)

    plt.figure(figsize=(12, 8))

    # Use the sorted list in hue_order so that the bars are plotted in our desired order
    ax = sns.barplot(
        data=data,
        x='buffer_size',
        y='normalized_mean',
        hue='algo_name',
        hue_order=sorted_algos,
        palette='tab10',
        errorbar=None
    )

    # Draw error markers using the global draw_errorbars in absolute mode.
    draw_errorbars(ax, data, sorted_algos, std_threshold, threshold_mode='absolute', loc=0.05)

    ax.set_xticks(ax.get_xticks())  # Silence warning on unbound number of ticks
    new_labels = []
    for t in ax.get_xticklabels():
        try:
            label_val = float(t.get_text())
            new_labels.append(format_bytes(label_val))
        except ValueError:
            new_labels.append(t.get_text())
    ax.set_xticklabels(new_labels)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper left', fontsize=9)

    plt.title(f'{system}, {collective.lower()}, {nnodes} nodes', fontsize=18)
    plt.xlabel('Message Size', fontsize=15)
    plt.ylabel('Normalized Execution Time', fontsize=15)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, axis='y')
    plt.tight_layout()

    name = f'{collective.lower()}_{nnodes}_{datatype}_{timestamp}_barplot.png'
    dir = f'plot/{system}'
    full_name = os.path.join(dir, name)
    plt.savefig(full_name, dpi=300)
    plt.close()

def generate_cut_barplot(data: pd.DataFrame, system, collective, nnodes, datatype, timestamp, std_threshold : float = 0.5) :
    # Compute the sorted order of algorithm names
    sorted_algos = sorted(data['algo_name'].unique().tolist(), key=sort_key)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios': [1, 3]},
        figsize=(12, 8)
    )

    # Create the two barplots with the same hue order.
    sns.barplot(
        ax=ax_top,
        data=data,
        x='buffer_size',
        y='normalized_mean',
        hue='algo_name',
        hue_order=sorted_algos,
        palette='tab10',
        errorbar=None
    )
    sns.barplot(
        ax=ax_bot,
        data=data,
        x='buffer_size',
        y='normalized_mean',
        hue='algo_name',
        hue_order=sorted_algos,
        palette='tab10',
        errorbar=None
    )

    # Remove duplicate legend from the top axis
    if ax_top.get_legend():
        ax_top.get_legend().remove()

    # Set y-limits: lower axis from 0 to 3.5, upper axis from 3.5 to a bit above the max
    y_min = 1.8
    y_max = data['normalized_mean'].max() * 1.1  # add 10% headroom
    if y_max > 10.0:
        y_max = 10.0

    # Draw errorbars on both axes
    draw_errorbars(ax_top, data, sorted_algos, std_threshold, threshold_mode='relative', loc=(y_max - y_min) * 0.1, top=True, y_min=y_min)
    draw_errorbars(ax_bot, data, sorted_algos, std_threshold, threshold_mode='relative', loc=0.05, y_min=y_min)
    ax_bot.set_ylim(0, y_min - 0.05)
    ax_top.set_ylim(y_min, y_max)

    # Add markers for bars that exceed the top axis limit (y_max = 10)
    top_limit = ax_top.get_ylim()[1]
    for container in ax_top.containers:
        for bar in container:
            # Check if the bar's true height is greater than the current top limit
            if hasattr(bar, "get_height") and bar.get_height() > top_limit:
                x = bar.get_x() + bar.get_width() / 2.0
                # Place a marker (an upward pointing triangle here) just inside the top limit
                ax_top.scatter(x, top_limit - 0.5, marker='^', color='black', s=100, zorder=4)
    # Hide the spines between the two plots and adjust ticks
    ax_top.spines['bottom'].set_visible(True)
    ax_bot.spines['top'].set_visible(True)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Add break markers (diagonal lines) on the y-axis
    d = .005  # size of break mark in axes coordinates
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Add grid lines to the bottom axis
    ax_top.grid(True, which='both', linestyle='-', linewidth=0.25, axis='y')
    ax_bot.grid(True, which='both', linestyle='-', linewidth=0.5, axis='y')

    # Set labels and title (set overall title using suptitle)
    ax_bot.set_xlabel('Buffer Size', fontsize=15)
    ax_bot.set_ylabel('Normalized Mean Execution Time', fontsize=15)
    ax_top.set_ylabel('')
    fig.suptitle(f'{system}, {collective.lower()}, {nnodes} nodes ({datatype})', fontsize=18)

    ax_bot.set_xticks(ax_bot.get_xticks()) # Silence warning on unbound number of ticks
    new_labels = []
    for t in ax_bot.get_xticklabels():
        try:
            # Convert the current label (as string) to float before formatting
            label_val = float(t.get_text())
            new_labels.append(format_bytes(label_val))
        except ValueError:
            new_labels.append(t.get_text())
    ax_bot.set_xticklabels(new_labels)

    # Add legend to the bottom axis
    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        ax_bot.legend(handles, labels, loc='lower left', fontsize=7)

    # Adjust layout to make room for the suptitle
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))


    # Save the figure
    name = f'{collective.lower()}_{nnodes}_{datatype}_{timestamp}_barplot_cut.png'
    dir = f'plot/{system}'
    full_name = os.path.join(dir, name)
    plt.savefig(full_name, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate graphs")
    parser.add_argument("--summary-file", required=True, help="Path to the summarized results")
    parser.add_argument("--datatype", required=False, help="Data type to graph, defaults to all the presents")
    parser.add_argument("--algorithm", required=False, help="Algorithm to graph, defaults to all the presents")
    parser.add_argument("--collective", required=False, help="Collective type to graph, defaults to all the presents")
    parser.add_argument("--filter-by", required=False, help="Filter algorithms name by substrings (e.g swing,ompi,mpich)")
    args = parser.parse_args()

    if not os.path.isfile(args.summary_file):
        print(f"Summary file {args.summary_file} not found. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.summary_file)

    # Drop the columns with unvarying values from the summary
    system_name = df['system'].iloc[0]
    timestamp = df['timestamp'].iloc[0]
    mpi_lib = df['mpi_lib'].iloc[0]
    nnodes = df['nnodes'].iloc[0]

    drop_cols = ['array_dim','nnodes','system','timestamp','test_id','MPI_Op',
                 'CUDA','notes','mpi_lib','mpi_lib_version','libswing_version']
    df.drop(columns=drop_cols, inplace=True)

    if args.collective is not None:
        df = df[df["collective_type"] == args.collective]
    if args.datatype is not None:
        df = df[df["datatype"] == args.datatype]
    if args.algorithm is not None:
        algorithms = [algo.strip() for algo in args.algorithm.split(',')]
        df = df[df["algo_name"].isin(algorithms)]
    if args.filter_by is not None:
        filter_by = [filt.strip() for filt in args.filter_by.split(',')]
        df = df[df["algo_name"].str.contains('|'.join(filter_by), case=False, na=False)]

    # Control if the dataframe is not empty at this point
    if df.empty:
        print(f"Filtered data is empty. Exiting.", file=sys.stderr)
        sys.exit(1)

    dir = f'plot/{system_name}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for datatype, group in df.groupby('datatype'):
        for collective, subgroup in group.groupby('collective_type'):
            generate_lineplot(subgroup, system_name, collective, nnodes, datatype, timestamp);
            normalized_data = normalize_dataset(subgroup, mpi_lib)
            generate_barplot(normalized_data, system_name, collective, nnodes, datatype, timestamp)
            generate_cut_barplot(normalized_data, system_name, collective, nnodes, datatype, timestamp)

if __name__ == "__main__":
    main()

