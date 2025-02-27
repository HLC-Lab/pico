import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def normalize_dataset(data: pd.DataFrame, mpi_lib : str, base : str | None = None) -> pd.DataFrame:
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

def generate_barplot(data: pd.DataFrame, system, collective, nnodes, datatype, timestamp, std_threshold : float = 0.35) :

    plt.figure(figsize=(12, 8))

    ax = sns.barplot(
        data=data,
        x='buffer_size',
        y='normalized_mean',
        hue='algo_name',
        palette='tab10',
        errorbar=None
    )


    for algo in data['algo_name'].unique():
        algo_group = data[data['algo_name'] == algo]
        container = ax.containers[data['algo_name'].unique().tolist().index(algo)]

        for bar, (_, row) in zip(container, algo_group.iterrows()):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            std_dev = row['normalized_std']
            # if std_dev is too big place a red dot, otherwise put error bars
            if std_dev > std_threshold:
                ax.scatter(x, y + 0.05, color='red', s=50, zorder=5) 
            else:
                ax.errorbar(x, y, yerr=std_dev, fmt='none', ecolor='black', capsize=3, zorder=4) 

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

if __name__ == "__main__":
    main()

