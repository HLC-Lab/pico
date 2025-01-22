import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# The dictionary is in the format:
# algorithm_number : [ name, plotting color, marker style ]
plot_data_dict = {
    0: ["Default", "#7f7f7f", "."],  # Gray
    1: ["Linear", "#ff9896", "2"],  # Light Red
    2: ["Non Overlapping", "#c5b0d5", "|"],  # Lavender
    3: ["Recursive Doubling", "#d62728", "s"],  # Red
    4: ["Ring", "#2ca02c", "v"],  # Green
    5: ["Ring Segmented", "#98df8a", "^"],  # Light Green
    6: ["Rabenseifner", "#9467bd", "p"],  # Purple
    7: ["Allgather Reduce", "#ffbb78", "1"],  # Peach
    8: ["Swing Lat", "#1f77b4", "*"],  # Blue
    9: ["Swing Bdw mcpy", "#17becf", "8"],  # Cyan
    10: ["Swing Bdw dt 1", "#bcbd22", "d"],  # Olive
    11: ["Swing Bdw dt 2", "#8c564b", "o"],  # Brown
    12: ["Swing Bdw seg", "#e377c2", "D"],  # Pink
    13: ["Swing Bdw static", "#ff7f0e", "X"],
    14: ["Recursive Doubling OVER", "#ff69b4", "H"],  # 
    15: ["Swing Lat OVER", "#9b59b6", "x"],  # Orange
    16: ["Swing Bdw static OVER", "#f39c12", "P"]  # Yellow Orange
    # Add more mappings as needed
}

# The dictionary is in the format:
# number of elements : [ size for 32 bits dt, size for 64 bits dt]
buffer_size_dict = {
    8: ['32B', '64B'],
    64: ['256B', '512B'],
    512: ['2KiB', '4KiB'],
    2048: ['8KiB', '16KiB'],
    16384: ['64KiB', '128KiB'],
    131072: ['512KiB', '1MiB'],
    1048576: ['4MiB', '8MiB'],
    8388608: ['32MiB', '64MiB'],
    67108864: ['256MiB', '512MiB']
}


def generate_lineplots(data, output_dir : str,
                       system : str):
    """
    Create line plots with logarithmic scale on both axis.

    Parameters:
    - data: DataFrame containing the data to plot
    - output_dir: Directory where the plots will be saved
    - system: String to indicate the system
    """
    # Create a subdirectory for graphs
    lineplot_dir = os.path.join(output_dir, 'lineplots/')
    os.makedirs(lineplot_dir, exist_ok=True)

    for nof_proc, group in data.groupby('nof_proc'):
        # Sort by alg_number to ensure ordered plotting
        sorted_group = group.sort_values(by='alg_number')

        plt.figure(figsize=(12, 8))
        for idx, alg in enumerate(sorted_group['alg_number'].unique()):
            alg_data = sorted_group[sorted_group['alg_number'] == alg]
            alg_data = alg_data.sort_values(by='array_size')
            plt.plot(
                alg_data['array_size'],
                alg_data['mean'],
                label=alg_data['algo_name'].iloc[0],
                color=alg_data['color'].iloc[0],
                marker=alg_data['marker'].iloc[0],
                markersize=7,
                linewidth=1.5)

        # Set graph title and labels
        plt.title(f'{system}, {nof_proc} nodes', fontsize=18)
        plt.xlabel(f'Array Size, {group["dtype"].iloc[0]}', fontsize=15)
        plt.ylabel('Mean Execution Time (ns)', fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.tight_layout()

        plt.legend(fontsize=13)

        plt.savefig(os.path.join(lineplot_dir, \
                    f'{system}_{group["dtype"].iloc[0]}_{nof_proc}.png'), \
                    dpi=300)
        plt.close()


def generate_barplots(data, output_dir : str, \
                      system : str, \
                      std_threshold : float = 0.25) :
    """
    Create bar plots with custom error bars and highlight bars
    with high standard deviation.

    Parameters:
    - data: DataFrame containing the data to plot
    - output_dir: Directory where the plots will be saved
    - system: String to indicate the system
    - std_threshold: Threshold for normalized_std to highlight
      unreliable confidence intervals
    """
    barplot_dir = os.path.join(output_dir, 'barplots/')
    os.makedirs(barplot_dir, exist_ok=True)

    colors = {algo_name: plot_data_dict[i][1] \
            for i, (algo_name, _, _) in plot_data_dict.items()}
    grouped_data = data.groupby('nof_proc')

    for nof_proc, group in grouped_data:
        plt.figure(figsize=(12, 8))

        ax = sns.barplot(
            data=group,
            x='buffer_size',
            y='normalized_mean',
            hue='algo_name',
            palette=colors,
            errorbar=None
        )

        for container, algo_name in zip(ax.containers, \
                                        group['algo_name'].unique()):
            algo_group = group[group['algo_name'] == algo_name]
            for bar, (_, row) in zip(container, algo_group.iterrows()):
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                std_dev = row['normalized_std']
                # if std_dev is too big place a red dot,
                # otherwise put error bars
                if std_dev > std_threshold:
                    ax.scatter(x, y + 0.05, color='red', s=50, zorder=5) 
                else:
                    ax.errorbar(x, y, yerr=std_dev, fmt='none', \
                                ecolor='black', capsize=3, zorder=4) 

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower left', fontsize=15)
        plt.title(f'{system}, {nof_proc} nodes', fontsize=18)
        plt.xlabel('Message Size', fontsize=15)
        plt.ylabel('Normalized Execution Time', fontsize=15)

        plt.grid(True, which='both', linestyle='-', linewidth=0.5, axis='y')
        plt.tight_layout()

        dtype = group['dtype'].iloc[0]
        plt.savefig(os.path.join(barplot_dir, \
                                 f'{system}_barplot_{dtype}_{nof_proc}.png'), \
                                 dpi=300)
        plt.close()



def filter_dataset(data, alg: list[int] | None = None, \
                sizes: list[int] | None =None, \
                max_size: int | None = None, \
                min_size: int | None = None) -> pd.DataFrame:
    """
    Filters dataset based on algorithm, array size range,
    or specific sizes.

    Parameters:
    - data: DataFrame to filter.
    - alg: List of algorithm numbers to include.
    - max_size: Maximum array size.
    - min_size: Minimum array size.
    - sizes: List of specific array sizes to include.

    Returns:
    - Filtered DataFrame.
    """
    if alg is not None:
        data = data[data['alg_number'].isin(alg)]
    if max_size is not None:
        data = data[data['array_size'] < max_size]
    if min_size is not None:
        data = data[data['array_size'] > min_size]
    if sizes is not None:
        data = data[data['array_size'].isin(sizes)]
    return data


def normalize_dataset(data, base: int = 0) -> pd.DataFrame:
    """
    Normalizes data relative to the execution times of a base algorithm.

    Parameters:
    - data: DataFrame to normalize.
    - base: Algorithm number to use as the base for normalization.

    Returns:
    - DataFrame with normalized_mean and normalized_std columns.
    """
    # Check if the base algorithm exists in the data
    if base not in data['alg_number'].values:
        return data

    grouped_data = data.groupby(['nof_proc', 'array_size'])

    # Prepare empty Series for normalized values
    normalized_means = pd.Series(index=data.index, dtype=float)
    normalized_stds = pd.Series(index=data.index, dtype=float)

    # Normalize within each group
    for (nof_proc, array_size), group in grouped_data:
        base_row = group[group['alg_number'] == base]

        # Skip normalization for groups without the base algorithm
        if base_row.empty:
            continue

        base_mean = base_row['mean'].iloc[0]
        normalized_means.loc[group.index] = group['mean'] / base_mean
        normalized_stds.loc[group.index] = group['std_dev'] / base_mean

    # Add normalized columns to the data
    data['normalized_mean'] = normalized_means
    data['normalized_std'] = normalized_stds

    return data

def add_labels(data) -> pd.DataFrame:
    """
    Enriches the input DataFrame with additional columns for plotting
    and analysis. It maps algorithm numbers to their corresponding names,
    colors, and markers, and assigns human-readable buffer sizes based
    on the data type (`dtype`).

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The input DataFrame with additional columns.
    """
    # Map 'alg_number' to 'algo_name', 'color', and 'marker' using plot_data_dict
    data['algo_name'] = data['alg_number'].map(lambda x: \
                        plot_data_dict.get(x, [None])[0])
    data['color'] = data['alg_number'].map(lambda x: \
                        plot_data_dict.get(x, [None])[1])
    data['marker'] = data['alg_number'].map(lambda x: \
                        plot_data_dict.get(x, [None])[2])
    

    # Assign 'buffer_size' based on the data type (dtype)
    # FIX: THIS ASSUMES THAT THE DATAFRAME CONTAINS ONLY ONE DATATYPE
    if data['dtype'].iloc[0] in ['int', 'int32', 'float', 'float32']:
        data['buffer_size'] = data['array_size'].map(lambda x: \
                                buffer_size_dict.get(x, [None])[0])
    elif data['dtype'].iloc[0] in ['int64', 'float64', 'double']:
        data['buffer_size'] = data['array_size'].map(lambda x: \
                                buffer_size_dict.get(x, [None])[1])
    else:
        data['buffer_size'] = None

    return data


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate graphs for \
                                    algorithms based on test results.")
    parser.add_argument("--algorithms", nargs='+', type=int, required=True,
                        help="List of algorithm numbers to graph. The \
                              first one is used for normalization.")
    parser.add_argument("--system", type=str, required=True,
                        help="System name where the tests are performed \
                              (e.g., 'leonardo').")

    args = parser.parse_args()

    algorithms_to_graph = args.algorithms
    system = args.system

    root_folder = os.getcwd()

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        summary = os.path.join(subfolder_path, "summary", "results_summary.csv")

        if not os.path.isfile(summary):
            print(f"Summary of {subfolder_path} not made, skipping.")
            continue

        # Create graphs directory
        graph_dir = os.path.join(subfolder_path, 'graphs/')
        os.makedirs(graph_dir, exist_ok=True)

        data = filter_dataset(pd.read_csv(summary), algorithms_to_graph)
        if data.empty:
            print(f"Filtered data of {subfolder_path} is empty, skipping.")
            continue

        # Add columns to the DataFrame useful for plotting
        data = add_labels(data).sort_values(by=['array_size', 'nof_proc', 'alg_number'])

        generate_lineplots(data, output_dir=graph_dir, system=system)

        # Try normalizing the data and handle the case where base is not found
        data = normalize_dataset(data, algorithms_to_graph[0])
        # Check if normalization was skipped and if base algorithm was not found
        if 'normalized_mean' not in data.columns \
                or 'normalized_std' not in data.columns:
            print(f"Base algorithm {algorithms_to_graph[0]} not found for \
                    {subfolder}, barplots not created.")
        else:
            generate_barplots(data, output_dir=graph_dir, system=system)
