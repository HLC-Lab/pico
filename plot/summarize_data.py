# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

import os
import sys
import re
import argparse
import tarfile
import numpy as np
import pandas as pd

DTYPE_TO_BYTES = {
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8,
    'float': 4,
    'double': 8,
    'char': 1,
    'int': 4
}


def _meta_value(meta_row: pd.Series, column: str):
    """Return column value from metadata row, normalising missing/null entries."""
    if column not in meta_row.index:
        return None
    value = meta_row[column]
    if isinstance(value, str) and value.lower() == "null":
        return None
    return value


def process_benchmark_file(file_path, warmup_ratio=0.2):
    """
    Read a benchmark CSV file, discard the warmup iterations,
    extract the 'highest' column, and compute summary statistics.
    """
    #print(f"Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        print(f"Empty data error for file: {file_path}", file=sys.stderr)
        return None


    if "fugaku" in file_path:
        # Discard last line
        df = df.iloc[:-1]
        # Warmup already done before printing data
        warmup_ratio = 0.0
        
    warmup_count = int(len(df) * warmup_ratio)
    df = df.iloc[warmup_count:]

    highest = df['highest'].values.astype(np.float64)
    # Drop negative values
    highest = highest[highest >= 0]
    n_iter = len(highest)

    mean = np.mean(highest)
    median = np.median(highest)
    std = np.std(highest)
    var = np.var(highest)
    min_val = np.min(highest)
    max_val = np.max(highest)

    percentile_10 = np.percentile(highest, 10)
    percentile_25 = np.percentile(highest, 25)
    percentile_75 = np.percentile(highest, 75)
    percentile_90 = np.percentile(highest, 90)
    iqr = percentile_75 - percentile_25

    # Standard Error and Confidence Interval (95% CI)
    if n_iter > 1:
        standard_error = std / np.sqrt(n_iter)
        ci_lower = mean - 1.96 * standard_error
        ci_upper = mean + 1.96 * standard_error
    else:
        standard_error = None
        ci_lower, ci_upper = None, None

    # Outlier detection using the IQR method
    outliers = highest[(highest < (percentile_25 - 1.5 * iqr))
                       | (highest > (percentile_75 + 1.5 * iqr))]
    num_outliers = len(outliers)

    stats = {
        'mean': mean,
        'median': median,
        'std': std,
        'var': var,
        'min': min_val,
        'max': max_val,
        'n_iter': n_iter,
        'percentile_10': percentile_10,
        'percentile_25': percentile_25,
        'percentile_75': percentile_75,
        'percentile_90': percentile_90,
        'iqr': iqr,
        'standard_error': standard_error,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        # 'outliers': list(outliers),
        'num_outliers': num_outliers
    }
    return stats


def parse_filename(filename):
    """
    Parse the filename to extract the array dimension, algorithm name, and data type.
    Expected format: <array_dim>_<algo_name>_<dtype>.csv
    Example: 512_bine_static_over_int32.csv
    """
    # Dirty, but do not want to waste time on regex
    # Mannaggia a Lorenzo che non ha messo il datatype nel filename
    filename = filename.split(".")[0]  # Remove the .csv extension
    fields = filename.split("_")
    array_dim = int(fields[0])
    if fields[-1] != "int32":
        dtype = "int32"
        algo_name = "_".join(fields[1:])
    else:
        dtype = fields[-1]
        algo_name = "_".join(fields[1:-1])

    dtype_size = DTYPE_TO_BYTES.get(dtype, 4)  # Default to 4 bytes if unknown
    buffer_size = array_dim * dtype_size
    return {
        'array_dim': array_dim,
        'algo_name': algo_name,
        'dtype': dtype,
        'buffer_size': buffer_size
    }

    # Legacy regex-based parser kept for reference:
    # pattern = re.compile(r"(?P<array_dim>\d+)_(?P<algo_name>.+)_(?P<dtype>[^_]+)\.csv")
    # match = pattern.match(filename)
    # ...

def aggregate_results(results_dir: os.PathLike, metadata: pd.DataFrame, target_timestamp: str, system_name: str) -> pd.DataFrame:
    """
    Walk thorught restults_dir subdirectory and process each .csv file to aggregate results.
    Beware of metadata file.
    """
    all_results = []

    for test_id, meta_row in metadata.iterrows():
        test_dir = os.path.join(results_dir, str(test_id))
        if not os.path.isdir(test_dir):
            print(f"Test directory {test_dir} not found.", file=sys.stderr)
            continue

        for file in os.listdir(test_dir):
            if file.endswith(".csv"):
                filepath = os.path.join(test_dir, file)
                stats = process_benchmark_file(filepath)
                if stats is None:
                    continue
                file_params = parse_filename(file)

                result = {
                    'collective_type': _meta_value(meta_row, 'collective_type'),
                    'array_dim': file_params.get('array_dim'),
                    'buffer_size': file_params.get('buffer_size'),
                    'algo_name': file_params.get('algo_name'),
                    'datatype': file_params.get('dtype'),
                    'mean': stats['mean'],
                    'median': stats['median'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'nnodes': _meta_value(meta_row, 'nnodes'),
                    'tasks_per_node': _meta_value(meta_row, 'tasks_per_node'),
                    'n_iter': stats['n_iter'],
                    'percentile_10': stats['percentile_10'],
                    'percentile_25': stats['percentile_25'],
                    'percentile_75': stats['percentile_75'],
                    'percentile_90': stats['percentile_90'],
                    'iqr': stats['iqr'],
                    'standard_error': stats['standard_error'],
                    'ci_lower': stats['ci_lower'],
                    'ci_upper': stats['ci_upper'],
                    # 'outliers': stats['outliers'],
                    'num_outliers': stats['num_outliers'],
                    'system': system_name,
                    'timestamp': target_timestamp,
                    'test_id': str(test_id),
                    'MPI_Op': _meta_value(meta_row, 'MPI_Op'),
                    'gpu_awareness': _meta_value(meta_row, 'gpu_awareness'),
                    'gpu_lib': _meta_value(meta_row, 'gpu_lib'),
                    'gpu_lib_version': _meta_value(meta_row, 'gpu_lib_version'),
                    'notes': _meta_value(meta_row, 'notes'),
                    'mpi_lib': _meta_value(meta_row, 'mpi_lib'),
                    'mpi_lib_version': _meta_value(meta_row, 'mpi_lib_version'),
                    'libpico_version': _meta_value(meta_row, 'libpico_version'),
                }
                all_results.append(result)
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--result-dir", required=True, help="Full path to the test results")
    args = parser.parse_args()

    # Ensure the path follows the expected format: results/<system>/<timestamp>
    this_results_dir = os.path.normpath(args.result_dir)
    parts = this_results_dir.split(os.sep)
    if len(parts) < 3 or parts[-3] != "results":
        print(f"Invalid result directory structure: {this_results_dir}", file=sys.stderr)
        sys.exit(1)

    location, timestamp = parts[-2], parts[-1]
    tar_path = this_results_dir + ".tar.gz"

    # If result dir exists only as a tar.gz and not as a directory, extract it
    if not os.path.isdir(this_results_dir):
        if os.path.isfile(tar_path):
            print(f"Extracting {tar_path} to {os.path.dirname(this_results_dir)} for processing...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(this_results_dir))
        else:
            print(f"Directory {this_results_dir} does not exist and no tar.gz file found.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Directory {this_results_dir} exists. Using it for processing...")

    metadata_path = f"results/{location}_metadata.csv"
    if not os.path.isfile(metadata_path):
        print(f"Metadata file {metadata_path} not found.", file=sys.stderr)
        sys.exit(1)
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.set_index(["timestamp", "test_id"], inplace=True)
    test_metadata = metadata_df.loc[timestamp]

    # Aggregate all results by processing each benchmark file
    aggregated_df = aggregate_results(
        this_results_dir, test_metadata, timestamp, location)

    # Display the first few rows of the aggregated results
    print("Aggregated Data:")
    print(aggregated_df.head())

    # Optionally, save the aggregated summary to a CSV file
    aggregated_df.to_csv(
        f"{this_results_dir}/aggregated_results_summary.csv", index=False)


if __name__ == '__main__':
    main()
