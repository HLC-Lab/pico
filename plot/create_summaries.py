import os
import pandas as pd

# Define warmup rules based on array size
warmup_rules = {
    512: 2000,
    1048576: 200,
    8388608: 20,
    67108864: 2
}

def get_warmup_iterations(array_size : int) -> int:
    """
    Determine the number of warmup iterations based on the array size.

    Parameters:
        array_size: The size of the array.

    Returns:
        int: The number of warmup iterations.
    """
    for size, warmup in warmup_rules.items():
        if array_size <= size:
            return warmup
    return 0

def process_csv_file(filepath : str) -> dict | None:
    """
    Process a single CSV file to extract performance metrics.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        dict: A dictionary containing the processed results, or None if the file
              cannot be processed or contains no valid data.
    """
    filename = os.path.basename(filepath)
    filename, _ = os.path.splitext(filename)

    try:
        # Extract metadata from the filename
        nof_proc, array_size, dtype, alg_number = filename.split('_')[:4]
        nof_proc = int(nof_proc)
        array_size = int(array_size)
        alg_number = int(alg_number)
    except ValueError:
        print(f"Skipping file with unexpected filename format: {filename}")
        return None

    warmup_iterations = get_warmup_iterations(array_size)

    try:
        data = pd.read_csv(filepath, comment='#', header=None)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    # Skip warmup iterations
    data = data.iloc[warmup_iterations:]

    # Extract the first column and filter out invalid values
    highest_values = data.iloc[:, 0]
    valid_values = highest_values[highest_values > 0]

    if valid_values.empty:
        print(f"No valid data after filtering for {filename}")
        return None

    # Return summary statistics
    return {
        'nof_proc': nof_proc,
        'array_size': array_size,
        'alg_number': alg_number,
        'dtype': dtype,
        'mean': valid_values.mean(),
        'median': valid_values.median(),
        'std_dev': valid_values.std(),
        'count': len(valid_values)
    }

def aggregate_data(filepaths) -> pd.DataFrame:
    """
    Process multiple CSV files and aggregate the results.

    Parameters:
        filepaths (list): A list of file paths to process.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated results.
    """
    processed_data = []
    for filepath in filepaths:
        result = process_csv_file(filepath)
        if result is not None:
            processed_data.append(result)
    return pd.DataFrame(processed_data)

def list_csv_files(directory):
    """
    List all CSV files in a given directory.

    Parameters:
        directory (str): The directory to search for CSV files.

    Returns:
        list: A list of full paths to CSV files.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory: {directory}")
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

if __name__ == "__main__":
    root_folder = os.getcwd()

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            # Skip if summary already exists
            summary_path = os.path.join(subfolder_path, 'summary/')
            if os.path.isdir(summary_path):
                print(f"Summary of {subfolder_path} already made, skipping.")
                continue

            # Locate the 'data' folder
            data_folder_path = os.path.join(subfolder_path, 'data/')
            if not os.path.exists(data_folder_path):
                print(f"No 'data' folder found in {subfolder_path}, skipping.")
                continue

            # Process CSV files
            data_file_paths = list_csv_files(data_folder_path)
            df_results = aggregate_data(data_file_paths)

            # Create output directory and save results
            os.makedirs(summary_path, exist_ok=True)
            df_results.to_csv(os.path.join(summary_path, 'results_summary.csv'), index=False)
            df_results.to_json(os.path.join(summary_path, 'results_summary.json'), orient='records', lines=True)

            print(f"Processed {subfolder_path}, results saved in {summary_path}")
