import os
import csv
import sys
from typing import List, Optional

RESULTS_DIR = "results/"

# Update or create metadata CSV
def update_metadata(system_name: str, timestamp: str, number_of_nodes: int,
                     collective_type: str, algo_numbers: List[int], algo_names: List[str],
                     mpi_lib_type: str, mpi_lib_version : str,
                     libswing_version: str, cuda_aware: bool, datatype: List[str],
                     operator: Optional[str] = None, other: Optional[str] = None):
    """
    Updates or creates a CSV file to store metadata for test results.

    Parameters:
    system_name (str): The name of the system under test.
    timestamp (str): The timestamp of when the test was run.
    number_of_nodes (int): Number of nodes used in the test.
    collective_type (str): Type of collective operation used.
    algo_numbers (List[int]): Algorithms tested during the run.
    algo_names (List[str]): Names of the algorithms tested.
    mpi_lib_type (str): The MPI library type (Open MPI, MPICH, Cray MPI...).
    mpi_lib_version (str): Version of the MPI library used.
    libswing_version (str): Version of the libswing library used.
    cuda_aware (bool): Whether CUDA-aware support is enabled.
    datatype (List[str]): Data type(s) involved in the test.
    operator (Optional[str]): Operator used, if applicable.
    other (Optional[str]): Additional information about the test.

    Returns:
    None
    """
    output_file = os.path.join(RESULTS_DIR, f"{system_name}_metadata.csv")

    # Check if file exists to determine whether to write the header
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "number_of_nodes",
            "collective_type",
            "algo_numbers",
            "algo_names",
            "mpi_lib_type",
            "mpi_lib_version",
            "libswing_version",
            "cuda_aware",
            "datatype",
            "operator",
            "other"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the new metadata row
        writer.writerow({
            "timestamp": timestamp,
            "number_of_nodes": number_of_nodes,
            "collective_type": collective_type,
            "algo_numbers": ",".join(map(str, algo_numbers)),
            "algo_names": ",".join(algo_names),
            "mpi_lib_type": mpi_lib_type,
            "mpi_lib_version": mpi_lib_version,
            "libswing_version": libswing_version,
            "cuda_aware": str(cuda_aware),
            "datatype": ",".join(datatype),
            "operator": operator,
            "other": other
        })

if __name__ == "__main__":
    if len(sys.argv) != 14:
        print("Usage: python update_metadata.py <system_name> <timestamp> \
                <number_of_nodes> <collective_type> <algo_numbers> <algo_names> \
                <mpi_lib_type> <mpi_lib_version> <libswing_version> \
                <cuda_aware> <datatype> <operator> <other>")
        sys.exit(1)

    # Collect arguments from command line
    system_name = sys.argv[1]
    timestamp = sys.argv[2]
    number_of_nodes = int(sys.argv[3])
    collective_type = sys.argv[4]
    algo_numbers = list(map(int, sys.argv[5].split(" ")))
    algo_names = sys.argv[6].split(" ")
    mpi_lib_type = sys.argv[7]
    mpi_lib_version = sys.argv[8]
    libswing_version = sys.argv[9]
    cuda_aware = sys.argv[10].lower() == "yes"
    datatype = sys.argv[11].split(" ")
    operator = sys.argv[12] if sys.argv[12].lower() != "none" else None
    other = sys.argv[13] if sys.argv[13].lower() != "none" else None

    update_metadata(system_name, timestamp, number_of_nodes, \
                    collective_type, algo_numbers, algo_names, mpi_lib_type, \
                    mpi_lib_version, libswing_version, cuda_aware, \
                    datatype, operator, other)
    print(f"Metadata updated for {system_name} at {timestamp}.")
