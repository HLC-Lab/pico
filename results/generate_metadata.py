import os
import csv
import sys
from typing import List, Optional

RESULTS_DIR = "results/"

# Update or create metadata CSV
def update_metadata(system_name: str, timestamp: str, number_of_nodes: int,
                     collective_type: str, algorithms_tested: List[int],
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
    algorithms_tested (List[int]): Algorithms tested during the run.
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
            "algorithms_tested",
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
            "algorithms_tested": ",".join(map(str, algorithms_tested)),
            "mpi_lib_type": mpi_lib_type,
            "mpi_lib_version": mpi_lib_version,
            "libswing_version": libswing_version,
            "cuda_aware": str(cuda_aware),
            "datatype": ",".join(datatype),
            "operator": operator,
            "other": other
        })

if __name__ == "__main__":
    if len(sys.argv) != 13:
        print("Usage: python update_metadata.py <system_name> <timestamp> \
                <number_of_nodes> <collective_type> <algorithms_tested> \
                <mpi_lib_type> <mpi_lib_version> <libswing_version> \
                <cuda_aware> <datatype> <operator> <other>")
        sys.exit(1)

    # Collect arguments from command line
    system_name = sys.argv[1]
    timestamp = sys.argv[2]
    number_of_nodes = int(sys.argv[3])
    collective_type = sys.argv[4]
    algorithms_tested = list(map(int, sys.argv[5].split(" ")))
    mpi_lib_type = sys.argv[6]
    mpi_lib_version = sys.argv[7]
    libswing_version = sys.argv[8]
    cuda_aware = sys.argv[9].lower() == "yes"
    datatype = sys.argv[10].split(" ")
    operator = sys.argv[11] if sys.argv[11].lower() != "none" else None
    other = sys.argv[12] if sys.argv[12].lower() != "none" else None

    update_metadata(system_name, timestamp, number_of_nodes, \
                    collective_type, algorithms_tested, mpi_lib_type, \
                    mpi_lib_version, libswing_version, cuda_aware, \
                    datatype, operator, other)
    print(f"Metadata updated for {system_name} at {timestamp}.")
