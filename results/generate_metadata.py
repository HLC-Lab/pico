import os
import csv
import sys
from typing import List

RESULTS_DIR = "results/"


# Update or create metadata CSV
def update_metadata(system_name: str, timestamp: str, number_of_nodes: int, datatypes : List[str],
                    collective_type: str, algos: List[str], mpi_lib: str, mpi_lib_version : str,
                     libswing_version: str, cuda: bool, mpi_op: str | None, notes: str | None):
    output_file = os.path.join(RESULTS_DIR, f"{system_name}_metadata.csv")

    # Check if file exists to determine whether to write the header
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "number_of_nodes",
            "collective_type",
            "algos",
            "datatypes",
            "mpi_lib",
            "mpi_lib_version",
            "libswing_version",
            "CUDA",
            "MPI_Op",
            "notes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the new metadata row
        writer.writerow({
            "timestamp": timestamp,
            "number_of_nodes": number_of_nodes,
            "collective_type": collective_type,
            "algos": " ".join(algos),
            "datatypes": " ".join(datatypes),
            "mpi_lib": mpi_lib,
            "mpi_lib_version": mpi_lib_version,
            "libswing_version": libswing_version,
            "CUDA": cuda,
            "MPI_Op": mpi_op,
            "notes": notes
        })

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python update_metadata.py <system_name> <timestamp> \
                <number_of_nodes>", file=sys.stderr)
        sys.exit(1)

    # Collect arguments from command line
    system_name = sys.argv[1]
    timestamp = sys.argv[2]
    number_of_nodes = int(sys.argv[3])

    collective_type = os.getenv('COLLECTIVE_TYPE')
    algos = os.getenv('ALGOS')
    datatypes = os.getenv('TYPES')
    mpi_lib = os.getenv('MPI_LIB')
    mpi_lib_version = os.getenv('MPI_LIB_VERSION')
    libswing_version = os.getenv('LIBSWING_VERSION')
    cuda = os.getenv('CUDA')
    mpi_op = os.getenv('MPI_OP')
    notes = os.getenv('NOTES')
    if not (collective_type and algos and datatypes and mpi_lib and mpi_lib_version and libswing_version and cuda):
        print ("Environment variables not set.", file=sys.stderr)
        sys.exit(1)

    algos = algos.split(" ")
    datatypes = datatypes.split(" ")
    cuda = cuda.lower() == "true"

    update_metadata(system_name, timestamp, number_of_nodes, datatypes, \
                    collective_type, algos, mpi_lib, mpi_lib_version, \
                    libswing_version, cuda, mpi_op = mpi_op, notes = notes)
    print(f"Metadata updated for {system_name} at {timestamp}.")
