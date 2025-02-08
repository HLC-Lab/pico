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
    system_name = os.getenv('LOCATION')
    number_of_nodes = os.getenv('N_NODES')
    timestamp = os.getenv('TIMESTAMP')
    collective_type = os.getenv('COLLECTIVE_TYPE')
    algos = os.getenv('ALGOS')
    datatypes = os.getenv('TYPES')
    mpi_lib = os.getenv('MPI_LIB')
    mpi_lib_version = os.getenv('MPI_LIB_VERSION')
    libswing_version = os.getenv('LIBSWING_VERSION')
    cuda = os.getenv('CUDA')
    mpi_op = os.getenv('MPI_OP')
    notes = os.getenv('NOTES')
    if not (system_name and timestamp and number_of_nodes and number_of_nodes.isdigit() and collective_type and algos and datatypes and mpi_lib and mpi_lib_version and libswing_version and cuda):
        print (f"{__file__}: Environment variables not set.", file=sys.stderr)
        print (f"LOCATION={system_name}\nTIMESTAMP={timestamp}\nN_NODES={number_of_nodes}\nCOLLECTIVE_TYPE={collective_type}\nALGOS={algos}\nTYPES={datatypes}\nMPI_LIB={mpi_lib}\nMPI_LIB_VERSION={mpi_lib_version}\nLIBSWING_VERSION={libswing_version}\nCUDA={cuda}", file=sys.stderr)
        sys.exit(1)

    number_of_nodes = int(number_of_nodes)
    algos = algos.split(" ")
    datatypes = datatypes.split(" ")
    cuda = cuda.lower() == "true"

    update_metadata(system_name, timestamp, number_of_nodes, datatypes, \
                    collective_type, algos, mpi_lib, mpi_lib_version, \
                    libswing_version, cuda, mpi_op = mpi_op, notes = notes)
    print(f"Metadata updated for {system_name} at {timestamp}.")
