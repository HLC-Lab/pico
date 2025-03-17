import os
import csv
import sys

RESULTS_DIR = "results/"

# Update or create metadata CSV
def update_metadata(system_name: str, timestamp: str, test_id: int, number_of_nodes: int,
                    tasks_per_node: int, collective_type: str, mpi_lib: str, mpi_lib_version : str,
                    libswing_version: str, cuda: str, output_level: str, mpi_op: str | None, notes: str | None):
    output_file = os.path.join(RESULTS_DIR, f"{system_name}_metadata.csv")

    # Check if file exists to determine whether to write the header
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "test_id",
            "nnodes",
            "tasks_per_node",
            "collective_type",
            "mpi_lib",
            "mpi_lib_version",
            "libswing_version",
            "CUDA",
            "output_level",
            "MPI_Op",
            "notes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the new metadata row
        writer.writerow({
            "timestamp": timestamp,
            "test_id": test_id,
            "nnodes": number_of_nodes,
            "tasks_per_node": tasks_per_node,
            "collective_type": collective_type,
            "mpi_lib": mpi_lib,
            "mpi_lib_version": mpi_lib_version,
            "libswing_version": libswing_version,
            "CUDA": cuda.lower(),
            "output_level": output_level,
            "MPI_Op": mpi_op,
            "notes": notes
        })

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {__file__} <test_id>", file=sys.stderr)
        sys.exit(1)

    test_id = sys.argv[1]
    if not test_id.isdigit():
        print(f"{__file__}: test_id must be an integer.", file=sys.stderr)
    test_id = int(test_id)

    system_name = os.getenv('LOCATION')
    number_of_nodes = os.getenv('N_NODES')
    tasks_per_node = os.getenv('TASK_PER_NODE')
    timestamp = os.getenv('TIMESTAMP')
    collective_type = os.getenv('COLLECTIVE_TYPE')
    mpi_lib = os.getenv('MPI_LIB')
    mpi_lib_version = os.getenv('MPI_LIB_VERSION')
    libswing_version = os.getenv('LIBSWING_VERSION')
    cuda = os.getenv('CUDA')
    output_level = os.getenv('OUTPUT_LEVEL')
    mpi_op = os.getenv('MPI_OP')
    notes = os.getenv('NOTES') or "null"
    if not (system_name and timestamp and number_of_nodes and number_of_nodes.isdigit() 
            and tasks_per_node and tasks_per_node.isdigit()and collective_type and mpi_lib
            and mpi_lib_version and libswing_version and cuda and output_level):
        print (f"{__file__}: Environment variables not set.", file=sys.stderr)
        print (f"LOCATION={system_name}\nTIMESTAMP={timestamp}\n"
               f"N_NODES={number_of_nodes}\nTASK_PER_NODE={tasks_per_node}\n"
               f"COLLECTIVE_TYPE={collective_type}\n"
               f"MPI_LIB={mpi_lib}\nMPI_LIB_VERSION={mpi_lib_version}\n"
               f"LIBSWING_VERSION={libswing_version}\nCUDA={cuda}", file=sys.stderr)
        sys.exit(1)

    number_of_nodes = int(number_of_nodes)
    tasks_per_node = int(tasks_per_node)

    update_metadata(system_name, timestamp, test_id, number_of_nodes, \
                    tasks_per_node, collective_type, mpi_lib, mpi_lib_version, \
                    libswing_version, cuda, output_level, mpi_op = mpi_op, notes = notes)
    print(f"Metadata updated for {system_name} at {timestamp}.")
