import json
import sys
import os

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def check_constraints(algo_constraints, test_constraints, comm_sz: int):
    """Check if the algorithm constraints match the test constraints."""
    for constraint in algo_constraints:
        key = constraint["key"]
        conditions = constraint["conditions"]

        if key == "comm_sz":
            test_value = comm_sz
        else:
            if key not in test_constraints:
                return False
            test_value = test_constraints[key]

        for condition in conditions:
            operator = condition["operator"]
            value = condition["value"]

            if operator == ">=":
                if not (test_value >= value):
                    return False
            elif operator == "<=":
                if not (test_value <= value):
                    return False
            elif operator == "==":
                if not (test_value == value):
                    return False
            elif operator == "is_power_of_two":
                if not (test_value & (test_value - 1) == 0):
                    return False
            elif operator == "is_even":
                if not (test_value % 2 == 0):
                    return False

    return True

def check_library_dependencies(algo_data, mpi_type, mpi_version, libswing_version):
    """Check if the algorithm satisfies the required library dependencies."""
    if "library" in algo_data:
        if mpi_type.lower() == "openmpi":
            if "ompi" not in algo_data["library"]:
                return False
            if not algo_data["library"]["ompi"].startswith(f">={mpi_version}"):
                return False
        elif mpi_type.lower() == "mpich":
            if "mpich" not in algo_data["library"]:
                return False
            if not algo_data["library"]["mpich"].startswith(f">={mpi_version}"):
                return False
        if "libswing" in algo_data["library"]:
            if not libswing_version or not algo_data["library"]["libswing"].startswith(f">={libswing_version}"):
                return False
    return True

def get_matching_algorithms(algorithm_config, test_config, comm_sz: int):
    """Get algorithms that match the test configuration."""
    collective = test_config["collective"]
    mpi_type = test_config["mpi"]["type"]
    mpi_version = test_config["mpi"]["version"]
    libswing_version = test_config.get("libswing_version", None)
    include_tags = test_config["tags"]["include"]
    exclude_tags = test_config["tags"]["exclude"]
    include_specific = test_config["specific"]["include"]
    exclude_specific = test_config["specific"]["exclude"]
    
    matching_algorithms = []
    skip_algorithms = []
    
    if collective in algorithm_config["collective"]:
        for algo_id, algo_data in algorithm_config["collective"][collective].items():
            if not check_library_dependencies(algo_data, mpi_type, mpi_version, libswing_version):
                continue

            # If in include_specific, add the algorithm anyway (library still checked)
            if include_specific and algo_id in include_specific:
                if "constraints" in algo_data:
                    if not check_constraints(algo_data["constraints"], {}, comm_sz):
                        skip_algorithms.append(algo_id)

                matching_algorithms.append({
                    "id": algo_id,
                    "name": algo_data["name"]
                })
                continue

            if exclude_specific and algo_id in exclude_specific:
                continue
            if not any(tag in algo_data["tags"] for tag in include_tags):
                continue
            if any(tag in algo_data["tags"] for tag in exclude_tags):
                continue

            # Check constraints
            if "constraints" in algo_data:
                if not check_constraints(algo_data["constraints"], {}, comm_sz):
                    skip_algorithms.append(algo_id)

            # If all checks pass, add the algorithm to the matching list
            matching_algorithms.append({
                "id": algo_id,
                "name": algo_data["name"]
            })
    
    return matching_algorithms, skip_algorithms


def export_environment_variables(matching_algorithms, skip_algorithms, test_config):
    """Export environment variables for the shell script."""
    collective = test_config["collective"]
    if "REDUCE" in collective:
        mpi_op = test_config["MPI_Op"]
    else:
        mpi_op = "null"
    mpi_type = test_config["mpi"]["type"]
    mpi_version = test_config["mpi"]["version"]
    libswing_version = test_config.get("libswing_version", "")
    cuda = test_config["cuda"]
    algo_ids = " ".join([algo["id"] for algo in matching_algorithms])
    algo_names = " ".join([algo["name"] for algo in matching_algorithms])
    skip_ids = " ".join(skip_algorithms)
    notes = test_config["notes"]

    # Write the environment variables to a shell script that will be sourced
    with open("scripts/select_test/env_vars.sh", "w") as f:
        f.write(f"export COLLECTIVE_TYPE='{collective}'\n")
        f.write(f"export ALGOS='{algo_ids}'\n")
        f.write(f"export NAMES='{algo_names}'\n")
        f.write(f"export SKIP='{skip_ids}'\n")
        f.write(f"export MPI_LIB='{mpi_type}'\n")
        f.write(f"export MPI_LIB_VERSION='{mpi_version}'\n")
        f.write(f"export LIBSWING_VERSION='{libswing_version}'\n")
        f.write(f"export CUDA='{cuda}'\n")
        f.write(f"export MPI_OP='{mpi_op}'\n")
        f.write(f"export NOTES='{notes}'\n")


def main():
    # Check if arguments are correctly provided
    if len(sys.argv) != 2:
        print("Usage: python parse_test.py <number_of_nodes>")
        sys.exit(1)
    if not ( os.path.isfile("scripts/algorithm_config.json") and os.path.isfile("scripts/select_test/test.json") ):
        print("Error: algorithm_config.json or test.json not found.")
        sys.exit(1)

    number_of_nodes = int(sys.argv[1])
    algorithm_config = load_json("scripts/algorithm_config.json")
    test_config = load_json("scripts/select_test/test.json")

    # Get matching algorithms
    matching_algorithms, skip_algorithms = get_matching_algorithms(algorithm_config, test_config, number_of_nodes)
    
    # Write environment variables to a shell script to be sourced
    export_environment_variables(matching_algorithms, skip_algorithms, test_config)
    

if __name__ == "__main__":
    main()
