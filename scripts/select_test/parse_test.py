import json
import jsonschema
import sys
import os

# JON schema for the test configuration file
test_config_schema = {
    "type": "object",
    "properties": {
        "mpi": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["ompi", "ompi_swing"]},
                "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"}
            },
            "required": ["type", "version"],
            "additionalProperties": False
        },
        "libswing_version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "collective": {"type": "string", "enum": ["ALLREDUCE", "REDUCE_SCATTER", "ALLGATHER"]},
        "MPI_Op": {"type": "string"},
        "tags": {
            "type": "object",
            "properties": {
                "include": {"type": "array", "items": {"type": "string"}},
                "exclude": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["include", "exclude"],
            "additionalProperties": False
        },
        "specific": {
            "type": "object",
            "properties": {
                "include": {"type": "array", "items": {"type": "string"}},
                "exclude": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["include", "exclude"],
            "additionalProperties": False
        },
        "cuda": {"type": "boolean"},
        "notes": {"type": "string"}
    },
    "required": ["mpi", "libswing_version", "collective", "tags", "specific", "cuda", "notes"],
    "additionalProperties": False
}

def load_json(file_path):
    """Load the JSON test file and the algorithm
    config file"""
    with open(file_path, 'r') as file:
        return json.load(file)

def check_comm_sz(algo_constraints, comm_sz: int):
    """Check if the given algorithm satisfies the
    comm_sz related constraints"""
    for constraint in algo_constraints:
        if constraint["key"] != "comm_sz":
            continue

        test_value = comm_sz
        for condition in constraint["conditions"]:
            operator = condition["operator"]
            value = condition["value"]

            if operator == ">=" and not (test_value >= value):
                return False
            elif operator == "<=" and not (test_value <= value):
                return False
            elif operator == "==" and not (test_value == value):
                return False
            elif operator == "is_power_of_two" and not (test_value > 0 and (test_value & (test_value - 1)) == 0):
                return False
            elif operator == "is_even" and not (test_value % 2 == 0):
                return False

    return True


def check_skip(algo_constraints) -> bool:
    """Check if the algorithm should be added to the SKIP list."""
    for constraint in algo_constraints:
        if constraint["key"] != "count":
            continue

        for condition in constraint["conditions"]:
            operator = condition["operator"]
            value = condition["value"]
            if operator == ">=" and value == "comm_sz":
                return True

    return False


def check_library_dependencies(algo_data, mpi_type, mpi_version, libswing_version) -> bool:
    """
    Check if the algorithm's library dependencies are met.

    Returns:
        True if the dependencies are satisfied; False otherwise.
    """
    if "library" not in algo_data:
        print(f"Warning: No library data found for algorithm {algo_data.get('name', 'UNKNOWN')}.", file=sys.stderr)
        sys.exit(1)

    library_dependencies = algo_data["library"]

    if "libswing" in library_dependencies and library_dependencies["libswing"] <= libswing_version:
        return True

    if str(mpi_type) in library_dependencies and library_dependencies[str(mpi_type)] <= mpi_version:
        return True

    return False


def get_matching_algorithms(algorithm_config, test_config, comm_sz: int):
    """Get algorithms that match the test configuration."""
    collective = test_config["collective"]
    mpi_type = test_config["mpi"]["type"]
    mpi_version = test_config["mpi"]["version"]
    libswing_version = test_config["libswing_version"]
    include_tags = test_config["tags"]["include"]
    exclude_tags = test_config["tags"]["exclude"]
    include_specific = test_config["specific"]["include"]
    exclude_specific = test_config["specific"]["exclude"]
    
    matching_algorithms = []
    skip_algorithms = []
    
    if collective not in algorithm_config["collective"]:
        return matching_algorithms, skip_algorithms

    for algo_id, algo_data in algorithm_config["collective"][collective].items():
        # Check if the algorithm satisfies the library dependencies and comm_sz constraints.
        if not check_library_dependencies(algo_data, mpi_type, mpi_version, libswing_version):
            continue
        if "constraints" in algo_data and not check_comm_sz(algo_data["constraints"], comm_sz):
            continue

        # Handle specific inclusions: add these algorithms regardless of tags.
        if algo_id in include_specific:
            if "constraints" in algo_data and check_skip(algo_data["constraints"]):
                skip_algorithms.append(algo_id)
            matching_algorithms.append({ "id": algo_id, "name": algo_data["name"] })
            continue

        # Exclude if algorithm is in the specific exclusion list regardless of tags.
        if exclude_specific and algo_id in exclude_specific:
            continue

        # Tag based filtering
        if not any(tag in algo_data["tags"] for tag in include_tags):
            continue
        if any(tag in algo_data["tags"] for tag in exclude_tags):
            continue

        # Add to matching (and skip if skip-constraints are not met)
        if "constraints" in algo_data and check_skip(algo_data["constraints"]):
            skip_algorithms.append(algo_id)
        matching_algorithms.append({ "id": algo_id, "name": algo_data["name"] })

    if not matching_algorithms:
        print("No matching algorithms found for the given test configuration.", file=sys.stderr)
        sys.exit(1)

    return matching_algorithms, skip_algorithms


def export_environment_variables(matching_algorithms, skip_algorithms, test_config):
    """Export environment variables for the shell script."""
    collective = test_config["collective"]
    if "REDUCE" in collective:
        mpi_op = test_config["MPI_Op"]
    else:
        mpi_op = "null"
    mpi_type = test_config["mpi"]["type"].upper()
    mpi_version = test_config["mpi"]["version"]
    libswing_version = test_config.get("libswing_version", "")
    cuda = test_config["cuda"]
    algo_ids = " ".join([algo["id"] for algo in matching_algorithms])
    algo_names = " ".join([algo["name"] for algo in matching_algorithms])
    skip_ids = " ".join(skip_algorithms)
    notes = test_config["notes"]

    # Write the environment variables to a shell script that will be sourced
    with open("scripts/select_test/test_env_vars.sh", "w") as f:
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
    if not ( os.path.isfile("scripts/algorithm_config.json") and os.path.isfile("scripts/select_test/test_config.json") ):
        print("Error: algorithm_config.json or test_config.json not found.")
        sys.exit(1)

    number_of_nodes = int(sys.argv[1])
    algorithm_config = load_json("scripts/algorithm_config.json")
    test_config = load_json("scripts/select_test/test_config.json")

    # Validate the test_config.json
    try:
        jsonschema.validate(instance=test_config, schema=test_config_schema)
    except jsonschema.exceptions.ValidationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get matching algorithms
    matching_algorithms, skip_algorithms = get_matching_algorithms(algorithm_config, test_config, number_of_nodes)
    
    # Write environment variables to a shell script to be sourced
    export_environment_variables(matching_algorithms, skip_algorithms, test_config)
    

if __name__ == "__main__":
    main()
