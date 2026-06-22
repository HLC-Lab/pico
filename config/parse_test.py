# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

import json
from packaging import version
from jsonschema.validators import validate
from jsonschema.exceptions import ValidationError
import sys
import os

require_cvars = [ "mpich", "cray_mpich"]

# JON schema for the test configuration file
test_config_schema = {
    "type": "object",
    "properties": {
        "libpico_version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "collective": {"type": "string", "enum": ["ALLREDUCE", "ALLTOALL", "ALLGATHER", "BCAST", "GATHER", "REDUCE", "REDUCE_SCATTER", "SCATTER"]},
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
        }
    },
    "required": ["libpico_version", "collective", "MPI_Op", "tags", "specific"],
    "additionalProperties": False
}



def load_json(file_path: str | os.PathLike):
    """Load the JSON test file and the algorithm config file"""
    with open(file_path, 'r') as file:
        return json.load(file)


def check_comm_sz(algo_constraints, comm_sz: int):
    """Check if the given algorithm satisfies the comm_sz related constraints"""
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



def check_library_dependencies(algo_data, mpi_type, mpi_version, libpico_version) -> bool:
    """
    Check if the algorithm's library dependencies are met.

    Returns:
        True if the dependencies are satisfied; False otherwise.
    """
    if "library" not in algo_data:
        print(f"Warning: No library data found for algorithm {algo_data.get('name', 'UNKNOWN')}.", file=sys.stderr)
        sys.exit(1)

    library_dependencies = algo_data["library"]

    if "libpico" in library_dependencies and version.parse(library_dependencies["libpico"]) <= version.parse(libpico_version):
        return True

    if mpi_type.lower() in library_dependencies and version.parse(library_dependencies[mpi_type.lower()]) <= version.parse(mpi_version):
        return True

    return False



def get_matching_algorithms(algorithm_config, test_config, comm_sz: int, mpi_type: str, mpi_version: str, cuda: str):
    """Get algorithms that match the test configuration."""
    collective = test_config["collective"]
    libpico_version = test_config["libpico_version"]
    include_tags = test_config["tags"]["include"]
    exclude_tags = test_config["tags"]["exclude"]
    include_specific = test_config["specific"]["include"]
    exclude_specific = test_config["specific"]["exclude"]
    cuda_bool = cuda.lower() == "yes"

    matching_algorithms = []
    skip_algorithms = []
    is_segmented = []
    cvars = []

    if collective not in algorithm_config["collective"]:
        print(f"{__file__}: collective {collective} not found in ALGORITHM_CONFIG_FILE.", file=sys.stderr)
        sys.exit(1)

    for algo_name, algo_data in algorithm_config["collective"][collective].items():
        # Check if the algorithm satisfies the library dependencies and comm_sz constraints.
        if not check_library_dependencies(algo_data, mpi_type, mpi_version, libpico_version):
            continue
        if "constraints" in algo_data and not check_comm_sz(algo_data["constraints"], comm_sz):
            continue

        if cuda_bool and "cuda" not in algo_data["tags"]:
            continue

        # Handle specific inclusions: add these algorithms regardless of tags.
        if algo_name in include_specific:
            if "constraints" in algo_data and check_skip(algo_data["constraints"]):
                skip_algorithms.append(algo_name)
            matching_algorithms.append(algo_name)
            continue

        # Exclude if algorithm is in the specific exclusion list regardless of tags.
        if exclude_specific and algo_name in exclude_specific:
            continue

        # Tag based filtering
        if not any(tag in algo_data["tags"] for tag in include_tags):
            print(f"Excluding {algo_name} due to tag not found.")
            continue
        if any(tag in algo_data["tags"] for tag in exclude_tags):
            print(f"Excluding {algo_name} due to tag exclusion.")
            continue

        # Add to matching (and skip if skip-constraints are not met)
        if "constraints" in algo_data and check_skip(algo_data["constraints"]):
            skip_algorithms.append(algo_name)
        if mpi_type.lower() in require_cvars:
            cvars.append(algo_data["cvar"])

        matching_algorithms.append(algo_name)

        if "is_segmented" in algo_data["tags"]:
            is_segmented.append("yes")
        else:
            is_segmented.append("no")

    if not matching_algorithms:
        print(f"{__file__}: no allowed algorithms found for TEST_CONFIG_FILE.", file=sys.stderr)
        sys.exit(1)

    if mpi_type.lower() in require_cvars and not cvars:
        print(f"{__file__}: no cvars found for MPI_LIB={mpi_type}.", file=sys.stderr)
        sys.exit(1)

    return matching_algorithms, skip_algorithms, is_segmented, cvars



def export_environment_variables(matching_algorithms, skip_algorithms, is_segmented, cvars,
                                 test_config, output_file: str | os.PathLike) -> None:
    """Export environment variables for the shell script."""
    collective = test_config["collective"]
    if "REDUCE" in collective:
        mpi_op = test_config["MPI_Op"]
    else:
        mpi_op = "null"
    libpico_version = test_config.get("libpico_version", "")
    algo_names = " ".join(matching_algorithms)
    skip_names = " ".join(skip_algorithms)
    segmented = " ".join(is_segmented)
    cvars_str = " ".join(cvars) if cvars else ""

    # Write the environment variables to a shell script that will be sourced
    try:
        with open(output_file, "w") as f:
            f.write(f"export COLLECTIVE_TYPE='{collective}'\n")
            f.write(f"export ALGOS='{algo_names}'\n")
            f.write(f"export SKIP='{skip_names}'\n")
            f.write(f"export LIBPICO_VERSION='{libpico_version}'\n")
            f.write(f"export MPI_OP='{mpi_op}'\n")
            f.write(f"export IS_SEGMENTED=({segmented})\n")
            if cvars_str:
                f.write(f"export CVARS=({cvars_str})\n")
    except IOError as e:
        print(f"{__file__}: Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    algorithm_file = os.getenv("ALGORITHM_CONFIG_FILE")
    test_file = os.getenv("TEST_CONFIG")
    output_file = os.getenv("TEST_ENV")
    number_of_nodes = os.getenv("MPI_TASKS")
    mpi_type = os.getenv("MPI_LIB")
    mpi_version = os.getenv("MPI_LIB_VERSION")
    # TODO: Modify cuda variable and method 
    cuda = os.getenv("GPU_AWARENESS")
    if not (algorithm_file and test_file and output_file and number_of_nodes and
            number_of_nodes.isdigit() and mpi_type and mpi_version and cuda):
        print(f"\n{__file__}: Environment variables not set.\n", file=sys.stderr)
        print(f"ALGORITHM_CONFIG_FILE={algorithm_file}\nTEST_CONFIG={test_file}"
              f"\nTEST_ENV={output_file}\nN_NODES={number_of_nodes}\n"
              f"MPI_LIB={mpi_type}, MPI_LIB_VERSION={mpi_version}\n"
              f"CUDA={cuda}\n", file=sys.stderr)
        sys.exit(1)
    number_of_nodes = int(number_of_nodes)

    if not (os.path.isfile(algorithm_file) and os.path.isfile(test_file)):
        print(f"{__file__}: {algorithm_file} or {test_file} not found.", file=sys.stderr)
        sys.exit(1)

    algorithm_config = load_json(algorithm_file)
    test_config = load_json(test_file)

    # Validate the test_config.json
    try:
        validate(instance=test_config, schema=test_config_schema)
    except ValidationError as e:
        print(f"{__file__}Validation error: {e}", file=sys.stderr)
        sys.exit(1)


    # Get matching algorithms
    matching_algorithms, skip_algorithms, is_segmented, cvars = get_matching_algorithms(
        algorithm_config, test_config, number_of_nodes, mpi_type, mpi_version, cuda)

    # Write environment variables to a shell script to be sourced
    export_environment_variables(matching_algorithms, skip_algorithms, is_segmented, cvars, test_config, output_file)
    

if __name__ == "__main__":
    main()
