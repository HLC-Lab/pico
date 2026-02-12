# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

import os
import re
import sys
import json

def resolve_algorithm_declarations_dirs() -> list[str]:
    env_dir = os.getenv("ALGORITHM_DECLARATIONS_DIR")
    if env_dir:
        return [env_dir]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(repo_root, "config", "algorithms", "MPI", "Open-MPI"),
        os.path.join(repo_root, "config", "algorithms", "MPI", "LibPico"),
        os.path.join(repo_root, "config", "algorithms", "Open-MPI"),
        os.path.join(repo_root, "config", "algorithms", "LibPico"),
    ]
    dirs = [path for path in candidates if os.path.isdir(path)]
    if dirs:
        return dirs

    print(f"{__file__}: algorithm declarations directory not found.", file=sys.stderr)
    sys.exit(1)


# Find the dynamic_rule for the given collective type and algorithm
def find_dynamic_rule(algorithm_decls_dirs: list[str], collective_type: str, algorithm: str) -> int:
    """ Find the dynamic rule for the given collective type and algorithm
        Args:
            algorithm_decls_dir (str | os.PathLike): Base directory for per-collective JSON files
            collective_type (str): The collective type
            algorithm (str): The algorithm
        Returns:
            int: The dynamic rule
    """
    collective_file = f"{collective_type.lower()}.json"
    searched_files = []

    for algorithm_decls_dir in algorithm_decls_dirs:
        algorithm_file = os.path.join(algorithm_decls_dir, collective_file)
        if not os.path.isfile(algorithm_file):
            continue
        searched_files.append(algorithm_file)
        with open(algorithm_file, 'r') as json_file:
            algorithm_config = json.load(json_file)
        if algorithm not in algorithm_config:
            continue
        algo_data = algorithm_config[algorithm]
        if "selection" not in algo_data:
            print(f"{__file__}: algorithm {algorithm} missing selection in {algorithm_file}.", file=sys.stderr)
            sys.exit(1)
        return algo_data["selection"]

    if searched_files:
        searched = "\n".join(searched_files)
        print(f"{__file__}: algorithm {algorithm} not found for collective type {collective_type}.", file=sys.stderr)
        print(f"Searched:\n{searched}", file=sys.stderr)
        sys.exit(1)

    print(f"{__file__}: no declaration files found for collective type {collective_type}.", file=sys.stderr)
    sys.exit(1)


# Modify the .txt fil
def modify_dynamic_rule(rule_file: str | os.PathLike, collective_type: str, new_rule: int) -> None:
    """ Modify the dynamic rule in the .txt file 
        Args:
            rule_file (str | os.PathLike): The path to the .txt file
            collective_type (str): The collective type
            new_rule (int): The new dynamic rule
    """
    with open(rule_file, 'r') as txt_file:
        lines = txt_file.readlines()

    pattern = r'\b' + re.escape(collective_type) + r'\b'
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            if i + 4 < len(lines):  # Ensure we don't go out of bounds
                lines[i+4] = f"0 {new_rule} 0 0 # Algorithm\n"
                with open(rule_file, 'w') as txt_file:
                    txt_file.writelines(lines)
                return
            else:
                print(f"{__file__}: Insufficient lines in the file after '{collective_type}'.", file=sys.stderr)
                sys.exit(1)

    print (f"{__file__}: Collective type {collective_type} not found in the .txt file.", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print(f"{__file__} Usage: python change_dynamic_rules.py <algorithm>", file=sys.stderr)
        sys.exit(1)
    algorithm = sys.argv[1]
    mpi_lib = (os.getenv("MPI_LIB") or "").upper()
    if mpi_lib in {"MPICH", "CRAY_MPICH", "NCCL"}:
        print(f"{__file__}: MPI_LIB={mpi_lib} should not use change_dynamic_rules.py.", file=sys.stderr)
        sys.exit(1)

    algorithm_decls_dirs = resolve_algorithm_declarations_dirs()
    dynamic_rule_file = os.getenv('DYNAMIC_RULE_FILE')
    collective_type = os.getenv('COLLECTIVE_TYPE')
    if not (dynamic_rule_file and collective_type):
        print(f"{__file__}: Environment variables not set.", file=sys.stderr)
        print(f"DYNAMIC_RULE_FILE={dynamic_rule_file}\nCOLLECTIVE_TYPE={collective_type}\nALGORITHM_DECLARATIONS_DIR={os.getenv('ALGORITHM_DECLARATIONS_DIR')}", file=sys.stderr)
        sys.exit(1)

    new_rule = find_dynamic_rule(algorithm_decls_dirs, collective_type, algorithm)
    modify_dynamic_rule(dynamic_rule_file, collective_type, new_rule)

if __name__ == "__main__":
    main()
