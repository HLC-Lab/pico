import os
import sys
import json

# Find the dynamic_rule for the given collective type and algorithm
def find_dynamic_rule(algorithm_config_file: str | os.PathLike, collective_type: str, algorithm: str) -> int:
    """ Find the dynamic rule for the given collective type and algorithm
        Args:
            algorithm_config_file (str | os.PathLike): The path to the JSON file
            collective_type (str): The collective type
            algorithm (str): The algorithm
        Returns:
            int: The dynamic rule
    """
    # Load the JSON file
    with open(algorithm_config_file, 'r') as json_file:
        algorithm_config = json.load(json_file)

    if collective_type not in algorithm_config["collective"]:
        print (f"{__file__}: collective type {collective_type} not found in the JSON file.", file=sys.stderr)
        sys.exit(1)

    dynamic_rule = -1
    for algo_name, algo_data in algorithm_config["collective"][collective_type].items():
        if algo_name == algorithm:
            dynamic_rule = algo_data["dynamic_rule"]
            break

    if dynamic_rule == -1:
        print (f"{__file__}: algorithm {algorithm} not found for collective type {collective_type}.", file=sys.stderr)
        sys.exit(1)
    
    return dynamic_rule


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

    for i, line in enumerate(lines):
        if collective_type in line:
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
    algorithm_config_file = os.getenv('ALGORITHM_CONFIG_FILE')
    dynamic_rule_file = os.getenv('DYNAMIC_RULE_FILE')
    collective_type = os.getenv('COLLECTIVE_TYPE')
    if not (algorithm_config_file and dynamic_rule_file and collective_type):
        print(f"{__file__}: Environment variables not set.", file=sys.stderr)
        print(f"ALGORITHM_CONFIG_FILE={algorithm_config_file}\nDYNAMIC_RULE_FILE={dynamic_rule_file}\nCOLLECTIVE_TYPE={collective_type}", file=sys.stderr)
        sys.exit(1)

    new_rule = find_dynamic_rule(algorithm_config_file, collective_type, algorithm)
    modify_dynamic_rule(dynamic_rule_file, collective_type, new_rule)

if __name__ == "__main__":
    main()
