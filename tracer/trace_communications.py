import json
import csv
import re
import sys
import argparse
from math import log
from sympy import sympify, symbols

def load_communication_pattern(filename):
    with open(filename, 'r') as f:
        pattern = json.load(f)
    return pattern

def load_allocation(filename):
    """
    Reads a CSV file mapping MPI_Rank to hostname.
    Expected CSV header: MPI_Rank,allocation
    """
    allocation = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rank = int(row['MPI_Rank'])
            hostname = row['allocation']
            allocation[rank] = hostname
    return allocation

def map_rank_to_cell(allocation, node_to_cell):
    """
    Maps each MPI rank to a cell based on its hostname and the node-to-cell mapping.
    """
    rank_to_cell = {}
    for rank, hostname in allocation.items():
        node_id = re.search(r'lrdn(\d+)',hostname)
        if node_id is not None:
            node_id = int(node_id.group(1))
            cell = node_to_cell.get(node_id)
            rank_to_cell[rank] = cell
        else:
            print(f"{__file__}:Node ID not found for rank {rank} and hostname {hostname}", file=sys.stderr)

    return rank_to_cell

def load_topology(filename):
    """
    Reads a topology map file and returns a mapping from node id to cell id.
    Expected format in each line: "NODE 0001 RACK 1 CELL 1 ROW 1 ...".
    """
    node_to_cell = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if "CELL" in parts and "NODE" in parts:
                try:
                    node_index = parts.index("NODE")
                    node_id = int(parts[node_index + 1])
                    cell_index = parts.index("CELL")
                    cell_id = int(parts[cell_index + 1])
                    node_to_cell[node_id] = cell_id
                except (ValueError, IndexError):
                    continue
    return node_to_cell

def preprocess_expression(expr_str):
    """
    Preprocess the expression string:
      - Replace caret (^) with exponentiation (**)
      - Replace textual operators (e.g. 'xor', 'mod') with valid Python operators.
    """
    expr_str = expr_str.replace("^", "**")
    expr_str = expr_str.replace("mod", "%")
    expr_str = expr_str.replace("xor", "^")

    return expr_str

def apply_substitutions(s, subs):
    for key, value in subs.items():
        s = s.replace(str(key), str(value))
    return s


def rho(step) -> int:
    return (1 - ((-2)**(step+1))) // 3

def fi(rank, step, num_ranks)-> int:
    if rank % 2 == 0:
        return (rank + rho(step)) % num_ranks
    else:
        return (rank - rho(step) + num_ranks) % num_ranks

def count_inter_cell_bytes(comm_pattern, rank_to_cell):
    """
    Iterates over the communication pattern and sums the bytes for communications 
    that cross cell boundaries.
    
    Assumes that the communication pattern JSON has been instantiated with concrete values.
    """
    final_count = {}
    num_ranks = len(rank_to_cell)

    # Iterate over each algorithm defined under ALLREDUCE.
    for algorithm, alg_data in comm_pattern.items():
        external_bytes = 0
        internal_bytes = 0
        parameters = alg_data.get("parameters", {})

        try:
            num_ranks_sym = parameters["num_ranks"]
            rank_sym      = parameters["rank"]
            step_sym      = parameters["step"]
            buffer_size_sym = parameters["buffer_size"]
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        phases = alg_data.get("phases", [])
        for phase in phases:
            steps_expr        = preprocess_expression(phase.get("steps"))
            send_to_expr      = preprocess_expression(phase.get("send_to"))
            recv_from_expr    = preprocess_expression(phase.get("recv_from"))
            message_size_expr = preprocess_expression(phase.get("message_size"))

            steps = int(eval(steps_expr.replace(num_ranks_sym, str(num_ranks))))
            for step in range(steps):
                substitutions = {
                    buffer_size_sym: 1,
                    step_sym: step,
                    num_ranks_sym: num_ranks
                }

                message_size = eval(apply_substitutions(message_size_expr, substitutions))
                for rank in range(num_ranks):
                    substitutions[rank_sym] = rank
                    send_to = int(eval(apply_substitutions(send_to_expr, substitutions)))
                    recv_from = int(eval(apply_substitutions(recv_from_expr, substitutions)))

                    if rank_to_cell.get(rank) != rank_to_cell.get(send_to):
                        external_bytes += message_size
                    else:
                        internal_bytes += message_size

                    if rank_to_cell.get(rank) != rank_to_cell.get(recv_from):
                        external_bytes += message_size
                    else:
                        internal_bytes += message_size


        final_count[algorithm] = (internal_bytes/2, external_bytes/2)
    
    return final_count

def main():
    parser = argparse.ArgumentParser(description="Analyze inter-cell communication in collective operations.")
    parser.add_argument("--map", default='tracer/maps/leonardo.txt', help="Path to the topology map file")
    parser.add_argument("--comm", default='tracer/algo_patterns.json', help="Path to the instantiated communication pattern JSON file")
    parser.add_argument("--coll", default="ALLREDUCE", help="Collective operation to analyze")
    parser.add_argument("--alloc", required=True, help="Path to the allocation CSV file")
    args = parser.parse_args()

    allocation = load_allocation(args.alloc)
    node_to_cell = load_topology(args.map)
    rank_to_cell = map_rank_to_cell(allocation, node_to_cell)
    comm_pattern = load_communication_pattern(args.comm).get(args.coll, {})

    count = count_inter_cell_bytes(comm_pattern, rank_to_cell)

    print("-" * 40)
    print(f"\t\t{args.coll.lower()}")
    print("-" * 40)

    for algorithm, (internal, external) in count.items():
        total = internal + external
        print(f"{'Algorithm:':<20}{algorithm}")
        print(f"{'Internal bytes:':<20}{internal} n bytes")
        print(f"{'External bytes:':<20}{external} n bytes")
        print(f"{'Total bytes:':<20}{total} n bytes")
        print("-" * 40)  # Separator between entries

    print("\n`n` denotes the size of the send buffer")

if __name__ == "__main__":
    main()
