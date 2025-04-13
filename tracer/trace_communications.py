import json
import csv
import re
import sys
import os
import argparse
from math import log
from pprint import pprint

def load_communication_pattern(filename):
    with open(filename, 'r') as f:
        pattern = json.load(f)
    return pattern

def load_allocation(filename, location):
    """
    Reads a CSV file mapping MPI_Rank to hostname.
    Expected CSV header: MPI_Rank,allocation
    """
    allocation = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rank = int(row['MPI_Rank'])
            if location != "lumi":
                hostname = row['allocation']
            else:
                hostname = row['xname']
            allocation[rank] = hostname
    return allocation

def map_rank_to_cell(allocation, node_to_cell, location):
    """
    Maps each MPI rank to a cell based on its hostname and the node-to-cell mapping.
    """
    patterns = {
        "leonardo": r'lrdn(\d+)',
        "lumi": r'x(\d+)',
        "mare_nostrum": r'as(\d+)'
    }
    if location not in patterns:
        print(f"{__file__}: Location '{location}' not supported.", file=sys.stderr)
        sys.exit(1)

    pattern = patterns[location]
    rank_to_cell = {}
    for rank, hostname in allocation.items():
        match = re.search(pattern, hostname)
        if match:
            node_id = int(match.group(1))
            cell = node_to_cell.get(node_id) if location == "leonardo" else node_id
            rank_to_cell[rank] = cell
        else:
            print(f"{__file__}: Node ID not found for rank {rank} and hostname {hostname}\n", file=sys.stderr)
            sys.exit(1)

    return rank_to_cell


def load_topology(filename, location):
    """
    Reads a topology map file and returns a mapping from node id to cell id.
    Expected format in each line: "NODE 0001 RACK 1 CELL 1 ROW 1 ...".
    """
    if location != "leonardo":
        return {}

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


def apply_substitutions(s, subs):
    for key, value in subs.items():
        s = s.replace(str(key), str(value))
    return s


rhos = [1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525]

def fi(rank, step, num_ranks)-> int:
    if rank % 2 == 0:
        return (rank + rhos[step]) % num_ranks
    else:
        return (rank - rhos[step] + num_ranks) % num_ranks


def count_inter_cell_bytes(comm_pattern, rank_to_cell):
    """
    Iterates over the communication pattern and sums the bytes for communications 
    that cross cell boundaries using precompiled expressions and a globals dict
    with function fi exposed to the eval environment.
    """
    final_count = {}
    num_ranks = len(rank_to_cell)

    # Globals to expose to the eval expressions.
    eval_globals = {"fi": fi}

    # Iterate over each algorithm defined under ALLREDUCE.
    for algorithm, alg_data in comm_pattern.items():
        external_bytes = 0
        internal_bytes = 0
        parameters = alg_data.get("parameters", {})

        try:
            num_ranks_sym   = parameters["num_ranks"]
            rank_sym        = parameters["rank"]
            step_sym        = parameters["step"]
            num_steps_sym   = parameters["num_steps"]
            buffer_size_sym = parameters["buffer_size"]
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        phases = alg_data.get("phases", [])
        for phase in phases:
            steps_expr        = phase.get("steps")
            send_to_expr      = phase.get("send_to")
            message_size_expr = phase.get("message_size")

            # Evaluate steps expression once (substituting num_ranks_sym)
            steps_eval_expr = steps_expr.replace(num_ranks_sym, str(num_ranks))
            steps = int(eval(steps_eval_expr))

            # Precompile expressions for message_size and send_to
            message_size_code = compile(message_size_expr, "<string>", "eval")
            send_to_code      = compile(send_to_expr, "<string>", "eval")

            for step in range(steps):
                # Build base substitutions that change per step.
                base_subs = {
                    buffer_size_sym: 1,
                    step_sym: step,
                    num_ranks_sym: num_ranks,
                    num_steps_sym: steps
                }
                # Evaluate message_size once per step using precompiled code
                message_size = eval(message_size_code, eval_globals, base_subs)
                for rank in range(num_ranks):
                    subs = dict(base_subs)
                    subs[rank_sym] = rank
                    send_to = int(eval(send_to_code, eval_globals, subs))

                    if rank_to_cell.get(rank) != rank_to_cell.get(send_to):
                        external_bytes += message_size
                    else:
                        internal_bytes += message_size

        final_count[algorithm] = (internal_bytes, external_bytes)

    return final_count

def tree_coll_lat(rank_to_cell, swing: bool, doubling: bool):
    comm_sz = len(rank_to_cell)
    steps = int(log(comm_sz, 2))
    external_bytes, internal_bytes = 0, 0
    recvd = [0] * comm_sz
    recvd[0] = 1
    recvd2 = [0] * comm_sz
    recvd2[0] = 1

    for step in range(steps):
        recvd = recvd2.copy()
        for rank in range(comm_sz):
            if recvd[rank] == 1:
                if swing:
                    if doubling:
                        send_to = fi(rank, step, comm_sz)
                    else:
                        send_to = fi(rank, steps - step - 1, comm_sz)
                else:
                    if doubling:
                        send_to = rank ^ (1 << step)
                    else:
                        send_to = rank ^ (1 << (steps - step - 1))

                if rank_to_cell.get(rank) != rank_to_cell.get(send_to):
                    external_bytes += 1
                else:
                    internal_bytes += 1

                recvd2[send_to] = 1

    return internal_bytes, external_bytes

def create_recv_step_array(rank_to_cell, swing : bool, first_halving : bool):
    comm_sz = len(rank_to_cell)
    steps = int(log(comm_sz, 2))
    recvd = [0] * comm_sz
    recvd[0] = 1
    recvd2 = [0] * comm_sz
    recvd2[0] = 1

    recv_step = [steps - 1] * comm_sz
    recv_step[0] = 0

    recv_step_aux = [steps - 1] * comm_sz
    recv_step_aux[0] = 0

    for step in range(steps):
        recv_step = recv_step_aux.copy()
        recvd = recvd2.copy()
        for rank in range(comm_sz):
            if recvd[rank] != 1:
                continue
            if swing:
                if first_halving:
                    send_to = fi(rank, steps - step - 1, comm_sz)
                else:
                    send_to = fi(rank, step, comm_sz)
            else:
                if first_halving:
                    send_to = int(rank) ^ (1 << (steps - step - 1))
                else:
                    send_to = int(rank) ^ (1 << step)

            recvd2[send_to] = 1
            recv_step_aux[send_to] = step
    recv_step[0] = -1

    return recv_step

def coll_bdw(rank_to_cell, swing : bool, first_halving: bool, reduce: bool):
    comm_sz = len(rank_to_cell)
    steps = int(log(comm_sz, 2))
    external_bytes, internal_bytes = 0, 0
    recv_step = create_recv_step_array(rank_to_cell, swing, first_halving)

    # Scatter Phase
    for step in range(steps):
        message_size = 1 / (2 ** (step + 1))
        for rank in range(comm_sz):
            if recv_step[rank] == step:
                if swing:
                    if first_halving:
                        recv_from = fi(rank, steps - step - 1, comm_sz)
                    else:
                        recv_from = fi(rank, step, comm_sz)
                else:
                    if first_halving:
                        recv_from = int(rank) ^ (1 << (steps - step - 1))
                    else:
                        recv_from = int(rank) ^ (1 << step)

                if rank_to_cell.get(rank) != rank_to_cell.get(recv_from):
                    external_bytes += message_size
                else:
                    internal_bytes += message_size

    # Gather Phase
    for step in range(steps):
        message_size = 1 / ( 2 ** (steps - step))
        for rank in range(comm_sz):
            if recv_step[rank] == steps - step - 1 and not reduce:
                continue

            if swing:
                if first_halving:
                    send_to = fi(rank, step, comm_sz)
                else:
                    send_to = fi(rank, steps - step - 1, comm_sz)
            else:
                if first_halving:
                    send_to = int(rank) ^ (1 << step)
                else:
                    send_to = int(rank) ^ (1 << (steps - step - 1))

            if rank_to_cell.get(rank) != rank_to_cell.get(send_to):
                external_bytes += message_size
            else:
                internal_bytes += message_size

    return internal_bytes, external_bytes

def tree_coll(rank_to_cell, swing : bool, doubling: bool, gather: bool = False):
    comm_sz = len(rank_to_cell)
    steps = int(log(comm_sz, 2))
    external_bytes, internal_bytes = 0, 0

    # NOTE: For gather:
    #       - recvd is logically sent
    #       - send_to is logically recv_from
    #       (the actual code is the same, but the meaning is different)
    recvd = [0] * comm_sz
    recvd[0] = 1
    recvd2 = [0] * comm_sz
    recvd2[0] = 1

    for step in range(steps):
        recvd = recvd2.copy()

        if gather == True:
            message_size = 1 / (2 ** (steps - step))
        else: # scatter
            message_size = 1 / (2 ** (step + 1))

        for rank in range(comm_sz):
            if recvd[rank] == 1:
                if swing:
                    if doubling:
                        send_to = fi(rank, step, comm_sz)
                    else:
                        send_to = fi(rank, steps - step - 1, comm_sz)
                else:
                    if doubling:
                        send_to = int(rank) ^ (1 << step)
                    else:
                        send_to = int(rank) ^ (1 << (steps - step - 1))

                if rank_to_cell.get(rank) != rank_to_cell.get(send_to):
                    external_bytes += message_size
                else:
                    internal_bytes += message_size
                recvd2[send_to] = 1

    return internal_bytes, external_bytes

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze inter-cell communication in collective operations."
    )
    parser.add_argument("--location", required=True, help="Location of the system (e.g., 'leonardo', 'lumi')")
    parser.add_argument("--alloc", required=True, help="Path to the allocation CSV file")
    parser.add_argument("--map", default='tracer/maps/leonardo.txt', help="Path to the topology map file")
    parser.add_argument("--comm", default='tracer/algo_patterns.json', help="Path to the instantiated communication pattern JSON file")
    parser.add_argument("--coll", default="ALLREDUCE,ALLGATHER,BCAST,GATHER,REDUCE,REDUCE_SCATTER,SCATTER", help="Collective operation to analyze (comma-separated)")
    parser.add_argument("--save", action='store_true', help="Save the results to a CSV file")
    parser.add_argument("--out", help="Output CSV file name")
    return parser.parse_args()


def save_to_csv(rows, num_ranks: int, alloc_file: str, out_file: str = "") -> None:
    """Save the analysis results to a CSV file."""
    if out_file:
        output_file = out_file if out_file.endswith(".csv") else out_file + ".csv"
    else:
        output_file = f"{os.path.dirname(alloc_file)}/traced_{os.path.basename(alloc_file)}"
    try:
        with open(output_file, "w", newline='') as csvfile:
            fieldnames = ["COLLECTIVE", "ALGORITHM", "INTERNAL", "EXTERNAL", "TOTAL"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"COLLECTIVE": "num_ranks", "ALGORITHM": num_ranks, "INTERNAL": "", "EXTERNAL": "", "TOTAL": ""})

        print(f"Results saved to {output_file}")
    except IOError as e:
        print(f"Failed to write CSV file: {e}", file=sys.stderr)


def print_group_mapping(rank_to_cell) -> None:
    """
        Print the mapping of MPI ranks to cells.
        The mapping is displayed in a formatted table.
    """
    num_of_digits = len(str(len(rank_to_cell)))
    cell_to_rank = {}

    for key, value in rank_to_cell.items():
        cell_to_rank.setdefault(value, []).append(key)
    print("=" * 100)
    print("GROUP MAPPING")
    print("-" * 100)
    print("Cell ID -> [MPI_Ranks]")
    print("-" * 100)
    for cell, ranks in cell_to_rank.items():
        formatted_ranks = " ".join(f"{rank:>{num_of_digits}}" for rank in ranks)
        print(f"Cell {cell:>2} -> [{formatted_ranks}]")
    print("=" * 100)


def main():
    args = parse_arguments()

    if not os.path.isfile(args.alloc):
        print(f"Allocation file not found: {args.alloc}", file=sys.stderr)
        return 1

    allocation = load_allocation(args.alloc, args.location)
    node_to_cell = load_topology(args.map, args.location)
    rank_to_cell = map_rank_to_cell(allocation, node_to_cell, args.location)

    if len(rank_to_cell) & (len(rank_to_cell) - 1) != 0:
        print(f"Number of ranks ({len(rank_to_cell)}) is not a power of 2.", file=sys.stderr)
        return 1

    rows = []

    collectives = args.coll.split(",")
    for coll in collectives:
        if coll == "BCAST":
            count = {
                "binomial_doubling": tree_coll_lat(rank_to_cell, swing=False, doubling=True),
                "binomial_halving": tree_coll_lat(rank_to_cell, swing=False, doubling=False),
                "swing_doubling": tree_coll_lat(rank_to_cell, swing=True, doubling=True),
                "swing_halving": tree_coll_lat(rank_to_cell, swing=True, doubling=False),
                "swing_bdw_doubling_halving": coll_bdw(rank_to_cell, swing=True, first_halving=False, reduce=False),
                "swing_bdw_halving_doubling": coll_bdw(rank_to_cell, swing=True, first_halving=True, reduce=False),
                "binomial_bdw_doubling_halving": coll_bdw(rank_to_cell, swing=False, first_halving=False, reduce=False),
                "binomial_bdw_halving_doubling": coll_bdw(rank_to_cell, swing=False, first_halving=True, reduce=False)
            }
        elif coll == "REDUCE":
            count = {
                "binomial_doubling": tree_coll_lat(rank_to_cell, swing=False, doubling=True),
                "binomial_halving": tree_coll_lat(rank_to_cell, swing=False, doubling=False),
                "swing_doubling": tree_coll_lat(rank_to_cell, swing=True, doubling=True),
                "swing_halving": tree_coll_lat(rank_to_cell, swing=True, doubling=False),
                "swing_bdw_halving_doubling": coll_bdw(rank_to_cell, swing=True, first_halving=False, reduce=True),
                "swing_bdw_doubling_halving": coll_bdw(rank_to_cell, swing=True, first_halving=True, reduce=True),
                "binomial_bdw_halving_doubling": coll_bdw(rank_to_cell, swing=False, first_halving=False, reduce=True),
                "binomial_bdw_doubling_halving": coll_bdw(rank_to_cell, swing=False, first_halving=True, reduce=True)
            }
        elif coll == "SCATTER":
            count = {
                "binomial_doubling": tree_coll(rank_to_cell, swing=False, doubling=True, gather=False),
                "binomial_halving": tree_coll(rank_to_cell, swing=False, doubling=False, gather=False),
                "swing_doubling": tree_coll(rank_to_cell, swing=True, doubling=True, gather=False),
                "swing_halving": tree_coll(rank_to_cell, swing=True, doubling=False, gather=False)
            }
        elif coll == "GATHER":
            count = {
                "binomial_doubling": tree_coll(rank_to_cell, swing=False, doubling=True, gather=True),
                "binomial_halving": tree_coll(rank_to_cell, swing=False, doubling=False, gather=True),
                "swing_doubling": tree_coll(rank_to_cell, swing=True, doubling=True, gather=True),
                "swing_halving": tree_coll(rank_to_cell, swing=True, doubling=False, gather=True)
            }
        else:
            coll_comm_pattern = load_communication_pattern(args.comm).get(coll, {})
            count = count_inter_cell_bytes(coll_comm_pattern, rank_to_cell)

        print("\n\n" + "=" * 40)
        print(f"\t{coll.lower()}")
        print("=" * 40)

        for algorithm, (internal, external) in count.items():
            total = internal + external
            print(f"{'Algorithm:':<20}{algorithm}")
            print(f"{'Internal bytes:':<20}{internal} n bytes")
            print(f"{'External bytes:':<20}{external} n bytes")
            print(f"{'Total bytes:':<20}{total} n bytes")
            print("-" * 40)  # Separator between entries
            if args.save:
                rows.append({
                    "COLLECTIVE": coll.lower(),
                    "ALGORITHM": algorithm,
                    "INTERNAL": internal,
                    "EXTERNAL": external,
                    "TOTAL": total
                })

    print("\n`n` denotes the size of the send buffer\n\n")

    print_group_mapping(rank_to_cell);

    if args.save:
        save_to_csv(rows, len(rank_to_cell), args.alloc, args.out)


if __name__ == "__main__":
    main()
