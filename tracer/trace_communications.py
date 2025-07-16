# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

import json
import csv
import re
import sys
import os
import argparse
from math import log, ceil
from pprint import pprint

def load_communication_pattern(filename):
    with open(filename, 'r') as f:
        pattern = json.load(f)
    return pattern

def load_allocation(filename, location, hostname_only):
    """
    Reads a CSV file mapping MPI_Rank to hostname.
    Expected CSV header: MPI_Rank,allocation
    """
    allocation = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rank = int(row['MPI_Rank'])
            if location != "lumi" or hostname_only:
                hostname = row['allocation']
            else:
                hostname = row['xname']
            allocation[rank] = hostname
    return allocation

def lumi_node_to_cell(node_id):
    if node_id < 1000 or (node_id > 3047 and node_id < 5001) or node_id > 7977:
        print(f"{__file__}: Node ID {node_id} is out of range.", file=sys.stderr)
        sys.exit(1)
    if node_id <= 1255:
        return 1000
    elif node_id <= 1511:
        return 1001
    elif node_id <= 1767:
        return 1002
    elif node_id <= 2023:
        return 1003
    elif node_id <= 2279:
        return 1004
    elif node_id <= 2535:
        return 1005
    elif node_id <= 2791:
        return 1006
    elif node_id <= 3047:
        return 1007
    if node_id <= 5123:
        return 1100
    elif node_id <= 5247:
        return 1101
    elif node_id <= 5371:
        return 1102
    elif node_id <= 5495:
        return 1103
    elif node_id <= 5619:
        return 1104
    elif node_id <= 5743:
        return 1105
    elif node_id <= 5867:
        return 1200
    elif node_id <= 5991:
        return 1201
    elif node_id <= 6115:
        return 1202
    elif node_id <= 6239:
        return 1203
    elif node_id <= 6363:
        return 1204
    elif node_id <= 6487:
        return 1205
    elif node_id <= 6611:
        return 1300
    elif node_id <= 6735:
        return 1301
    elif node_id <= 6859:
        return 1302
    elif node_id <= 6983:
        return 1303
    elif node_id <= 7107:
        return 1304
    elif node_id <= 7231:
        return 1305
    elif node_id <= 7355:
        return 1400
    elif node_id <= 7479:
        return 1401
    elif node_id <= 7603:
        return 1402
    elif node_id <= 7727:
        return 1403
    elif node_id <= 7851:
        return 1404
    elif node_id <= 7977:
        return 1405
    else:
        print(f"{__file__}: Node ID {node_id} is out of range.", file=sys.stderr)
        sys.exit(1)


def map_rank_to_cell(allocation, node_to_cell, location, hostname_only):
    """
    Maps each MPI rank to a cell based on its hostname and the node-to-cell mapping.
    """
    patterns = {
        "leonardo": r'lrdn(\d+)',
        "mare_nostrum": r'as(\d+)'
    }
    if hostname_only:
        patterns["lumi"] = r'nid(\d+)'
    else:
        patterns["lumi"] = r'x(\d+)'

    if location not in patterns:
        print(f"{__file__}: Location '{location}' not supported.", file=sys.stderr)
        sys.exit(1)

    pattern = patterns[location]
    rank_to_cell = {}
    for rank, hostname in allocation.items():
        match = re.search(pattern, hostname)
        if match:
            node_id = int(match.group(1))
            if location == "lumi" and hostname_only:
                # Get the node ID from the hostname directly
                cell = lumi_node_to_cell(node_id)
            else:
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
    if location == "leonardo":
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
    else:
        return {}


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
    eval_globals = {"fi": fi, "ceil": ceil}

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

def tree_coll_lat(rank_to_cell, bine: bool, doubling: bool):
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
                if bine:
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

def create_recv_step_array(rank_to_cell, bine : bool, first_halving : bool):
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
            if bine:
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

def coll_bdw(rank_to_cell, bine : bool, first_halving: bool, reduce: bool):
    comm_sz = len(rank_to_cell)
    steps = int(log(comm_sz, 2))
    external_bytes, internal_bytes = 0, 0
    recv_step = create_recv_step_array(rank_to_cell, bine, first_halving)

    # Scatter Phase
    for step in range(steps):
        message_size = 1 / (2 ** (step + 1))
        for rank in range(comm_sz):
            if recv_step[rank] == step:
                if bine:
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

            if bine:
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

def scatter(rank_to_cell, bine : bool, doubling: bool):
    comm_sz = len(rank_to_cell)
    steps = int(log(comm_sz, 2))
    external_bytes, internal_bytes = 0, 0

    recvd = [0] * comm_sz
    recvd[0] = 1
    recvd2 = [0] * comm_sz
    recvd2[0] = 1

    for step in range(steps):
        recvd = recvd2.copy()
        message_size = 1 / (2 ** (step + 1))

        for rank in range(comm_sz):
            if recvd[rank] == 1:
                if bine:
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
    parser.add_argument("--coll", default="ALLREDUCE,ALLGATHER,ALLTOALL,BCAST,REDUCE,REDUCE_SCATTER,SCATTER", help="Collective operation to analyze (comma-separated)")
    parser.add_argument("--save", action='store_true', help="Save the results to a CSV file")
    parser.add_argument("--out", help="Output CSV file name")
    parser.add_argument("--hostname_only", action='store_true', help="If only hostname is present (i.e., for LUMI, no xname availiable)")
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
    print("Num Cells:", len(cell_to_rank))
    print("=" * 100)

def compute_extra_bytes_recursive_doubling(rank_to_cell):
    rank_to_cell_shrunk = rank_to_cell.copy()
    extra_internal = 0
    extra_external = 0
    # For recursive_doubling and bine_latency
    # extra_ranks are those greater than the largest power of 2 less than the number of ranks.
    largest_power_of_2 = 1 << (len(rank_to_cell).bit_length() - 1)
    extra_ranks = len(rank_to_cell) - largest_power_of_2
    # Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
    # sets new rank to -1.
    # Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
    # apply appropriate operation, and set new rank to rank/2
    for rank in range (0, 2*extra_ranks):
        if rank % 2 == 0:
            send_to = rank + 1

            if rank_to_cell.get(rank) != rank_to_cell.get(send_to):
                extra_external += 2 # We consider two (send at the beginning, and receive at the end)
            else:
                extra_internal += 2 # We consider two (send at the beginning, and receive at the end)
    
    # Renumber ranks in rank_to_cell so that the even ranks smaller than 2*extra_ranks are removed
    rank_to_cell_shrunk = {rank: cell for rank, cell in rank_to_cell.items() if rank >= 2 * extra_ranks or rank % 2 != 0}
    # Now renumber them so that they are contiguous
    rank_to_cell_shrunk = {new_rank: cell for new_rank, (old_rank, cell) in enumerate(rank_to_cell_shrunk.items())}
    return rank_to_cell_shrunk, extra_internal, extra_external

def compute_extra_bytes_rabenseifner(rank_to_cell):
    rank_to_cell_shrunk = rank_to_cell.copy()
    extra_internal = 0
    extra_external = 0
    # extra_ranks are those greater than the largest power of 2 less than the number of ranks.
    largest_power_of_2 = 1 << (len(rank_to_cell).bit_length() - 1)
    extra_ranks = len(rank_to_cell) - largest_power_of_2
    # Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
    # sets new rank to -1.
    # Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
    # apply appropriate operation, and set new rank to rank/2
    for rank in range (0, 2*extra_ranks):
        if rank % 2 != 0:
            send_to = rank + 1
            # Odd processes:
            # - Sendrecv of half buffer, and send of half buffer at the beginning
            # - Recv full buffer
            extra_bytes = (1/2 + 1/2)
        else:
            send_to = rank - 1
            # Even processes:
            # - Sendrecv of half buffer, and receive of half buffer at the beginning
            # - Send full buffer
            extra_bytes = (1/2 + 1)

        if rank_to_cell.get(rank) != rank_to_cell.get(send_to):
            extra_external += extra_bytes
        else:
            extra_internal += extra_bytes
    
    # Renumber ranks in rank_to_cell so that the odd ranks smaller than 2*extra_ranks are removed
    rank_to_cell_shrunk = {rank: cell for rank, cell in rank_to_cell.items() if rank >= 2 * extra_ranks or rank % 2 == 0}
    # Now renumber them so that they are contiguous
    rank_to_cell_shrunk = {new_rank: cell for new_rank, (old_rank, cell) in enumerate(rank_to_cell_shrunk.items())}
    return rank_to_cell_shrunk, extra_internal, extra_external

def compute_extra_bytes(rank_to_cell, algorithm):
    if algorithm == "recursive_doubling" or algorithm == "bine_latency":
        return compute_extra_bytes_recursive_doubling(rank_to_cell)
    elif algorithm == "rabenseifner":
        return compute_extra_bytes_rabenseifner(rank_to_cell)
    elif algorithm == "bine_bandwidth":
        if len(rank_to_cell) % 2 == 0:
            # For even ranks we do not need extra bytes
            return rank_to_cell, 0, 0
        else:
            return compute_extra_bytes_rabenseifner(rank_to_cell)

def main():
    args = parse_arguments()

    if not os.path.isfile(args.alloc):
        print(f"Allocation file not found: {args.alloc}", file=sys.stderr)
        return 1

    allocation = load_allocation(args.alloc, args.location, args.hostname_only)
    node_to_cell = load_topology(args.map, args.location)
    rank_to_cell = map_rank_to_cell(allocation, node_to_cell, args.location, args.hostname_only)
    rows = []

    collectives = args.coll.split(",")
    for coll in collectives:
        if (len(rank_to_cell) & (len(rank_to_cell) - 1) != 0) and coll != "ALLREDUCE":
            print(f"Number of ranks ({len(rank_to_cell)}) is not a power of 2.", file=sys.stderr)
            return 1
                
        if coll == "BCAST":
            count = {
                "binomial_doubling": tree_coll_lat(rank_to_cell, bine=False, doubling=True),
                "binomial_halving": tree_coll_lat(rank_to_cell, bine=False, doubling=False),
                "bine_doubling": tree_coll_lat(rank_to_cell, bine=True, doubling=True),
                "bine_halving": tree_coll_lat(rank_to_cell, bine=True, doubling=False),
                "bine_bdw_doubling_halving": coll_bdw(rank_to_cell, bine=True, first_halving=False, reduce=False),
                "bine_bdw_halving_doubling": coll_bdw(rank_to_cell, bine=True, first_halving=True, reduce=False),
                "binomial_bdw_doubling_halving": coll_bdw(rank_to_cell, bine=False, first_halving=False, reduce=False),
                "binomial_bdw_halving_doubling": coll_bdw(rank_to_cell, bine=False, first_halving=True, reduce=False)
            }
        elif coll == "REDUCE":
            count = {
                "binomial_doubling": tree_coll_lat(rank_to_cell, bine=False, doubling=True),
                "binomial_halving": tree_coll_lat(rank_to_cell, bine=False, doubling=False),
                "bine_doubling": tree_coll_lat(rank_to_cell, bine=True, doubling=True),
                "bine_halving": tree_coll_lat(rank_to_cell, bine=True, doubling=False),
                "bine_bdw_halving_doubling": coll_bdw(rank_to_cell, bine=True, first_halving=False, reduce=True),
                "bine_bdw_doubling_halving": coll_bdw(rank_to_cell, bine=True, first_halving=True, reduce=True),
                "binomial_bdw_halving_doubling": coll_bdw(rank_to_cell, bine=False, first_halving=False, reduce=True),
                "binomial_bdw_doubling_halving": coll_bdw(rank_to_cell, bine=False, first_halving=True, reduce=True)
            }
        elif coll == "SCATTER":
            count = {
                "binomial_doubling": scatter(rank_to_cell, bine=False, doubling=True),
                "binomial_halving": scatter(rank_to_cell, bine=False, doubling=False),
                "bine_doubling": scatter(rank_to_cell, bine=True, doubling=True),
                "bine_halving": scatter(rank_to_cell, bine=True, doubling=False)
            }
        elif coll == "ALLREDUCE":
            patterns = load_communication_pattern(args.comm).get(coll, {})
            # Loop over patterns and count bytes for each algorithm
            count = {}
            for algorithm, alg_data in patterns.items():
                extra_internal, extra_external = 0, 0
                if algorithm == "ring":
                    internal, external = count_inter_cell_bytes({algorithm : alg_data}, rank_to_cell)[algorithm]
                else:
                    if len(rank_to_cell) & (len(rank_to_cell) - 1) != 0:
                        rank_to_cell_shrunk, extra_internal, extra_external = compute_extra_bytes(rank_to_cell, algorithm)
                    else:
                        rank_to_cell_shrunk = rank_to_cell
                    internal, external = count_inter_cell_bytes({algorithm : alg_data}, rank_to_cell_shrunk)[algorithm]
                
                count[algorithm] = (internal + extra_internal, external + extra_external)    
        else:
            count = count_inter_cell_bytes(load_communication_pattern(args.comm).get(coll, {}), rank_to_cell)

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

    print_group_mapping(rank_to_cell)

    if args.save:
        save_to_csv(rows, len(rank_to_cell), args.alloc, args.out)


if __name__ == "__main__":
    main()
