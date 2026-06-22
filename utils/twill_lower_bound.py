# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

"""
twill_lower_bound.py — volume/bandwidth lower bound for an alltoall on a tapered
topology, from a TWILL_MAP (rank->group) file and per-group global-link bandwidth.

This is analysis-side only (nothing in the hot path). It reports how close a
measured alltoall time gets to the bytes-over-bandwidth floor.

Definitions (per-pair bytes b = scount * sizeof(dtype)):
  group sizes        w[j] = |group j|,  P = sum_j w[j]
  cross-group bytes  X = b * (P^2 - sum_j w[j]^2)        # all inter-group blocks
  egress of group j  E[j] = b * w[j] * (P - w[j])        # bytes leaving group j
  aggregate bound    T_lb_vol  = X / sum_j bw[j]          # if perfectly balanced
  bottleneck bound   T_lb_neck = max_j E[j] / bw[j]       # the slowest group

T_lb_neck is the honest bound when group sizes/bandwidths are uneven (the point
of TWILL); T_lb_vol matches the plan's `cross_group_bytes / aggregate_global_bw`.

Examples:
  # uniform 25 GB/s global egress per group, 4 MiB per pair
  python3 utils/twill_lower_bound.py --map twill.map \\
          --group-bw-gbps 25 --per-pair-bytes $((4*1024*1024))

  # per-group bandwidths + a measured time, to print the achieved fraction
  python3 utils/twill_lower_bound.py --map twill.map \\
          --group-bw-gbps 25,25,12.5 --per-pair-bytes 1048576 --measured-us 930
"""

import argparse
import sys
from typing import List


def read_group_sizes(path: str) -> List[int]:
    """Read a TWILL_MAP file (one group id per rank) into dense group sizes."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line))
    if not ids:
        sys.exit(f"error: no entries in map file {path}")
    uniq = sorted(set(ids))
    remap = {g: i for i, g in enumerate(uniq)}
    sizes = [0] * len(uniq)
    for g in ids:
        sizes[remap[g]] += 1
    return sizes


def parse_bw_gbps(spec: str, n_groups: int) -> List[float]:
    """Parse --group-bw-gbps: one value (broadcast) or a comma list of length G."""
    vals = [float(x) for x in spec.split(",") if x != ""]
    if len(vals) == 1:
        vals = vals * n_groups
    if len(vals) != n_groups:
        sys.exit(f"error: --group-bw-gbps has {len(vals)} values, expected 1 or {n_groups}")
    if any(v <= 0 for v in vals):
        sys.exit("error: --group-bw-gbps values must be positive")
    return [v * 1e9 for v in vals]  # GB/s -> bytes/s


def main() -> int:
    ap = argparse.ArgumentParser(description="Alltoall volume/bandwidth lower bound (TWILL).")
    ap.add_argument("--map", required=True, help="TWILL_MAP file (rank->group id)")
    ap.add_argument("--group-bw-gbps", required=True,
                    help="per-group global egress bandwidth in GB/s: one value or G comma-separated")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--per-pair-bytes", type=int, help="bytes each ordered pair exchanges")
    grp.add_argument("--per-pair-elems", type=int, help="elements per pair (needs --elem-bytes)")
    ap.add_argument("--elem-bytes", type=int, default=4, help="bytes per element (default 4)")
    ap.add_argument("--measured-us", type=float, default=None,
                    help="optional measured alltoall time (microseconds) to report the achieved fraction")
    args = ap.parse_args()

    sizes = read_group_sizes(args.map)
    G = len(sizes)
    P = sum(sizes)
    b = args.per_pair_bytes if args.per_pair_bytes is not None else args.per_pair_elems * args.elem_bytes
    if b <= 0:
        sys.exit("error: per-pair byte count must be positive")
    bw = parse_bw_gbps(args.group_bw_gbps, G)

    sum_sq = sum(w * w for w in sizes)
    cross_bytes = b * (P * P - sum_sq)
    egress = [b * w * (P - w) for w in sizes]
    agg_bw = sum(bw)

    t_lb_vol = cross_bytes / agg_bw                      # seconds
    neck_terms = [(egress[j] / bw[j], j) for j in range(G)]
    t_lb_neck, neck_j = max(neck_terms)

    print(f"ranks P={P}, groups G={G}, sizes min/max={min(sizes)}/{max(sizes)}")
    print(f"per-pair bytes b={b}")
    print(f"cross-group bytes X={cross_bytes:.3e}  (intra-group fraction "
          f"{sum_sq/(P*P):.3f})")
    print(f"aggregate global BW={agg_bw/1e9:.3f} GB/s")
    print(f"T_lb (volume / aggregate) = {t_lb_vol*1e6:.3f} us")
    print(f"T_lb (bottleneck group {neck_j}, size {sizes[neck_j]}, "
          f"{bw[neck_j]/1e9:.3f} GB/s) = {t_lb_neck*1e6:.3f} us")

    if args.measured_us is not None:
        m = args.measured_us / 1e6
        print(f"measured = {args.measured_us:.3f} us")
        print(f"  fraction of volume bound     : {t_lb_vol/m:.3f}  "
              f"(slowdown {m/t_lb_vol:.2f}x)")
        print(f"  fraction of bottleneck bound : {t_lb_neck/m:.3f}  "
              f"(slowdown {m/t_lb_neck:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
