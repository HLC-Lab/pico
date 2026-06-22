# twill.md — TWILL alltoall (Tapered-Weight Interleaved Linear aLLtoall)

## 1. Overview

`twill` is a family of external libpico **alltoall** algorithms designed for
*tapered* topologies (e.g. dragonfly with oversubscribed global links). A naive
linear alltoall ordering creates transient destination-group hotspots: at one
instant many senders target the same group, saturating that group's few global
links while others idle. Total cross-group volume is fixed; TWILL chooses a
destination *ordering* that keeps the instantaneous per-group load proportional
to each group's capacity — **with no assumption of uniform group sizes**, taking
only a rank→group map as input.

Three variants share one engine; they differ only in the relabeling `rho`:

| algorithm name (PICO)  | relabeling `rho`                         | role |
|------------------------|-------------------------------------------|------|
| `twill_group_over`     | smooth weighted round-robin over groups   | the contender |
| `twill_random_over`    | pseudorandom permutation (shared seed)    | zero-system-info baseline |
| `twill_shift_over`     | identity → classic `(s+t) mod P`          | isolates schedule vs engine effect |

All three work at **any** communicator size (no power-of-two constraint).

## 2. How it works (relabel-then-shift)

1. Build a relabeling `rho[pos] -> rank` (position → rank). For `twill_group`,
   `rho` is a smooth weighted round-robin (credit/stride scheduling) over groups
   with weight `w[j] = |group j|`, deterministic lowest-group-id tie-break. Any
   window of `k` consecutive positions then holds ≈ `k·w[j]/P` members of group
   `j` (prefix deviation `< 1`).
2. Linear shift in relabeled space. At step `t ∈ [0,P)`:
   * sender `s` sends to `pi(s,t)    = rho[(rho_inv[s] + t) mod P]`
   * receiver `r` receives from `sigma(r,t) = rho[(rho_inv[r] - t) mod P]`
   * `sigma` is the exact inverse of `pi` (`sigma(pi(s,t),t) == s`), so every
     step is a perfect matching (no rank-level incast) and receivers can place
     incoming data directly at `rbuf[sigma(r,t)]` with no staging copies.
   * `t = 0` is the self block (local copy, no MPI).

The schedule is only an ordering; a bounded window of nonblocking `Isend`/`Irecv`
consumes it with **no barriers and no per-step synchronization** (skew tolerance
is a design feature). The schedule core (the `twill_*` helpers) lives in
`libpico_utils.h` alongside the other shared helpers; the engine and the three
PICO entry points live in `libpico_alltoall.c` with the other alltoall algorithms.

## 3. Environment variables

Read once per context build (on first call / communicator-size change), exactly
like `SEGSIZE`. Open MPI / SLURM forward the launching environment to all ranks,
so exporting them before the run is enough.

| variable       | default            | meaning |
|----------------|--------------------|---------|
| `TWILL_WINDOW` | `32`               | outstanding-op window `W` (clamped `≥ 1`). Prediction: `twill_group`'s advantage grows as `W` shrinks; with `W ≥ P−1` all matching schedules converge. |
| `TWILL_SEED`   | fixed constant     | seed for `twill_random` (accepts `0x..` hex). All ranks must agree. |
| `TWILL_CACHE`  | `1`                | `1`: cache the schedule across iterations (steady state). `0`: rebuild/free `rho` every call so its construction cost is in every timed iteration. Map *discovery* is still done once either way. |
| `TWILL_MAP`    | unset              | path to a rank→group map file (see §4). Highest priority. |
| `TWILL_GROUP`  | `node`             | when no `TWILL_MAP`: an integer `N` = synthetic uniform groups of size `N` (rank `i` → group `i/N`, independent of placement), or `node` = node = group from shared-memory domains. |
| `TWILL_SKEW_US`| `0` (off)          | max microseconds of a per-rank, per-call random sleep injected before the exchange (skew-tolerance probing). Counted in total iter time, not in the `twill_exchange` tag. |

These knobs are documented in the JSON `desc` of each algorithm in
`config/algorithms/MPI/LibPico/alltoall.json` and flow through PICO's
environment configs the same way `SEGMENTED`/`SEGSIZE` do.

## 4. Group map input

Evaluated in priority order at context build:

1. **`TWILL_MAP=<path>`** — a text file with one integer per line; line `i` is
   the group id of rank `i`. Ids may be arbitrary / non-dense (libpico densifies
   by sorting unique ids). Every rank reads the file; it must have ≥ `P` entries.
2. **`TWILL_GROUP=<N>`** — synthetic uniform contiguous groups of size `N`:
   rank `i` → group `i/N`, so `⌈P/N⌉` groups (the last is ragged if `P` is not a
   multiple of `N`). Set manually, **independent of tasks-per-node**. Note this
   only changes the *schedule's* notion of groups; it yields a performance effect
   only where the physical topology actually tapers at that boundary.
3. **`TWILL_GROUP=node` (default)** — node = group, derived with
   `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` + an `MPI_Allgather` of the node
   leader's global rank. Correct for node-level taper, and a sane default for
   dragonfly groups *only if a real map is supplied*.
4. **single group** — fallback if the above are unavailable/fail (warned once on
   rank 0). `twill_group` then degenerates to a plain shift; `twill_random` and
   `twill_shift` never need the map.

A cross-rank **consistency guard** (`MPI_Bcast` of a 64-bit hash of
(kind, window, seed, map) + `MPI_Allreduce(MIN)` compare) runs once per context
build and aborts the collective with a clear message on mismatch.

### Generating a map: `utils/twill_gen_map.py`

Turns hostnames (from a PICO `alloc_<n>.csv`, a hostfile, or
`$SLURM_JOB_NODELIST` via `scontrol show hostnames`) into a `TWILL_MAP` file. A
`--regex` extracts a site-specific group field from each nodename; without it,
node = group (generic fallback).

```bash
# dragonfly group = digits after 'g' in nodenames like nid-g03-c1, 4 ranks/node
python3 utils/twill_gen_map.py --from-slurm --tasks-per-node 4 \
        --regex 'g(\d+)' --out twill.map
TWILL_MAP=twill.map mpirun ... pico_core <count> <iter> twill_group_over int32
```

## 5. Instrumentation (CPU-only)

When built with `-DPICO_INSTRUMENT` (and not NCCL / CUDA-aware), TWILL exposes
three tags (see [`instrument.md`](instrument.md)):

* `twill_setup` — context build (env read, map discovery, `rho` construction).
  On an instrumented run this separates one-time discovery from steady state.
* `twill_self` — the self-block local copy.
* `twill_exchange` — the windowed nonblocking progress loop.

## 6. Unit test

`utils/twill_schedule_test.c` is a standalone test of the schedule core
(bijectivity, `sigma∘pi = id`, per-step matching, group balance, ragged sizes).
It makes no MPI calls and runs as a single process (no mpirun); it is compiled
with mpicc only because the helpers live in `libpico_utils.h`, which pulls in
`mpi.h`. Run it:

```bash
make -C utils test-twill
```

## 7. v1 scope / limitations

* CPU / host MPI buffers. The self-block copy is a host `memcpy` (as in
  `alltoall_bine`); TWILL is **not** wired into the NCCL path and is not
  device-buffer correct under CUDA-aware MPI.
* Assumes `scount == rcount` with matched signatures (as PICO guarantees for
  alltoall). `scount`/`rcount` are guarded against `> INT_MAX`.
* `MPI_IN_PLACE` is rejected with `MPI_ERR_ARG` (as `pairwise_ompi` does).
