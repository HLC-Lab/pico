# Implementation Plan: Tapered-Weight Interleaved Linear aLLtoall (TWILL) in PICO (`twill`)

Target: the PICO framework (https://github.com/HLC-Lab/pico). The algorithm is implemented as
libpico external alltoall algorithms, and all validation/benchmarking goes through PICO's existing
machinery (ground-truth correctness checks, orchestrator, CSV results, plotting). Clone/inspect the
repo first; file paths below refer to the repo root.

---

## Part A — Algorithm design (the actual contribution)

### A.1 Problem

On tapered topologies (dragonfly with oversubscribed global links), naive linear alltoall orderings
create transient destination-group hotspots: at any instant many senders target the same group,
saturating that group's few global links while others idle. Total cross-group volume is invariant;
the goal is a destination *ordering* that keeps instantaneous per-group load proportional to group
capacity, with **no assumption of uniform group sizes** and only a rank→group map as input.

### A.2 Relabel-then-shift schedule (core construction)

Properties required: (1) at every step the sender→destination map is a perfect matching (no
rank-level incast, and analytically invertible so receivers can pre-post receives from known
sources, zero staging copies); (2) at every step destinations are spread across groups proportional
to group weights, and each sender's destination-group sequence interleaves groups smoothly.

Construction:

1. **Group-interleaved relabeling ρ** (position → rank), built by smooth weighted round-robin
   (stride scheduling) over groups, weight `w[j] = |group j|`:

   ```
   credit[j] = 0; ptr[j] = 0                 # ptr: next member of group j (members sorted by rank)
   for pos in 0..P-1:
       credit[j] += w[j]  for all j
       jmax = argmax_j credit[j]             # deterministic tie-break: lowest group id
       rho[pos] = members[jmax][ptr[jmax]++]
       credit[jmax] -= P
   ```

   Smooth WRR guarantees any window of k consecutive positions contains ≈ `k*w[j]/P` members of
   group j (deviation ≤ 1). Also compute `rho_inv`.

2. **Linear shift in relabeled space.** Sender `s` at step `t ∈ [0,P)`:
   `pi(s,t) = rho[(rho_inv[s] + t) mod P]`; receiver `r` at step `t` receives from
   `sigma(r,t) = rho[(rho_inv[r] - t + P) mod P]`. Both O(1) after O(P) setup. Every step is a
   bijection; `sigma(pi(s,t),t) == s` must hold (unit-tested). Step `t=0` is the self block
   (local copy, no MPI).

3. **Variants via choice of ρ** (each registered as a separate PICO algorithm so the algorithm
   name lands in the results CSV):
   - `twill_group`: smooth-WRR ρ as above (the contender).
   - `twill_random`: ρ = pseudorandom permutation from a shared seed (all ranks compute identical
     ρ locally; deterministic PRNG, see pitfalls). Zero-system-information baseline that keeps the
     matching property.
   - `twill_shift`: ρ = identity → classic `(s+t) mod P`, but through the same windowed engine.
     Isolates "schedule effect" from "engine effect" when compared against the existing
     `pairwise_ompi_over` (which uses blocking `MPI_Sendrecv` per step).
   - Pathological baseline: existing PICO algorithms (`pairwise_ompi_over`, vendor `MPI_Alltoall`
     via the internal wrapper) serve as references; a deliberately-bad `twill_naive`
     (everyone sends to 0,1,2,... in order; receiver mirrors with the inverse ordering of the
     senders, i.e., receiver r expects sources in order 0,1,2,...) is optional but valuable to
     demonstrate the hotspot pathology. Implement it last.

### A.3 Windowed nonblocking engine (shared by all twill variants)

No barriers, no per-step synchronization — the schedule is only an ordering consumed by a bounded
window of outstanding nonblocking operations:

```
W = window size (env TWILL_WINDOW, default 32)
copy self block (respecting send/recv extents)
post min(W+8, P-1) MPI_Irecv from sigma(r, t) for t = 1,2,...   # recv window slightly ahead
post min(W,   P-1) MPI_Isend to   pi(s, t)   for t = 1,2,...
while sends or recvs remain:
    MPI_Waitsome over all outstanding requests
    for each completion, post the next send/recv in schedule order (if any remain)
```

Receive buffers are placed directly at `rbuf + sigma(r,t) * rcount * rext` (this is why analytic
invertibility matters). One fixed tag is sufficient (each ordered pair exchanges exactly one
message per alltoall); use a distinct tag constant to be safe. Single-threaded; no
`MPI_THREAD_MULTIPLE` requirement.

---

## Part B — PICO integration points (exact)

The PICO signature for alltoall algorithms is fixed by `ALLTOALL_MPI_ARGS` in `include/libpico.h`:

```c
const void *sbuf, size_t scount, MPI_Datatype sdtype,
void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
```

(`scount`/`rcount` are per-pair counts, standard MPI semantics — confirm against
`alltoall_wrapper` in `pico_core/pico_core_utils.h`, which forwards them to `MPI_Alltoall`.)

Files to touch — follow the existing `alltoall_bine` / `pairwise_ompi` pattern exactly:

1. **`libpico/libpico_twill_alltoall.c` (new file)** — or append to `libpico/libpico_alltoall.c`
   if the Makefile globs sources; check `libpico/Makefile` and prefer whichever requires the
   smaller diff. Contains:
   - the schedule code (ρ construction for all variants, `pi`/`sigma`),
   - the windowed engine,
   - a small lazily-initialized, cached context (see B.1),
   - entry points `int alltoall_twill_group(ALLTOALL_MPI_ARGS)`,
     `alltoall_twill_random(...)`, `alltoall_twill_shift(...)` — thin wrappers that select ρ
     and call the shared engine. Return `MPI_SUCCESS`/error codes and follow the
     `err_hndl`-style error reporting used by the existing implementations.
   - `PICO_TAG_BEGIN/END` instrumentation (see B.4).

2. **`include/libpico.h`** — declare the three functions next to the existing alltoall
   declarations:
   ```c
   int alltoall_twill_group(ALLTOALL_MPI_ARGS);
   int alltoall_twill_random(ALLTOALL_MPI_ARGS);
   int alltoall_twill_shift(ALLTOALL_MPI_ARGS);
   ```

3. **`pico_core/pico_core_utils.c`**, in `get_alltoall_function` (inside the `#ifndef PICO_NCCL`
   block), add:
   ```c
   CHECK_STR(algorithm, "twill_group_over",  alltoall_twill_group);
   CHECK_STR(algorithm, "twill_random_over", alltoall_twill_random);
   CHECK_STR(algorithm, "twill_shift_over",  alltoall_twill_shift);
   ```

4. **`config/algorithms/MPI/LibPico/alltoall.json`** — add one entry per variant, modeled on the
   existing `bine_over` entry but **without** the `is_power_of_two` constraint (working at any
   comm size is the point). Keep the `count >= comm_sz` constraint as in existing entries, set
   `"selection": "pico"`, `"tags": ["twill", "external", "tapered"]`, and a `desc` mentioning the
   relevant env vars. Match the JSON schema of the existing entries exactly (the TUI parses this).

5. **No changes to `pico_core/pico_core_alltoall.c`** (allocator already generic) and no changes
   to the test loop — `DEFINE_TEST_LOOP(alltoall, ...)` already calls through the function pointer.

### B.1 Context caching under the fixed signature

The PICO signature has no room for a context argument, so use a `static` cached context inside the
libpico file (this matches PICO's idiom of module-level state, cf. `bine_allreduce_segsize`):

- On each call, validate the cache against `(comm size, MPI_Comm handle)`. On mismatch (first call
  or different communicator size), rebuild: read env, load/derive the group map, build ρ/ρ_inv for
  the selected variant(s). Cache ρ per variant lazily.
- PICO runs many iterations per (algorithm, size) job; setup amortizes. Wrap the setup in
  `PICO_TAG_BEGIN("twill_setup") / PICO_TAG_END` so instrumented runs can separate it; optionally
  note in the README that iteration 0 includes setup when not instrumented.
- Free on `MPI_Finalize` is not available from here; a small atexit-free of host memory is fine
  (no MPI calls in it). Keep allocations modest: 2–3 int arrays of length P + request/status
  arrays of length ~2W.
- **`TWILL_CACHE` (default 1):** with `TWILL_CACHE=0`, the *computational* state (ρ, ρ⁻¹,
  request/status arrays) is rebuilt and freed on every call, so its per-call cost is included in
  every timed iteration. The map *discovery* (split_type/allgather, file read, consistency hash)
  is still performed only once and its raw result retained: rerunning collectives before every
  alltoall would implicitly synchronize the ranks (removing skew — a measurement artifact that
  flatters schedule-sensitive algorithms), and re-reading a map file every iteration is filesystem
  noise no deployment exhibits. This yields three cleanly separated numbers: steady state
  (`TWILL_CACHE=1`), per-call schedule-construction cost (`TWILL_CACHE=0`), and one-time
  discovery cost (the `twill_setup` instrumentation tag). Guard against double-free between the
  per-call free path and the atexit path.

### B.2 Group map input (env-driven, like SEGSIZE)

Priority order, evaluated at context build:

1. `TWILL_MAP=<path>`: text file, line `i` = group id of rank `i` (arbitrary, possibly non-dense
   ids; densify by sorting unique ids). Every rank reads the file. Validate `P` lines.
2. `TWILL_GROUP=node` (default when no map given): derive groups from shared-memory domains via
   `MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, ...)` + an `MPI_Allgather` of (node leader
   rank) to build rank→group on every rank. This makes node = group with zero external info —
   correct for node-level taper, and a sane default for dragonfly groups only if a map is supplied.
3. If both unavailable/fail → fall back to a single group (ρ degenerates appropriately for
   `twill_group`; warn once on rank 0). `twill_random`/`twill_shift` never need the map.

Consistency guard: `MPI_Allreduce` a 64-bit hash of (map, seed) with `MPI_BAND`/compare-to-bcast at
context build; abort with a clear message on mismatch. This runs once per context build, not per
iteration.

Other env knobs: `TWILL_WINDOW` (default 32), `TWILL_SEED` (default fixed constant; used by
`twill_random`), `TWILL_CACHE` (default 1; see B.1). These get exported through PICO's environment configs / test descriptions the
same way `SEGMENTED`/`SEGSIZE` are (see `config/environment/*` and how
`pico_core/pico_core_utils.c` reads `getenv("SEGSIZE")`). Document them in the JSON `desc` and in
a short `libpico/twill.md` note (mirror the style of `libpico/instrument.md`).

### B.3 Correctness via PICO

PICO's driver allocates `rbuf_gt` and validates custom collectives against the internal ground
truth automatically — so MPI-level correctness comes for free once registered. Run locally
(`config/environment/local`) with comm sizes {2, 3, 7, 8, 16} including non-powers-of-two
(our entries have no power-of-two constraint, unlike `bine_over`) and with a ragged `TWILL_MAP`
(e.g., groups of sizes [1,1,5] on 7 ranks).

Additionally add a tiny standalone, MPI-free unit test for the schedule math (bijectivity of every
step, `sigma∘pi = id`, per-step and per-prefix group balance for WRR ρ). Keep it self-contained:
`tests/` in PICO is for framework tests, so place it as `libpico/test_twill_schedule.c` with a
`make test-twill` target in `libpico/Makefile` (host compiler, no mpirun), or guard with
`#ifdef TWILL_UNIT_TEST` main(). Smallest-diff option wins; do not restructure PICO's build.

### B.4 Instrumentation

Per `libpico/instrument.md`: wrap context build in `twill_setup`, the self-copy in `twill_self`,
and the main progress loop in `twill_exchange`. CPU-only (instrumentation is disabled under
NCCL/CUDA-aware builds — twill is CPU/host-buffer MPI in v1; do not wire it into the NCCL path,
which PICO's alltoall doesn't support anyway).

---

## Part C — Evaluation through PICO

All benchmarking via the standard PICO flow (TUI or JSON test description →
`scripts/submit_wrapper.sh` → orchestrator → `results/<system>/<timestamp>/` CSVs → `plot/`).

Test matrix (one PICO test description, iterated by the orchestrator):
- Algorithms: `twill_group_over`, `twill_random_over`, `twill_shift_over`,
  `pairwise_ompi_over`, internal `MPI_Alltoall` (and vendor-tuned variants from the Open MPI /
  Cray algorithm configs where available on the target system).
- Message sizes: standard PICO sweep, 8 B → 4 MiB per pair.
- `TWILL_WINDOW` ∈ {2, 8, 32, 128}: encode as separate test runs (env var per run, like the
  SEGSIZE sweep pattern). Prediction: twill_group's advantage grows as W shrinks; with W ≥ P−1
  all matching schedules converge toward each other.
- Group maps: real system map (dragonfly group map on the target machine — e.g., derivable from
  SLURM topology/nodenames; provide `scripts`-style helper `gen_twill_map.py` that turns
  `scontrol show hostnames` + a site-specific regex into a `TWILL_MAP` file, with a generic
  fallback of node=group) and synthetic ragged maps for correctness runs.
- Skew sensitivity (stretch): PICO measures per-iteration; if no built-in skew injection exists,
  add `TWILL_SKEW_US` (pre-collective random sleep inside the twill entry points, outside the
  instrumented exchange tag, default 0). Keep it env-gated and off by default so it never
  perturbs normal runs.
- Cache A/B (one extra run at a fixed medium message size): repeat the twill variants with
  `TWILL_CACHE=0` to quantify per-call schedule-construction overhead, and report the
  `twill_setup` tag from an instrumented run as the one-time discovery cost. State in the
  writeup that headline numbers are steady-state on a reused communicator — the same regime in
  which vendor MPI amortizes its own per-communicator caching.

Headline metric: completion time vs message size per algorithm (PICO's standard plots), plus
time-vs-window at fixed medium size. Optional: report fraction of the volume/bandwidth lower bound
`T_lb = cross_group_bytes / aggregate_global_bw` in post-processing (a small script in `plot/scripts`
taking the map + taper bandwidth as input) — analysis-side only, nothing in the hot path.

Expected outcome to verify the theory from the design discussion: on tapered systems with
static-ish routing the gap (twill_group vs shift/pairwise) is large; on fabrics with fine-grained
adaptive routing + strong CC (e.g., Slingshot) the gap shrinks — both results are publishable
signal, so record system metadata (PICO already captures it).

---

## Milestones (Claude Code work units; each ends with a `NOTES.md` entry)

1. **M1 — Schedule core + unit test.** ρ construction (group/random/shift), `pi`/`sigma`,
   standalone unit test green (bijectivity, inverse, group balance, ragged sizes incl. [1], [1,1,37],
   P prime). No PICO wiring yet.
2. **M2 — Engine + PICO wiring.** Windowed engine, the three entry points, declarations,
   `CHECK_STR` registration, JSON entries. Local PICO run with correctness checks passing for
   sizes {2,3,7,8,16} × message sweep × the three variants, plus ragged `TWILL_MAP`.
3. **M3 — Hardening + docs.** Node-derived default map, consistency hash, env handling,
   `INT_MAX` guards (scount is `size_t`, `MPI_Isend` takes int), instrumentation tags,
   `libpico/twill.md`, `gen_twill_map.py`.
4. **M4 — Evaluation assets.** Test-description JSON for the sweep, window-sweep run configs,
   lower-bound post-processing script, (stretch) `TWILL_SKEW_US`.

## Pitfalls / guardrails

- **Fixed signature discipline:** no extra parameters; everything via env + cached static context.
  Rebuild the cache if the communicator size changes between calls; don't assume `MPI_COMM_WORLD`.
- **Determinism across ranks:** WRR tie-breaks by lowest group id; `twill_random` must use a
  self-implemented PRNG (splitmix64 + Fisher–Yates), never `rand()` — all ranks must compute
  byte-identical ρ from `TWILL_SEED`.
- **Receive correctness:** receives must use `sigma`, the exact inverse of `pi`; the unit test for
  `sigma(pi(s,t),t)==s` is a gate before any MPI run.
- **No barriers / no per-step sync** anywhere in the engine (skew tolerance is a design feature).
- **Datatypes:** use `MPI_Type_get_extent` for buffer arithmetic exactly as `pairwise_ompi` does;
  v1 assumes `scount == rcount` semantics consistent with MPI_Alltoall (the framework guarantees
  matched signatures). Offsets in `ptrdiff_t`, guard `scount <= INT_MAX`.
- **MPI_IN_PLACE:** reject with `MPI_ERR_ARG` like `pairwise_ompi` does.
- **Don't touch the NCCL path**; keep all additions inside `#ifndef PICO_NCCL`-compatible code
  (mirror how existing libpico alltoall code is built).
- **Smallest possible diff to PICO:** no build-system restructuring, no new dependencies; match
  existing code style, license header, and error-handling conventions of
  `libpico/libpico_alltoall.c`.
- Request bookkeeping in the engine: `Waitsome` returns slot indices, not steps — keep slot→state
  structs; the drain loop (when fewer than W steps remain) is where off-by-ones live.
