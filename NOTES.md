# TWILL implementation notes

Tapered-Weight Interleaved Linear aLLtoall (TWILL) for PICO. See
`twill_alltoall_pico_plan.md` for the full design.

## M1 — Schedule core + unit test (done)

- Schedule core (the `twill_*` helpers) — state-free, no MPI calls, shared by the
  engine and the unit test. Lives in `libpico/libpico_utils.h` with the other
  shared helpers (moved there from a dedicated header):
  - `twill_splitmix64` / `twill_rand_bounded` — deterministic PRNG (no `rand()`),
    so every rank derives byte-identical `rho` from `TWILL_SEED`.
  - `twill_densify` — arbitrary group ids → dense `0..G-1` by sorting unique ids
    ascending (lowest raw id → group 0, which makes the WRR tie-break meaningful).
  - `twill_build_rho_group` — smooth weighted round-robin (credit/stride
    scheduling), weight `w[j]=|group j|`, deterministic lowest-id tie-break.
  - `twill_build_rho_shift` (identity) and `twill_build_rho_random` (Fisher–Yates).
  - `twill_pi` / `twill_sigma` — O(1) per-step maps; `sigma` is the exact inverse.
- `utils/twill_schedule_test.c` — standalone test (single process, no MPI calls;
  built with mpicc only because `libpico_utils.h` pulls in `mpi.h`).
  `make -C utils test-twill`.
- Verified: `rho` bijectivity + `rho_inv` consistency; per-step perfect matching;
  `sigma(pi(s,t),t)==s`; self block at `t=0`; group balance (exact integer
  arithmetic). Configs incl. ragged `[1]`, `[1,1,37]`, `[1,37]`, primes (P=13),
  interleaved ids, P=1000 random groups, P=4000. **38,151,582 checks, 0 failures.**

Balance findings (worst observed deviation from ideal `k*w_j/P`):
- **Prefix deviation `< 1` always** — the rigorous smooth-WRR guarantee. Max seen
  0.927 (P=1000, G=16 random groups).
- **Window deviation `≤ 2` always** (max 1.854). The plan's "≤ 1" is optimistic
  in the worst case; `< 2` is the provable bound (prefix `< 1` at both ends) and
  is what the test asserts.

## M2 — Engine + PICO wiring (done)

- `libpico/libpico_alltoall.c` (twill section appended after `alltoall_bine`,
  per the one-file-per-collective convention) — cached static context (validated
  against `(comm, P)`), env knobs, group-map discovery (`TWILL_MAP` file →
  node=group via `MPI_COMM_TYPE_SHARED` → single-group fallback), cross-rank
  consistency hash (`MPI_Bcast`+`MPI_Allreduce(MIN)`), and the shared windowed
  nonblocking engine (self copy + `min(W+8,P-1)` Irecv / `min(W,P-1)` Isend,
  `MPI_Waitsome` drain, no barriers). Three entry points
  `alltoall_twill_{group,random,shift}`. The schedule core (`twill_*` helpers)
  lives in `libpico_utils.h` with the other shared helpers.
- Wiring: declarations in `include/libpico.h`; three `CHECK_STR` registrations
  in `pico_core/pico_core_utils.c` (`twill_{group,random,shift}_over`); three
  JSON entries in `config/algorithms/MPI/LibPico/alltoall.json` (no power-of-two
  constraint, `count >= comm_sz`, tags `["twill","external","tapered"]`).
- Builds clean (`-O3 -Wall`, no new warnings).

Validation:
- **Standalone MPI driver** (vs `PMPI_Alltoall`, fully distinct per-(rank,block,
  element) data): comm sizes {2,3,7,8,16} × {group,random,shift} × per-pair
  counts {1,2,3,8,100,1024}; window sweep {1,2,8,128}; ragged `TWILL_MAP`
  ([1,1,5] on 7, [1,3,5,7] on 16); `TWILL_CACHE=0`. **All OK.**
- **End-to-end through `pico_core`** (its own ground-truth check vs
  `PMPI_Alltoall`, random data, `iter=5` to exercise the cached-schedule path):
  the full {2,3,7,8,16} × 3-variant × count sweep + window/map/cache cases —
  **49/49 pass**, results CSVs written. Requires env `COLLECTIVE_TYPE=ALLTOALL`
  (PICO selects the collective family from it).

## M3 — Hardening + docs (done)

Most hardening was folded into the M2 engine: node-derived default map,
cross-rank consistency hash, env handling (`TWILL_WINDOW/SEED/CACHE/MAP/GROUP`),
`INT_MAX` guards on `scount`/`rcount`, `MPI_IN_PLACE` rejection, and the
`twill_setup`/`twill_self`/`twill_exchange` instrumentation tags. This milestone
adds:

- `libpico/twill.md` — user note (overview, relabel-then-shift, env vars, map
  input + consistency guard, instrumentation, unit test, v1 scope), mirroring
  `instrument.md`.
- `utils/twill_gen_map.py` — builds a `TWILL_MAP` from a PICO `alloc_<n>.csv`,
  a hostfile, or `$SLURM_JOB_NODELIST`; `--regex` extracts a site-specific group
  field, else node=group. Verified: `g([0-9]+)` over 4 nodes × 2 ranks →
  2 groups of size 4 (`0 0 0 0 1 1 1 1`); node=group fallback → distinct ids.

Instrumentation validated end-to-end (`-DPICO_INSTRUMENT` build, `n=8`):
tags emit with balanced depth — `twill_exchange` ≈ 97.9% of iter time,
`twill_setup`/`twill_self` ≈ 0.2/0.3% (setup amortized by the cache). Instrument
CSV header: `rank0,twill_setup,twill_self,twill_exchange`.

## M4 — Evaluation assets (done)

- `tests/cpu/twill_alltoall.sh` — TUI run-config (`submit_wrapper.sh --file`)
  for the headline run: full OMPI vendor field + `bine_over` + `pairwise_ompi_over`
  + the three twill variants over the standard size sweep. `TWILL_*` knobs respect
  the ambient env; `TWILL_ONLY=yes` restricts to the twill variants; the config
  sets `NOTES="twill W=.. group=.. cache=.."` so runs are tagged in the metadata.
  `SKIP`/`IS_SEGMENTED` are derived from the algorithm list (always match length).
- **Single-allocation sweep** (no changes to `submit_wrapper.sh` / `orchestrator.sh`
  — they're reused verbatim). Three twill files:
  - `tests/cpu/twill_sweep.sh` — submits ONE `sbatch` job (params mirror
    submit_wrapper's Leonardo/QOS line, `--export=ALL`) whose body
    (`tests/cpu/twill_sweep_inner.sh`) runs the grid INSIDE the job's allocation, so
    each combo's `srun` attaches as a job STEP ⇒ exactly one `squeue` job, all combos
    on the same nodes. (Earlier used `salloc <command>`, which runs the body on the
    login node and relies on srun inheriting `SLURM_JOB_ID`; on Leonardo that does
    NOT hold, so every srun self-submitted its own job — the "one job per combo" bug.)
    Knobs via env: `TWILL_WINDOWS`/`TWILL_CACHES`/`TWILL_GROUPS`
    (defaults {2,8,32,128}×{0,1}×{4,8,16,32}); `TWILL_ONLY=yes`. Pre-compiles once on
    the login node via `TWILL_COMPILE_ONLY=yes` through the *local* config (submit_wrapper
    resets `COMPILE_ONLY` before sourcing the config, so a cmdline `COMPILE_ONLY=yes`
    is lost and it would otherwise submit a stray full benchmark job).
  - `tests/cpu/twill_sweep_inner.sh` — loops the grid; per combo calls the stock
    `submit_wrapper.sh --file tests/cpu/twill_sweep_local.sh`.
  - `tests/cpu/twill_sweep_local.sh` — sources `twill_alltoall.sh` but sets
    `LOCATION=local` (so submit_wrapper runs the stock orchestrator *in-process*,
    no nested sbatch) with `RUN=srun` (each combo's srun lands on the surrounding
    salloc nodes ⇒ SAME nodes ⇒ comparable). Each combo is a separate
    submit_wrapper call, so it gets its own standard `results/local/<ts>/` run;
    the combo is recorded in the metadata `NOTES` (twill_compare labels by NOTES).
    `twill_alltoall.sh`'s run-control knobs (LOCATION/RUN/COMPILE_ONLY/DEBUG/DRY)
    were made `${VAR:-default}` to allow this.
  - Verified locally: override/param/loop logic. The `salloc`+`srun`+`LOCATION=local`
    glue is Leonardo-specific (flagged in the script for first-run sanity-check).
  - `TWILL_ONLY=yes` (default) sweeps only the twill variants; baselines come from
    one full-field run via `submit_wrapper.sh --file tests/cpu/twill_alltoall.sh`.
- `utils/twill_compare.py` — quick text comparison of results (no plotting),
  stdlib-only. Reads result dirs OR `.tar.gz` archives (the runs compress+delete
  by default), summarizes the `highest` column (median/min/mean µs) per
  (algorithm, per-pair size), derives `comm_sz` from the CSV header and labels
  runs by the metadata `NOTES`. Per-run size×algorithm table (fastest marked,
  optional `--baseline` speedups) or `--cross ALGO` pivot (rows=run/knobs ×
  cols=size) for the sweep. Verified end-to-end on real `pico_core` output
  (dir + tarball + cross-run).
- `utils/twill_lower_bound.py` — volume/bandwidth lower bound from a
  `TWILL_MAP` + per-group global bandwidth: reports `T_lb` (aggregate/volume)
  and the per-group bottleneck bound, plus the achieved fraction vs a measured
  time. Verified against hand calculations on the [1,3,5,7]
  ragged map.
- Stretch: `TWILL_SKEW_US` (default 0/off) — per-rank, per-call random pre-
  exchange sleep for skew-tolerance probing; outside the `twill_exchange` tag.
  Correctness re-verified with skew on (300 µs) across the matrix.

Shell configs pass `bash -n`; Python passes `py_compile`. Note PICO requires
`COLLECTIVE_TYPE=ALLTOALL` (the orchestrator sets it); for ad-hoc local runs of
`pico_core` set `OUTPUT_DIR`/`DATA_DIR`/`OUTPUT_LEVEL`/`LOCATION` too.

## Build & test quick reference

```
# schedule unit test (single process, no mpirun)
make -C utils test-twill
# full build
PICOCC=mpicc PICO_DIR=$(pwd) make all
# instrumented build (twill_setup/self/exchange tags)
PICO_INSTRUMENT=1 PICOCC=mpicc PICO_DIR=$(pwd) make all
```
