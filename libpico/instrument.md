# instrument.md — Instrumenting Custom Collectives

## 1. Overview

* Instrumentation lets you mark regions inside custom MPI collectives that should be **timed separately** from the main collective execution.
* Controlled via two macros:

  ```c
  PICO_TAG_BEGIN("tag");
  PICO_TAG_END("tag");
  ```
* Enabled only when building with:

  ```c
  -DPICO_INSTRUMENT
  ```

  and **not** with `PICO_NCCL` or `PICO_MPI_CUDA_AWARE`.
* Declarations live in [`include/libpico.h`](../include/libpico.h).
* Implementation is in `libpico/instrument_collectives.c`.

> See the general build/test README for how to enable the instrumentation flag. Default build behavior is through declarative test files, and
  automation scripts for test orchestration, so the entrypoint for the build should always be [`scripts/submit_wrapper.sh`](../scripts/submit_wrapper.sh)
  and [`tui/main.py`](../tui/main.py) for test declaration.

---

## 2. Where to instrument

* Place `PICO_TAG_BEGIN/END` **inside your CPU-only collective implementations** (e.g. the ones inside `libpico_<collective>.c` files).
* Do **not** instrument GPU (`NCCL` or CUDA-aware) collectives.
* The core benchmarking driver (`pico_core`) automatically handles discovery, buffer allocation, clearing, and snapshotting, you only need to annotate your collectives.

---

## 3. How to instrument a collective

### Minimal example

From `libpico_alltoall.c`, based on Open MPI’s pairwise algorithm:

```c
int alltoall_pairwise_ompi(const void *sbuf, size_t scount, MPI_Datatype sdtype, 
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int err = 0, rank, size, step, sendto, recvfrom;
  void *tmpsend, *tmprecv;
  ptrdiff_t lb, sext, rext;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Type_get_extent(sdtype, &lb, &sext);
  MPI_Type_get_extent(rdtype, &lb, &rext);

  for (step = 1; step < size + 1; step++) {
    sendto   = (rank + step) % size;
    recvfrom = (rank + size - step) % size;
    tmpsend  = (char*)sbuf + (ptrdiff_t)sendto * sext * (ptrdiff_t)scount;
    tmprecv  = (char*)rbuf + (ptrdiff_t)recvfrom * rext * (ptrdiff_t)rcount;

    /* Example: instrument sendrecv operation */
    PICO_TAG_BEGIN("sendrecv");
    err = MPI_Sendrecv(tmpsend, scount, sdtype, sendto, 0,
                       tmprecv, rcount, rdtype, recvfrom, 0,
                       comm, MPI_STATUS_IGNORE);
    PICO_TAG_END("sendrecv");

    if (err != MPI_SUCCESS) return err;
  }

  return MPI_SUCCESS;
}
```

### Nesting rules

* Nesting is **allowed**: you may call `PICO_TAG_BEGIN("tag")` multiple times.
* Time is accumulated until the outermost `PICO_TAG_END("tag")`.
* **Important:** when leaving the collective, every `PICO_TAG_BEGIN` must be matched with the same number of `PICO_TAG_END` → depth must be 0.
* **Allowed:**

  ```c
  PICO_TAG_BEGIN("foo");
    // ...
  PICO_TAG_END("foo");
  ```
* **Not allowed:**

  ```c
  for (int i = 0; i < steps; i++){
      PICO_TAG_BEGIN("foo");
      // ...
  }
  PICO_TAG_END("foo");
  // function returns here → depth != 0
  ```

The library enforces this and errors if tags remain open at snapshot or clear time.

---

## 4. Output behavior

* When instrumentation is enabled, **only rank 0 times are recorded by default.**
  This avoids overwhelming output volume across nodes.
* If you suspect faulty behavior on other nodes, ensure the faulty node is mapped to rank 0 before requesting a run.

### CSV output format

* Each line corresponds to one benchmark iteration.
* Columns are:

  ```
  rank0_iter_time, tag0_time, tag1_time, ..., tagK_time
  ```
* Where:

  * `rank0_iter_time` is the total measured iteration time on rank 0.
  * `tagN_time` is the accumulated excluded time for that tag in that iteration.
  * Tags appear in the order they were first encountered in your collectives.

---

## 5. Notes

* Maximum number of tags is `LIBPICO_MAX_TAGS` (default: 32).
  You can change it at compile time with `-DLIBPICO_MAX_TAGS=N`, but this is not recommended unless absolutely needed.
* Strings passed to `PICO_TAG_BEGIN/END` must live for the program lifetime (string literals are safe).
* Instrumentation is meant for **deep dives after normal benchmarking**: run standard benchmarks first, then recompile with `PICO_INSTRUMENT` for fine-grained breakdowns.


