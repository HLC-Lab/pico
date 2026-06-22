# Schedgen Updates and Dynamic Collective Selection

Schedgen is the trace generator originally developed by Torsten Hoefler and
Timo Schneider at the Scalable Parallel Computing Lab (SPCL), ETH Zurich,
and released under the BSD 4-Clause license. This version keeps the original
goal file writer while integrating new tooling to support dynamic selection of
collective algorithms when building MPI traces for PICO workflows.

## How Schedgen Works

Schedgen is driven by the `--ptrn` command-line argument parsed via
`gengetopt`. For synthetic patterns, the requested generator constructs a
`Goal` writer, iterates over all ranks, and emits the corresponding GOAL
operations (`send`, `recv`, `calc`, and dependency edges) before flushing the
buffered schedule to disk. The helpers in `schedgen.cpp` and the `coll/`
subdirectory encapsulate the per-algorithm logic—tree traversals, ring
pipelines, or randomised shuffles—and translate MPI semantics such as roots,
segments, and communicator sizes into concrete point-to-point traffic inside
the GOAL file.

When `--ptrn trace` is selected, Schedgen switches to the pipeline implemented
in `process_trace.cpp`. It replays an instrumented MPI execution, recreates
point-to-point activity, and, for each collective call, consults the
rule-driven selector (`coll/collective_selector.cpp`). The selector chooses a
registered algorithm based on communicator size and message size, using either
the built-in defaults or the user-provided rules file passed with
`--selector-rules`. Each chosen algorithm delegates back to the same `Goal`
writer, so both synthetic patterns and trace replays ultimately share the
same GOAL emission machinery.

## Built-in Algorithms

- Pattern generators available through `--ptrn` include
  `binarytreebcast`, `binomialtreebcast`, `binomialtreereduce`, `pipelinedring`,
  `pipelinedringdep`, `doublering`, `gather`, `scatter`, `linbarrier`,
  `dissemination`, `random_bisect`, `random_bisect_fd_sym`, `linear_alltoall`,
  `linear_alltoallv`, `allreduce_recdoub`, `allreduce_ring`, `resnet152`,
  `chained_dissem`, and the trace-driven mode `trace`.
- The dynamic selector recognises the following algorithm names for each
  collective kind: `barrier` → `dissemination`; `allreduce` → `recursive_doubling`,
  `ring`, `reduce_scatter_allgather`; `iallreduce` → `dissemination`; `bcast` →
  `binomial`, `binary`; `allgather` → `dissemination`; `reduce` → `binomial`;
  `alltoall` → `linear`. These names match the GOAL emitters registered under
  `coll/collective_registry.cpp` and can be referenced in rule files.

## Recent Improvements

- Integrated a rule-based collective selector that chooses an algorithm based
  on communicator size and message size while traces are produced.
- Added optional `--selector-rules` and `--selector-debug` CLI flags to
  configure and inspect collective choices.
- Refactored the trace-processing pipeline so that the selector is consulted at
  every collective site before emitting GOAL operations.

Without a rules file, Schedgen falls back to the following defaults:

```
barrier            -> dissemination
allreduce          -> recursive_doubling
iallreduce         -> dissemination
bcast              -> binomial
allgather          -> dissemination
reduce             -> binomial
alltoall           -> linear
```

## Using Dynamic Selection

When generating traces from an MPI log (the `trace` pattern), pass a rules file
to override the defaults:

```
./schedgen trace --selector-rules <dynamic-rules-descriptor> \
                 --selector-debug \
                 --filename goal.out <other trace options>
```

- `--selector-rules` points to a text file that lists per-collective rules.
- `--selector-debug` prints the chosen algorithm for each collective to
  standard output, which is useful when validating new rules.

If no file is provided, only the built-in fallbacks listed above are used.

## Rule File Format

Each non-comment, non-empty line has six space-separated fields:

```
<collective> <algorithm> <min_comm> <max_comm> <min_msg> <max_msg>
```

- `<collective>` must be one of: `barrier`, `allreduce`, `iallreduce`, `bcast`,
  `allgather`, `reduce`, `alltoall`.
- `<algorithm>` is the implementation label stored in the GOAL trace (for
  example `ring`, `recursive_doubling`, `dissemination`).
- `<min_comm>` / `<max_comm>` bound the communicator size (number of ranks).
- `<min_msg>` / `<max_msg>` bound the message size in bytes; Schedgen computes
  this from the original trace metadata (`count * datatype_size`).

The bounds are inclusive. Use `*` to denote an open upper bound (`max_comm` or
`max_msg`) and `0` for a lower bound that should always match.

Example:

```
# allreduce: small messages use recursive doubling, large use ring
allreduce recursive_doubling 0 512 0 131072
allreduce ring 0 * 131072 *
```

Lines beginning with `#` are ignored.

## Credits

- Original design and implementation: Torsten Hoefler and Timo Schneider
  (SPCL, ETH Zurich).
- Dynamic selection integration and refactoring: Saverio Pasqualoni as part of
  the PICO project at Sapienza University of Rome.
