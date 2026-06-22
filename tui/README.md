# PICO Benchmarking TUI

A Textual-based terminal UI for building repeatable PICO benchmarking runs.  
The TUI now drives the entire test description: environment & SLURM settings,
test dimensions, library/task mix, algorithm selection, and final export of the
`.json` + `.sh` bundle consumed by `scripts/submit_wrapper.sh`.

## Quick Start

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:
   ```
   pip install textual rich packaging
   ```
3. From the project root launch:
   ```
   python tui/main.py
   ```
4. Navigate with the keyboard:
   - `Tab` / `Shift+Tab` to move focus
   - `Enter` to confirm
   - `n` / `p` (or `N` / `P`) for next/previous step
   - `h`, `H`, or `?` for contextual help
   - `q` to open the quit dialog

The app targets Textual ≥ 0.39; we actively test with 0.50+.

## Workflow Overview

The UI is split into four screens that exchange state through a shared
`SessionConfig` object (see `tui/models.py`).

### 1. Configure

`tui/tui/steps/configure.py`

- **Environment** — loads `config/environment/<name>/<name>_general.json`.
  Optional SLURM metadata is fetched from `<name>_slurm.json`.
- **Partition + QoS** — unlocked for SLURM environments, with validation of
  node counts (`nodes_limit`) and wall-clock time (`time_limit`).
- **Run toggles** — compile-only, debug, dry-run, compression, delete after
  compression, plus optional inject parameters for raw `sbatch`/env knobs.
- **Advanced scheduling** — enable node exclusion and dependency chaining.
- **Test dimensions** — pick datatype, buffer sizes, and segment sizes from
  multi-select lists; element counts are recalculated live from datatype size.
- **Validation** — inputs disable/enable automatically; inline error labels
  explain violations (e.g., invalid node counts or time format).

Once all mandatory data are valid, `Next` becomes available.

### 2. Libraries

`tui/tui/steps/libraries.py`

- Pulls library definitions from `config/environment/<env>_libraries.json`.
- Start with a single row and add/remove more using `+` / `–`.
- Each row lets you:
  - choose a library (per environment)
  - set CPU/GPU tasks-per-node (validated against partition limits)
  - toggle usage of PICO custom backends
- Select at least one collective (Allreduce, Broadcast, …) via checkboxes.
- Multiple libraries of the same type are supported; the UI prevents
  duplicate selections in active rows.

### 3. Algorithms

`tui/tui/steps/algorithms.py`

- Each collective has its own tab. Use number keys to jump directly to tab `n`.
- Per library column lists the algorithms found in
  `config/algorithms/<standard>/<lib_type>/<collective>.json`.
- Version gating ensures only algorithms compatible with the configured
  library version can be checked.
- If the library row has `PICO backend` enabled, additional PICO-only entries
  are shown (annotated with “PICO custom”).
- `Next` stays disabled until every selected collective has at least one
  algorithm, and every library contributes at least one entry.

### 4. Summary & Save

`tui/tui/steps/summary.py`

- Presents both the raw JSON configuration (`session.to_dict()`) and a readable
  summary derived from `SessionConfig.get_summary()`.
- Saving writes two files under `tests/`:
  - `<name>.json` — the full configuration
  - `<name>.sh` — an executable wrapper created via `json_to_exports`, exporting
    the environment variables consumed by `submit_wrapper.sh`
- Filenames are auto-suffixed to avoid overwriting existing tests.

## Data Sources

- Environment definitions live in `config/environment/<env>/`.
  - `<env>_general.json` — human-readable description, platform toggles,
    optional interpreter module list.
  - `<env>_slurm.json` — SLURM partitions, QoS options, node/time limits.
  - `<env>_libraries.json` — available MPI/NCCL libraries with compiler hints,
    load mechanism (module/env), and GPU support metadata.
- Algorithms are resolved from `config/algorithms/` using the library standard,
  library type, and collective kind.

Updating these JSON files instantly reflects in the TUI; no code changes are
required for new partitions, libraries, or algorithm metadata.

## Helpful Features

- **Contextual help** (`h`, `H`, `?`) opens an overlay tailored to the focused
  widget, summarising current selections and constraints.
- **Quit dialog** (`q`) protects against accidental exits.
- **Live validation** — every field updates the `SessionConfig` object and
  enables/disables dependent inputs without reloading the screen.
- **Keyboard friendly** — the entire flow can be completed via the bindings
  listed in “Quick Start”.
- **Debug mode preset** — automatically reduces run time and iterations, while
  forcing debug compilation flags.

## Extending the UI

- Add new data attributes in `tui/models.py` and surface them in the relevant
  step screen.
- Introduce extra steps by subclassing `StepScreen` under `tui/tui/steps/` and
  chaining them through `self.next(...)` / `self.prev(...)`.
- When adding new configuration sections, remember to include the data in
  `SessionConfig.to_dict()` so the summary and export script stay in sync.

## Credits

- Original TUI prototype: Daniele De Sensi and Saverio Pasqualoni.
- Current iteration (2024–2025): major redesign by Saverio Pasqualoni, including
  multi-library support, algorithm selection, JSON/Shell export pipeline, and
  richer validation UX.
