# Benchmarking TUI

A Textual-based terminal UI to configure and launch collective benchmarking jobs.  
Guides you through:

1. **Environment** selection (e.g. local, lumi, leonardo)  
2. **Partition & QOS** (if SLURM is enabled)  
3. **MPI implementation**  
4. **Review & confirm** your choices  

---

## Quick Start

1. Create & activate a Python 3.9+ virtual environment  
2. Install dependencies:
```bash
   pip install textual rich
````
3. From project root run:
```bash
   python tui/main.py
```
4. Navigate dropdowns with arrow keys or typing, select with **Enter**.
   * **n** or click **Next** to advance (enabled only when all required selections are made).
   * **q** to quit at any time.
   * **h** to toggle context‐sensitive Help.

---

## Key Design Points

- **Modular Steps**  
  Each choice lives in its own screen class.  
- **Shared Session**  
  A single `SessionConfig` dataclass instance flows through all steps.  
- **Dynamic Widgets**  
  The QOS dropdown is replaced at runtime to avoid mutating internal `Select.options`.  
- **Navigation Guards**  
  “Next” buttons are disabled until required inputs are provided.  
- **Package Layout**  
  Logical separation between loader, data models, UI components, and individual step screens.

---

## How It Works

### 1. `main.py`
- Instantiates `BenchmarkApp`, applies `style.tcss` (in progress) <!-- TODO: make the style.tcss work -->
- Renders a `Header`, a `Router` container, and a `Footer`.

### 2. `tui/components.py` → `Router`
- On mount, creates a fresh `SessionConfig`.
- Pushes the **EnvironmentStep** screen first.

### 3. `tui/steps/help.py` → `HelpScreen`
- A simple modal that displays a single description string. Closes on **h** or the **Close** button.

### 4. `tui/steps/base.py` → `StepScreen`
- All step screens subclass `StepScreen`.
- Carries the shared `session` object.
- Provides `next()` to pop the current screen and push the next one.
- Declares abstract `get_help_desc()`—each step must override it.

### 5. `tui/steps/environment.py` → `EnvironmentStep`
- **Compose**:
  - A header label.
  - A `Select` of environments (`[(env, env), …]`).
  - A **Next** button (initially disabled).
- **on_select_changed**:
  - When the user picks an environment:
    1. Loads `general.json` (and `slurm.json` if SLURM-enabled).
    2. Saves into `session.environment`.
    3. Enables the **Next** button.
- **on_button_pressed**:
  - Chooses next screen:
    - **PartitionStep** if SLURM is enabled.
    - **MPIStep** otherwise.

### 6. `tui/steps/partition.py` → `PartitionStep`
- **Compose**:
  - Partition dropdown (`id="partition-select"`).
  - Placeholder QOS dropdown (empty, no `id`).
  - **Next** button (disabled).
- **on_select_changed**:
  1. When **partition** is chosen:
     - Reads its `"QOS"` entries.
     - Builds a brand-new QOS `Select` (no `id`) with the required options.
     - Removes the placeholder by querying `Select` widgets by index.
     - Mounts the real dropdown after the QOS label.
  2. When **QOS** is chosen:
     - Stores it in `session.partition.qos`.
  3. Enables **Next** only if both `session.partition.name` and `.qos` are set.
- **on_button_pressed**:
  - Advances to **MPIStep**.

### 7. `tui/steps/mpi.py` → `MPIStep`
- **Compose**:
  - MPI dropdown (`id="mpi-select"`).
  - **Next** button (disabled).
- **on_select_changed**:
  - When user picks an MPI:
    - Loads its JSON config.
    - Stores in `session.mpi`.
    - Enables **Next**.
- **on_button_pressed**:
  - Advances to **SummaryStep**.

### 8. `tui/steps/summary.py` → `SummaryStep`
- **Compose**:
  - Renders a JSON dump of:
    - `session.environment.general`
    - `session.partition.details` & `.qos`
    - `session.mpi.config`
  - A **Finish** button.
- **on_button_pressed**:
  - Exits the app.

Buttons are rendered disabled by default and become enabled only when all required fields are set.

---

## Context-Sensitive Help (`h`)

Press **h** in any step to toggle a help overlay:

* **No selection yet** → Generic prompt (`"First choose an environment."` / `"First choose a partition then pick a QOS."` / `"First pick an MPI library."`)
* **Dropdown focused** → Shows that option’s `"desc"` from the JSON.
* **Partial selections** → Guides you to the next required action (`"Pick QOS for this partition."`, etc.)
* **Final step** → Provides summary guidance (`"Review the configuration, then press Finish or 'q' to quit."`)

---

## Dependencies

* **Python 3.9+**
* **textual** ≥ 0.22
* **rich** ≥ 10

---

## Extending the TUI

* **Add a new step**: create `tui/steps/your_step.py`, subclass `StepScreen`, implement `compose()`, `get_help_desc()`, event handlers, and wire it into the previous step via `self.next(YourStep)`.
* **Expand `SessionConfig`** in `models.py` to carry more data.
* **Style** by editing `style.tcss`.
* **Swap config sources** in `config_loader.py` to YAML, a database, or REST API.

##  How to Expand This TUI

This Textual TUI is designed as a **step-by-step wizard** that gathers configuration from the user in discrete screens. Each screen handles a single responsibility (environment selection, partition selection, MPI, etc.) and passes the shared session state forward.

To extend or adapt the application, follow these strategies:

### Add a New Step (Screen)

Each screen is a subclass of `StepScreen` from `tui.steps.base`.

1. **Create a new file** under `tui/steps/`, e.g. `compiler.py`.
2. **Subclass `StepScreen`** and define:
   * `compose()` to render widgets.
   * `on_select_changed()` or `on_button_pressed()` to handle input.
   * `get_help_desc()` to get selection informations.
3. **Push the next step** using `self.next(NextStepClass)`.
4. **Update the previous step** to navigate to this new one.

#### Example

```python
# tui/steps/compiler.py
class CompilerStep(StepScreen):
    def compose(self):
        yield Static("Choose Compiler", classes="screen-header")
        yield Select([("GCC", "gcc"), ("Clang", "clang")], prompt="Compiler:", id="compiler-select")
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        self.session.compiler = event.value
        self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        self.next(SummaryStep)  # or the next logical screen

    def get_help_desc(self) -> str:
        return "Help informations"
```

Then in the MPI step:

```python
def on_button_pressed(self, event):
    if event.button.id == "next":
        self.next(__import__('tui.steps.compiler', fromlist=['CompilerStep']).CompilerStep)
```

### Add New Data to the Session

Edit `models.py` to expand the `SessionConfig`:

```python
@dataclass
class CompilerSelection:
    name: str = ''
    flags: Optional[str] = None

@dataclass
class SessionConfig:
    ...
    compiler: CompilerSelection = field(default_factory=CompilerSelection)
```

Each screen can then access this via `self.session.compiler`.

### Add Validations

Use `disabled=True` on buttons until required selections are made and remember to disable it again when selection is blank. You can also override `on_mount()` to validate conditions based on previous screens.

---
Enjoy building out your benchmarking wizard!

