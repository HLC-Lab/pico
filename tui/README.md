# WARN
THIS README IS OUTDATED BUT CAN BE STILL USED TO HAVE A ROUGH IDEA OF THE GENERAL TUI WORKFLOW

# Benchmarking TUI

A Textual-based terminal UI to configure and launch collective benchmarking jobs.  
Guides you through:

1. **Environment** selection (e.g. local, lumi, leonardo)  
2. **Partition & QOS** (if SLURM is enabled)  
3. **Node configuration** (Nodes, Tasks per node, Test time)  
4. **MPI implementation**  
5. **Review & confirm** your choices  

---

## Quick Start

1. Create & activate a Python 3.9+ virtual environment  
2. Install dependencies: `pip install textual rich`
3. From project root run: `python tui/main.py`
4. Navigate dropdowns with arrow keys or typing, select with **Enter**.
   * **n** or click **Next** to advance (enabled only when all required selections are made).
   * **q** to quit at any time.
   * **h** to toggle context‐sensitive Help.

---

## Key Design Points

* **Modular Steps**
  Each logical decision point lives in its own screen class.
* **Shared Session**
  A `SessionConfig` dataclass instance holds environment, partition, resources, MPI, etc.
* **Dynamic Rendering**
  Widgets (e.g. QOS dropdown, time input) are conditionally rendered depending on SLURM presence.
* **Validation Guards**
  Buttons are disabled unless all visible inputs are valid.
* **Config-Aware UI**
  Step behavior dynamically adapts based on selected environment and config data.

---

## How It Works

### 1. `main.py`

* Instantiates `BenchmarkApp`, applies `style.tcss`.
* Renders a `Header`, `Router` screen container, and a `Footer`.

### 2. `tui/components.py` → `Router`

* On mount, creates a `SessionConfig` instance.
* Pushes the **EnvironmentStep** first.

### 3. `tui/steps/environment.py` → `EnvironmentStep`

* Dropdown for selecting environment.
* Loads `general.json` and optionally `slurm.json`.
* Stores into `session.environment`.
* If SLURM enabled, proceeds to **PartitionStep**. Otherwise goes to **NodeConfigStep**.

### 4. `tui/steps/partition.py` → `PartitionStep`

* Dropdowns for **Partition** and **QOS**.
* When partition is selected:

  * Dynamically builds and mounts a QOS dropdown using its `"QOS"` list.
* Only selected QOS details are stored (not all QOS).
* On valid selection of both, proceeds to **NodeConfigStep**.

### 5. `tui/steps/node_config.py` → `NodeConfigStep`

* Configures compute resources:

  * **Always shows**: number of nodes (min 2).
  * **Only if SLURM**:

    * Tasks per node (max = `PARTITION_CPUS_PER_NODE`)
    * Test time (HH\:MM\:SS, max from `QOS_MAX_TIME`)
* Validates ranges based on the config.
* Stores values into `session.nodes`, `session.tasks_per_node`, `session.test_time`.
* Proceeds to **MPIStep**.

### 6. `tui/steps/mpi.py` → `MPIStep`

* Dropdown of MPI implementations.
* On selection, loads corresponding config and stores into `session.mpi`.
* Proceeds to **SummaryStep**.

### 7. `tui/steps/summary.py` → `SummaryStep`

* Displays a JSON summary of:

  * `environment.general`
  * `partition.details` and `qos`
  * node/test configuration
  * `mpi.config`
* Final **Finish** button exits the app.

---

## Context-Sensitive Help (`h`)

Press **h** in any step to toggle a help overlay:

* **No selection yet** → Prompts user to start selecting.
* **Dropdown focused** → Displays the selected option’s `"desc"` from the JSON.
* **Partial selections** → Guides next expected action.
* **Final step** → Gives summary confirmation guidance.

---

## Dependencies

* **Python 3.9+**
* **textual** ≥ 0.22
* **rich** ≥ 10

---

## Extending the TUI

### Add a New Step

1. Create a file in `tui/steps/`, e.g. `compiler.py`
2. Subclass `StepScreen` and implement:
   * `compose()` — UI widgets
   * Event handlers: `on_select_changed()`, `on_button_pressed()`
   * `get_help_desc()` — help string for focused widget
3. Push it from the previous step using `self.next(CompilerStep)` (and adjust next step)
4. Update `SessionConfig` in `models.py` if needed to save also new step's data
5. Adjust `SummaryStep` to show new saved data.

#### Example

```python
class CompilerStep(StepScreen):
    def compose(self):
        yield Static("Choose Compiler", classes="screen-header")
        yield Select([("GCC", "gcc"), ("Clang", "clang")], prompt="Compiler", id="compiler-select")
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        self.session.compiler = event.value
        self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        self.next(SummaryStep)

    def get_help_desc(self):
        return "Select your preferred compiler."
```

Then in the previous step (e.g. MPI):

```python
def on_button_pressed(self, event):
    if event.button.id == "next":
        self.next(CompilerStep)
```

### Add Data to the Session

Update `SessionConfig` in `models.py`:

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

Update `SummaryStep` in `tui/steps/summary.py`:

```python
@dataclass
class SummaryStep(StepScreen):
    ...
    summary = {
      ...
      "compiler": self.session.compiler
```

---

## Feature Highlights

* ✅ Step-by-step interactive benchmarking config
* ✅ SLURM-aware behavior (partition, QOS, resource bounds)
* ✅ Local environment fallback with simplified UI
* ✅ Validated inputs with clear error messages
* ✅ Contextual help (`h`) in every step
* ✅ Extensible via modular screen system
* ✅ Textual-based, works in any modern terminal

---

Enjoy benchmarking with clarity and structure!

