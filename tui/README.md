# Benchmarking TUI

This Textual-based TUI guides you through selecting:
1. **Compute environment** (e.g. local, lumi, leonardo)  
2. **SLURM partition & QOS** (if the environment uses SLURM)  
3. **MPI implementation**  
4. **Review/confirm** your choices  

---

## How It Works

### 1. `main.py`
- Instantiates `BenchmarkApp`, applies `style.tcss` (in progress) <!-- TODO: make the style.tcss work -->
- Renders a `Header`, a `Router` container, and a `Footer`.

### 2. `tui/components.py` → `Router`
- On mount, creates a fresh `SessionConfig`.
- Pushes the **EnvironmentStep** screen first.

### 3. `tui/steps/base.py` → `StepScreen`
- All step screens subclass `StepScreen`.
- Carries the shared `session` object.
- Provides `next()` to pop the current screen and push the next one.

### 4. `tui/steps/environment.py` → `EnvironmentStep`
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

### 5. `tui/steps/partition.py` → `PartitionStep`
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

### 6. `tui/steps/mpi.py` → `MPIStep`
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

### 7. `tui/steps/summary.py` → `SummaryStep`
- **Compose**:
  - Renders a JSON dump of:
    - `session.environment.general`
    - `session.partition.details` & `.qos`
    - `session.mpi.config`
  - A **Finish** button.
- **on_button_pressed**:
  - Exits the app.

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

## Usage

```bash
pip install textual
python main.py
````

Navigate using arrow keys or type to search your dropdowns, then press **Enter** to select. Press **n** or click **Next** once enabled to proceed through the flow. Press **q** at any time to quit.

---

## 🧩 How to Expand This TUI

This Textual TUI is designed as a **step-by-step wizard** that gathers configuration from the user in discrete screens. Each screen handles a single responsibility (environment selection, partition selection, MPI, etc.) and passes the shared session state forward.

To extend or adapt the application, follow these strategies:

### ➕ Add a New Step (Screen)

Each screen is a subclass of `StepScreen` from `tui.steps.base`.

1. **Create a new file** under `tui/steps/`, e.g. `compiler.py`.
2. **Subclass `StepScreen`** and define:

   * `compose()` to render widgets.
   * `on_select_changed()` or `on_button_pressed()` to handle input.
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
```

Then in the MPI step:

```python
def on_button_pressed(self, event):
    if event.button.id == "next":
        self.next(__import__('tui.steps.compiler', fromlist=['CompilerStep']).CompilerStep)
```

---

### 🧠 Add New Data to the Session

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

---

### 🎨 Customize Styling

Edit `style.tcss` to apply consistent theme elements:

* Use `classes="screen-header"` for section titles.
* Add your own classes for spacing, buttons, or theming.

---

### 🧪 Add Validations

Use `disabled=True` on buttons until required selections are made. You can also override `on_mount()` to validate conditions based on previous screens.

---

### 🔄 Replace Configuration Sources

To replace or expand `config_loader.py`:

* Connect to a REST API.
* Use `pyyaml` for YAML configs.
* Cache parsed configs globally to reduce I/O.

---

### 🛠 Advanced UI Features (Optional)

You can enhance screens using:

* `DataTable` for tabular display
* `Input` widgets for manual fields
* `Modal` or `Dialog` overlays for confirmation

See the [Textual documentation](https://textual.textualize.io/) for available widgets and event handling.

---
