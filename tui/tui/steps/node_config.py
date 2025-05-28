# tui/steps/node_config.py

"""
Step 3: Choose number of nodes, (tasks per node), and (test time).
Tasks/time only when SLURM is enabled.
"""

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Input, Button, Label, Switch
from .base import StepScreen


default_time = "00:30:00"

class NodeConfigStep(StepScreen):
    
    def compose(self):
        """Render inputs and Next button."""

        slurm_enabled = self.session.environment.general.get("SLURM", False)
        tp_max = self.task_max(slurm_enabled=slurm_enabled)
        time_max = self.time_max_str(slurm_enabled=slurm_enabled)

        yield Horizontal(
            Vertical(
                Static("Number of Nodes", classes="field-label"),
                Input(placeholder=f"min {self.min_nodes() or '2'}, max {self.max_nodes() or '∞'}", id="nodes-input"),
                Label("", id="nodes-error", classes="error"),
                classes="field",
            ),
            Vertical(
                Static("Exclude?", classes="field-label"),
                Switch(id="exclude-switch", value=False, disabled=not slurm_enabled),
                classes="switch-col",
            ),
            Vertical(
                Static("Excluded Nodes", classes="field-label"),
                Input(placeholder=f"What nodes do you want to exclude?", id="excluded-noodes", disabled=True),
                Label("", id="nodes-error", classes="error"),
                classes="field",
            ),
            classes="row",
        )

        yield Horizontal(
            Vertical(
                Static("Tasks per Node", classes="field-label"),
                Input(placeholder=f"1–{tp_max}", id="tasks-input", value="1", disabled=not slurm_enabled),
                Label("", id="tasks-error", classes="error")
            ),
            Vertical(
                Static("Test Time", classes="field-label"),
                Input(placeholder=f"HH:MM:SS (max {time_max})", id="time-input", value=default_time, disabled=not slurm_enabled),
                Label("", id="time-error", classes="error")
            ),
            classes="row",
        )

        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Next", id="next", disabled=True),
            classes="button-row"
        )

    # ─── Event Handlers ────────────────────────────────────────────────────────

    def on_input_changed(self, event):
        """Validate inputs and toggle Next button."""
        nid = event.input.id
        # Validate this field
        if nid == "nodes-input":
            ok, msg = self.validate_nodes(event.value)
            self.query_one("#nodes-error", Label).update(msg)
        elif nid == "tasks-input":
            ok, msg = self.validate_tasks(event.value)
            self.query_one("#tasks-error", Label).update(msg)
        elif nid == "time-input":
            ok, msg = self.validate_time(event.value)
            self.query_one("#time-error", Label).update(msg)

        # Check overall validity
        nodes_ok, _ = self.validate_nodes(self.query_one("#nodes-input", Input).value)
        if self.session.environment.general.get("SLURM", False):
            tasks_ok, _ = self.validate_tasks(self.query_one("#tasks-input", Input).value)
            time_ok, _  = self.validate_time(self.query_one("#time-input",  Input).value)
            all_ok = nodes_ok and tasks_ok and time_ok
        else:
            all_ok = nodes_ok

        self.query_one("#next", Button).disabled = not all_ok

    def on_button_pressed(self, event):
        """Store values and proceed."""
        if event.button.id == "next":
            self.session.nodes = int(self.query_one("#nodes-input", Input).value)
            if self.session.environment.general.get("SLURM", False):
                self.session.tasks_per_node = int(self.query_one("#tasks-input", Input).value)
                self.session.test_time      = self.query_one("#time-input",  Input).value
            from tui.steps.mpi import MPIStep
            self.next(MPIStep)
        elif event.button.id == "prev":
            self.session.nodes = 0
            self.session.tasks_per_node = 1
            self.session.test_time = default_time
            from tui.steps.configure import ConfigureStep
            self.prev(ConfigureStep)

    def get_help_desc(self) -> str:
        """Contextual help based on focused input."""
        f = getattr(self.focused, "id", "")
        if f == "nodes-input":
            if self.session.environment.general.get("SLURM", False):
                mn, _ = self.time_bounds()  # misuse; correct is min_nodes()
                return f"Nodes: {self.min_nodes()}–{self.max_nodes() or '∞'}"
            else:
                return "Nodes (min 2, no max)"
        if f == "tasks-input":
            return f"Tasks/node: 1–{self.task_max()}"
        if f == "time-input":
            return f"Time (max {self.time_max_str()})"
        return "Configure run resources before MPI."

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def min_nodes_no_slurm(self) -> int:
        return 2

    def min_nodes(self) -> int:
        qos_cfg = self.session.partition.qos_details or {}
        return int(qos_cfg.get("QOS_MIN_NODES", 2))

    def max_nodes(self) -> int | None:
        qos_cfg = self.session.partition.qos_details or {}
        m = qos_cfg.get("QOS_MAX_NODES")
        return int(m) if m is not None else None

    def task_max(self, slurm_enabled: bool = True) -> int:
        if not slurm_enabled:
            return 1
        val = self.session.partition.details.get("PARTITION_CPUS_PER_NODE")
        return int(val) if val is not None else 1

    def time_bounds(self) -> tuple[int, int | None]:
        """
        Return (min_sec, max_sec) for test time from QOS_MAX_TIME.
        """
        qos_cfg = self.session.partition.qos_details or {}
        tmax = qos_cfg.get("QOS_MAX_TIME")
        if not tmax:
            return (0, None)
        if "-" in tmax:
            days, timestr = tmax.split("-", 1)
            days = int(days)
        else:
            days = 0
            timestr = tmax
        h, m, s = map(int, timestr.split(":"))
        max_sec = days*86400 + h*3600 + m*60 + s
        return (1, max_sec)

    def time_max_str(self, slurm_enabled: bool = True) -> str:
        if not slurm_enabled:
            return "∞"
        qos_cfg = self.session.partition.qos_details or {}
        return qos_cfg.get("QOS_MAX_TIME", "∞")

    # ─── Validators ─────────────────────────────────────────────────────────────

    def validate_nodes(self, text: str) -> tuple[bool, str]:
        try:
            val = int(text)
        except ValueError:
            return False, "Must be integer"
        if self.session.environment.general.get("SLURM", False):
            mn = self.min_nodes()
            mx = self.max_nodes()
            if val < mn:
                return False, f"Min {mn}"
            if mx is not None and val > mx:
                return False, f"Max {mx}"
        else:
            mn = self.min_nodes_no_slurm()
            if val < mn:
                return False, f"Min {mn}"
        return True, ""

    def validate_tasks(self, text: str) -> tuple[bool, str]:
        try:
            val = int(text)
        except ValueError:
            return False, "Must be integer"
        mx = self.task_max()
        if val < 1:
            return False, "Min 1"
        if val > mx:
            return False, f"Max {mx}"
        return True, ""

    def validate_time(self, text: str) -> tuple[bool, str]:
        parts = text.split(":")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            return False, "Format HH:MM:SS"
        h, m, s = map(int, parts)
        total = h*3600 + m*60 + s
        _, mx = self.time_bounds()
        if total < 0:
            return False, "Too short"
        if mx is not None and total > mx:
            return False, f"Max {self.time_max_str()}"
        return True, ""


