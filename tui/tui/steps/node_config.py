# tui/steps/node_config.py

"""
Step 3: Choose number of nodes, (tasks per node), and (test time).
Tasks/time only when SLURM is enabled.
"""

from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, Button, Label, Switch
from .base import StepScreen


DEFAULT_TIME = "00:30:00"


class NodeConfigStep(StepScreen):
    
    @property
    def slurm_enabled(self) -> bool:
        """Check if SLURM is enabled in the current session."""
        return self.session.environment.general.get("SLURM", False)
    
    @property
    def qos_config(self) -> dict:
        """Get QOS configuration details."""
        return self.session.partition.qos_details or {}
    
    @property 
    def partition_config(self) -> dict:
        """Get partition configuration details."""
        return self.session.partition.details or {}

    def compose(self):
        """Render inputs and Next button."""
        min_nodes = self._get_min_nodes()
        max_nodes = self._get_max_nodes()
        max_tasks = self._get_max_tasks()
        max_time_str = self._get_max_time_str()

        yield Horizontal(
            Vertical(
                Static("Number of Nodes" if self.slurm_enabled else "Number of Tasks", classes="field-label"),
                Input(placeholder=f"min {min_nodes}, max {max_nodes}", id="nodes-input"),
                Label("", id="nodes-error", classes="error"),
                classes="field",
            ),
            Vertical(
                Static("Exclude?", classes="field-label"),
                Switch(id="exclude-switch", value=False, disabled=not self.slurm_enabled),
                classes="switch-col",
            ),
            Vertical(
                Static("Excluded Nodes", classes="field-label"),
                Input(placeholder="What nodes do you want to exclude?", id="excluded-nodes", disabled=True),
                Label("", id="excluded-nodes-error", classes="error"),
                classes="field",
            ),
            classes="row",
        )

        yield Horizontal(
            Vertical(
                Static("Tasks per Node", classes="field-label"),
                Input(placeholder=f"1–{max_tasks}", id="tasks-input", value="1", disabled=not self.slurm_enabled),
                Label("", id="tasks-error", classes="error")
            ),
            Vertical(
                Static("Test Time", classes="field-label"),
                Input(placeholder=f"HH:MM:SS (max {max_time_str})", id="time-input", value=DEFAULT_TIME, disabled=not self.slurm_enabled),
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
        self._validate_field(event.input.id, event.value)
        self._update_next_button()

    def on_switch_changed(self, event):
        """Handle exclude switch toggle."""
        if event.switch.id == "exclude-switch":
            excluded_input = self.query_one("#excluded-nodes", Input)
            excluded_input.disabled = not event.value
            if not event.value:
                excluded_input.value = ""

    def on_button_pressed(self, event):
        """Store values and proceed."""
        if event.button.id == "next":
            self._save_values()
            from tui.steps.mpi_collectives import MPICollectivesStep
            self.next(MPICollectivesStep)
        elif event.button.id == "prev":
            self._reset_values()
            from tui.steps.configure import ConfigureStep
            self.prev(ConfigureStep)

    def get_help_desc(self) -> str:
        """Contextual help based on focused input."""
        focus_id = getattr(self.focused, "id", "")
        
        help_map = {
            "nodes-input": f"Nodes: {self._get_min_nodes()}–{self._get_max_nodes()}",
            "tasks-input": f"Tasks/node: 1–{self._get_max_tasks()}",
            "time-input": f"Time (max {self._get_max_time_str()})",
        }
        
        return help_map.get(focus_id, "Configure run resources before MPI.")

    # ─── Configuration Getters ─────────────────────────────────────────────────

    def _get_min_nodes(self) -> int:
        """Get minimum number of nodes based on configuration."""
        if not self.slurm_enabled:
            return 2
        return int(self.qos_config.get("QOS_MIN_NODES", 2))

    def _get_max_nodes(self) -> str:
        """Get maximum number of nodes as string for display."""
        if not self.slurm_enabled:
            return "∞"
        max_nodes = self.qos_config.get("QOS_MAX_NODES")
        return str(max_nodes) if max_nodes is not None else "∞"

    def _get_max_tasks(self) -> int:
        """Get maximum tasks per node."""
        if not self.slurm_enabled:
            return 1
        cpus = self.partition_config.get("PARTITION_CPUS_PER_NODE")
        return int(cpus) if cpus is not None else 1

    def _get_max_time_str(self) -> str:
        """Get maximum time as string for display."""
        if not self.slurm_enabled:
            return "∞"
        return self.qos_config.get("QOS_MAX_TIME", "∞")

    def _get_max_time_seconds(self) -> int | None:
        """Get maximum time in seconds for validation."""
        if not self.slurm_enabled:
            return None
        
        max_time = self.qos_config.get("QOS_MAX_TIME")
        if not max_time:
            return None
            
        try:
            # Parse format like "7-00:00:00" or "01:30:00"
            if "-" in max_time:
                days_str, time_str = max_time.split("-", 1)
                days = int(days_str)
            else:
                days = 0
                time_str = max_time
                
            h, m, s = map(int, time_str.split(":"))
            return days * 86400 + h * 3600 + m * 60 + s
        except (ValueError, IndexError):
            return None

    # ─── Validation ─────────────────────────────────────────────────────────────

    def _validate_field(self, field_id: str, value: str) -> None:
        """Validate a single field and update its error label."""
        validators = {
            "nodes-input": self._validate_nodes,
            "tasks-input": self._validate_tasks,
            "time-input": self._validate_time,
        }
        
        if field_id in validators:
            _, error_msg = validators[field_id](value)
            error_label_id = field_id.replace("-input", "-error")
            self.query_one(f"#{error_label_id}", Label).update(error_msg)

    def _validate_nodes(self, text: str) -> tuple[bool, str]:
        """Validate number of nodes."""
        try:
            value = int(text)
        except ValueError:
            return False, "Must be integer"
        
        min_nodes = self._get_min_nodes()
        if value < min_nodes:
            return False, f"Min {min_nodes}"
        
        if self.slurm_enabled:
            max_nodes_raw = self.qos_config.get("QOS_MAX_NODES")
            if max_nodes_raw is not None and value > int(max_nodes_raw):
                return False, f"Max {max_nodes_raw}"
        
        return True, ""

    def _validate_tasks(self, text: str) -> tuple[bool, str]:
        """Validate tasks per node."""
        if not self.slurm_enabled:
            return True, ""
            
        try:
            value = int(text)
        except ValueError:
            return False, "Must be integer"
        
        if value < 1:
            return False, "Min 1"
        
        max_tasks = self._get_max_tasks()
        if value > max_tasks:
            return False, f"Max {max_tasks}"
        
        return True, ""

    def _validate_time(self, text: str) -> tuple[bool, str]:
        """Validate time format and limits."""
        if not self.slurm_enabled:
            return True, ""
        
        # Validate format
        parts = text.split(":")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            return False, "Format HH:MM:SS"
        
        try:
            h, m, s = map(int, parts)
            total_seconds = h * 3600 + m * 60 + s
        except ValueError:
            return False, "Invalid time values"
        
        if total_seconds <= 0:
            return False, "Must be positive"
        
        max_seconds = self._get_max_time_seconds()
        if max_seconds is not None and total_seconds > max_seconds:
            return False, f"Max {self._get_max_time_str()}"
        
        return True, ""

    def _update_next_button(self) -> None:
        """Enable/disable Next button based on all field validations."""
        nodes_valid, _ = self._validate_nodes(self.query_one("#nodes-input", Input).value)
        
        if self.slurm_enabled:
            tasks_valid, _ = self._validate_tasks(self.query_one("#tasks-input", Input).value)
            time_valid, _ = self._validate_time(self.query_one("#time-input", Input).value)
            all_valid = nodes_valid and tasks_valid and time_valid
        else:
            all_valid = nodes_valid
        
        self.query_one("#next", Button).disabled = not all_valid

    # ─── State Management ───────────────────────────────────────────────────────

    def _save_values(self) -> None:
        """Save current form values to session."""
        self.session.nodes = int(self.query_one("#nodes-input", Input).value)
        
        if self.slurm_enabled:
            self.session.tasks_per_node = int(self.query_one("#tasks-input", Input).value)
            self.session.test_time = self.query_one("#time-input", Input).value
        else:
            self.session.tasks_per_node = 1
            self.session.test_time = DEFAULT_TIME

    def _reset_values(self) -> None:
        """Reset session values to defaults."""
        self.session.nodes = 0
        self.session.tasks_per_node = 1
        self.session.test_time = DEFAULT_TIME
