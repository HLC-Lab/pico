# tui/steps/configure.py

from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Select, Button, Switch, Footer, Header
from .base import StepScreen
from config_loader import list_environments, get_environment_general, get_environment_slurm
from models import PartitionSelection

class ConfigureStep(StepScreen):
    """
    Unified screen: pick environment, partition and QOS.
    Uses helper methods to reset/initialize the dropdowns.
    """

    def compose(self):
        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("Environment:", classes="field-label"),
                Select([(e, e) for e in list_environments()], prompt="Environment:", id="env-select")
            ),
            classes="row"
        )

        yield Horizontal(
            Vertical(
                Static("Partition:", classes="field-label"),
                Select([], prompt="Partition:", id="partition-select", disabled=True)
            ),
            Vertical(
                Static("QOS:", classes="field-label"),
                Select([], prompt="QOS:", id="qos-select", disabled=True)
            ),
            classes="row"
        )

        yield Horizontal(
            Vertical(
                Static("Compile Only", classes="field-label"),
                Switch(id="compile-switch", value=False, classes="switch")
            ),
            Vertical(
                Static("Debug Mode", classes="field-label"),
                Switch(id="debug-switch", value=False, classes="switch")
            ),
            Vertical(
                Static("Dry Run Mode", classes="field-label"),
                Switch(id="dry-switch", value=False, classes="switch")
            ),
            classes="tight-switches"
        )

        yield Horizontal(
            Button("Prev", id="prev", disabled=True),
            Button("Next", id="next", disabled=True),
            classes="button-row"
        )

        yield Footer()


    def reset_select(self, widget: Select):
        """Clear out options, reset value to blank, disable."""
        widget._options = []
        widget._setup_variables_for_options([])
        widget._setup_options_renderables()
        widget.value = Select.BLANK
        widget.disabled = True

    def on_select_changed(self, event):
        sel = event.control

        part_w = self.query_one("#partition-select", Select)
        qos_w = self.query_one("#qos-select", Select)
        next_b = self.query_one("#next", Button)

        # Environment changed
        if sel.id == "env-select":
            env = event.value
            self.reset_select(part_w)
            self.reset_select(qos_w)
            self.session.partition = PartitionSelection()

            if env is not Select.BLANK:
                self.session.environment.name = env
                self.session.environment.general = get_environment_general(env)
                
                if self.session.environment.general.get("SLURM", False):
                    # Load SLURM config
                    self.session.environment.slurm = get_environment_slurm(env)
                    part_w.set_options([(p, p) for p in self.session.environment.slurm["PARTITIONS"]])
                    part_w.disabled = False

        # Partition changed
        elif sel.id == "partition-select":
            self.reset_select(qos_w)
            self.session.partition.qos = ""
            
            if event.value is not Select.BLANK:
                # Load partition details
                partition = event.value
                self.session.partition.name = partition
                self.session.partition.details = self.session.environment.slurm["PARTITIONS"][partition]
                
                # Populate QOS
                qos_options = [
                    q for q, opts in self.session.partition.details["QOS"].items()
                    if opts.get("required") or q == "default"
                ]
                qos_w.set_options([(q, q) for q in qos_options])
                qos_w.disabled = False

        # QOS changed
        elif sel.id == "qos-select":
            self.session.partition.qos = event.value
            if event.value is not Select.BLANK:
                self.session.partition.qos_details = (
                    self.session.partition.details["QOS"][event.value]
                )

        # Unified state check
        env_ok = bool(self.session.environment.name)
        slurm = self.session.environment.general.get("SLURM", False)
        part_ok = bool(self.session.partition.name) if slurm else True
        qos_ok = bool(self.session.partition.qos) if slurm else True
        next_b.disabled = not (env_ok and part_ok and qos_ok)


    def on_button_pressed(self, event):
        if event.button.id == "next":
            from tui.steps.node_config import NodeConfigStep
            self.next(NodeConfigStep)

    def on_switch_changed(self, event):
        compile_switch = self.query_one("#compile-switch", Switch)
        dry_switch = self.query_one("#dry-switch", Switch)

        cid = event.control.id
        val = event.value

        if cid == "compile-switch":
            self.session.compile_only = val
            if val:
                self.session.dry_run = False
                dry_switch.value = False
            dry_switch.disabled = val

        elif cid == "debug-switch":
            self.session.debug_mode = val

        elif cid == "dry-switch":
            self.session.dry_run = val
            if val:
                self.session.compile_only = False
                compile_switch.value = False
            compile_switch.disabled = val

    def get_help_desc(self) -> str:
        focused = getattr(self.focused, "id", None)
        if focused == "env-select":
            return self.session.environment.general.get("desc", "Select an environment.")
        if not self.session.environment.general.get("SLURM", False):
            return "Local mode: no partition or QOS needed."
        if not self.session.partition.name:
            return "Choose a partition for SLURM."
        if focused == "partition-select":
            return self.session.partition.details.get("desc", "Select a partition.")
        if not self.session.partition.qos:
            return "Choose a QOS for the selected partition."
        if focused == "qos-select":
            return self.session.partition.qos_details.get("desc", "Select a QOS.")
        return "Complete all selections and press Next."
