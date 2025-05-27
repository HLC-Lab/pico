# tui/steps/configure.py

from textual.widgets import Static, Select, Button
from .base import StepScreen
from config_loader import list_environments, get_environment_general, get_environment_slurm
from models import PartitionSelection

class ConfigureStep(StepScreen):
    """
    Unified screen: pick environment, partition and QOS.
    Uses helper methods to reset/initialize the dropdowns.
    """

    def compose(self):
        yield Static("Environment:", classes="field-label")
        yield Select([(e, e) for e in list_environments()],
                     prompt="Environment:", id="env-select")

        yield Static("Partition:", classes="field-label")
        yield Select([], prompt="Partition:", id="partition-select", disabled=True)

        yield Static("QOS:", classes="field-label")
        yield Select([], prompt="QOS:", id="qos-select", disabled=True)

        yield Button("Next", id="next", disabled=True)


    def reset_select(self, widget: Select):
        """Clear out options, reset value to blank, disable."""
        widget._options = []
        widget._setup_variables_for_options([])
        widget._setup_options_renderables()
        widget.value = Select.BLANK
        widget.disabled = True


    def on_select_changed(self, event):
        sel = event.control
        from textual.widgets import Select as _Select

        # Grab the widgets once
        part_w = self.query_one("#partition-select", Select)
        qos_w  = self.query_one("#qos-select",      Select)
        next_b = self.query_one("#next",             Button)


        # 1) Real environment picked
        if sel.id == "env-select":
            env = event.value
            # Reset partition and QOS selects
            self.reset_select(part_w)
            self.reset_select(qos_w)
            self.session.partition = PartitionSelection()

            if env is _Select.BLANK:
                next_b.disabled = True
                return

            self.session.environment.name    = env
            self.session.environment.general = get_environment_general(env)

            # SLURM?
            if self.session.environment.general.get("SLURM", False):
                sl = get_environment_slurm(env)
                self.session.environment.slurm = sl

                # Populate and enable partition
                parts = list(sl["PARTITIONS"].keys())
                part_w._options = [(p, p) for p in parts]
                part_w._setup_variables_for_options(part_w._options)
                part_w._setup_options_renderables()
                part_w.disabled = False

                next_b.disabled = True   # must choose partition+QOS
            else:
                # Non-SLURM: skip ahead
                self.reset_select(part_w)
                next_b.disabled = False
            return

        # 3) Partition chosen → populate QOS
        if sel.id == "partition-select":
            p   = event.value
            # Reset QOS select
            self.reset_select(qos_w)
            self.session.partition.qos = ""
            self.session.partition.details = {}

            if p is _Select.BLANK:
                next_b.disabled = True
                return

            cfg = self.session.environment.slurm["PARTITIONS"][p]
            self.session.partition.name    = p
            self.session.partition.details = cfg

            # Populate QOS
            qos_keys = [q for q,o in cfg["QOS"].items() if o.get("required") or q=="default"]
            qos_w._options = [(q, q) for q in qos_keys]
            qos_w._setup_variables_for_options(qos_w._options)
            qos_w._setup_options_renderables()
            qos_w.disabled = False
            return

        # 4) QOS chosen → store details
        if sel.id == "qos-select" and event.value is not _Select.BLANK:
            chosen = event.value
            self.session.partition.qos = chosen
            self.session.partition.qos_details = (
                self.session.partition.details["QOS"][chosen]
            )

        # 5) Finally update Next-button state
        env_ok  = bool(self.session.environment.name)
        slurm   = self.session.environment.general.get("SLURM", False)
        part_ok = bool(self.session.partition.name) if slurm else True
        qos_ok  = bool(self.session.partition.qos)  if slurm else True
        next_b.disabled = not (env_ok and part_ok and qos_ok)


    def on_button_pressed(self, event):
        if event.button.id == "next":
            from tui.steps.node_config import NodeConfigStep
            self.next(NodeConfigStep)


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
