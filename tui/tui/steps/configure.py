# tui/steps/configure.py

from textual.widgets import Static, Select, Button
from .base import StepScreen
from config_loader import list_environments, get_environment_general, get_environment_slurm

class ConfigureStep(StepScreen):
    """
    Unified screen: pick environment, partition and QOS.
    Mutates each Select widget in place.
    """
    def compose(self):
        # Environment row
        yield Static("Environment:", classes="field-label")
        yield Select(
            [(e, e) for e in list_environments()],
            prompt="Env:",
            id="env-select"
        )

        # Partition row
        yield Static("Partition:", classes="field-label")
        yield Select(
            [],                    # empty until env chosen
            prompt="Partition:",
            id="partition-select",
            disabled=True
        )

        # QOS row
        yield Static("QOS:", classes="field-label")
        yield Select(
            [],                    # empty until partition chosen
            prompt="QOS:",
            id="qos-select",
            disabled=True
        )

        # Next button
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        sel = event.control
        from textual.widgets import Select as _Select

        # ENV selected → load SLURM + populate partition-select
        if sel.id == "env-select" and event.value is not _Select.BLANK:
            env = event.value
            self.session.environment.name    = env
            self.session.environment.general = get_environment_general(env)

            part_widget = self.query_one("#partition-select", Select)
            if self.session.environment.general.get("SLURM", False):
                slurm = get_environment_slurm(env)
                self.session.environment.slurm = slurm

                parts = list(slurm["PARTITIONS"].keys())
                # in-place hack to refresh options
                part_widget._options = [(p, p) for p in parts]
                part_widget._setup_variables_for_options(part_widget._options)
                part_widget._setup_options_renderables()
                part_widget.disabled = False
            else:
                # no SLURM: skip partition/QOS
                self.query_one("#next", Button).disabled = False
                return

        # Partition selected → populate qos-select
        elif sel.id == "partition-select" and event.value is not _Select.BLANK:
            p = event.value
            cfg = self.session.environment.slurm["PARTITIONS"][p]
            self.session.partition.name    = p
            self.session.partition.details = cfg

            qos_widget = self.query_one("#qos-select", Select)
            qos_keys   = [
                q for q, opts in cfg["QOS"].items()
                if opts.get("required", False) or q == "default"
            ]
            qos_widget._options = [(q, q) for q in qos_keys]
            qos_widget._setup_variables_for_options(qos_widget._options)
            qos_widget._setup_options_renderables()
            qos_widget.disabled = False

        # QOS selected → store
        elif sel.id == "qos-select" and event.value is not _Select.BLANK:
            self.session.partition.qos = event.value

        # Enable Next?
        env_ok  = bool(self.session.environment.name)
        slurm   = self.session.environment.general.get("SLURM", False)
        part_ok = bool(self.session.partition.name) if slurm else True
        qos_ok  = bool(self.session.partition.qos)  if slurm else True

        self.query_one("#next", Button).disabled = not (env_ok and part_ok and qos_ok)

    def on_button_pressed(self, event):
        if event.button.id == "next":
            from tui.steps.mpi import MPIStep
            self.next(MPIStep)

    def get_help_desc(self) -> str:
        focused = getattr(self.focused, "id", None)
        if focused == "env-select":
            return self.session.environment.general.get("desc", "Select an environment.")
        if not self.session.partition.details:
            return "First choose a partition after selecting an environment."
        if focused == "partition-select":
            return self.session.partition.details.get("desc", "Select a partition.")
        if not self.session.partition.qos:
            return "Now pick a QOS for that partition."
        if focused == "qos-select":
            key = self.session.partition.qos
            return self.session.partition.details["QOS"][key].get("desc", "Select a QOS.")
        return "Complete all selections and press Next."
