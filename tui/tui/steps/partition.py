"""
Step 2: Choose SLURM partition and QOS.
"""
from textual.widgets import Select, Button, Static
from .base import StepScreen

class PartitionStep(StepScreen):
    def compose(self):
        yield Static("Select Partition", classes="screen-header")
        parts = list(self.session.environment.slurm["PARTITIONS"].keys())
        yield Select(
            [(p, p) for p in parts],
            prompt="Partition:",
            id="partition-select"
        )

        yield Static("Select QOS", classes="screen-header", id="qos-label")
        yield Select([], prompt="QOS:")  # placeholder, no id

        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        from textual.widgets import Select as _Select
        # Partition chosen
        if event.control.id == "partition-select":
            if event.value is _Select.BLANK:
                self.query_one("#next").disabled = True
                return
            p = event.value
            cfg = self.session.environment.slurm["PARTITIONS"][p]
            self.session.partition.name    = p
            self.session.partition.details = cfg

            qos_keys = [
                q for q, opts in cfg["QOS"].items()
                if opts.get("required", False) or q == "default"
            ]
            new_qos = Select(
                [(q, q) for q in qos_keys],
                prompt="QOS:"
            )
            placeholder = self.query(Select)[1]
            placeholder.remove()
            self.mount(new_qos, after="#qos-label")

        # QOS chosen
        elif isinstance(event.control, Select) and event.control.prompt == "QOS:":
            if event.value is _Select.BLANK:
                self.query_one("#next").disabled = True
                return
            self.session.partition.qos = event.value

        # Enable Next?
        if self.session.partition.name and self.session.partition.qos:
            self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        if event.button.id == "next":
            from tui.steps.mpi import MPIStep
            self.next(MPIStep)

    def get_help_desc(self) -> str:
        # no partition yet
        if not self.session.partition.details:
            return "First choose a partition, then pick a QOS."
        focused = getattr(self.focused, "id", None)
        # partition selected, QOS not yet
        if not self.session.partition.qos:
            if focused == "partition-select":
                return self.session.partition.details.get("desc", "No description.")
            return "Pick a QOS for your selected partition."
        # both done: show the chosen QOS desc
        qos = self.session.partition.qos
        return (
            self.session.partition.details["QOS"]
            .get(qos, {})
            .get("desc", "No description.")
        )
