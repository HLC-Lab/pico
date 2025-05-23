from textual.widgets import Select, Button, Static
from .base import StepScreen

class PartitionStep(StepScreen):
    def compose(self):
        yield Static("Select Partition", classes="screen-header")
        parts = list(self.session.environment.slurm["PARTITIONS"].keys())
        yield Select([(p, p) for p in parts], prompt="Partition:", id="partition-select")
        yield Static("Select QOS", classes="screen-header", id="qos-label")
        yield Select([], prompt="QOS:")
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        if event.control.id == "partition-select":
            p = event.value
            part = self.session.environment.slurm["PARTITIONS"][p]
            self.session.partition.name = p
            self.session.partition.details = part

            qos_keys = [
                q for q, opts in part["QOS"].items()
                if opts.get("required", False) or q == "default"
            ]
            new_qos = Select([(q, q) for q in qos_keys], prompt="QOS:")

            placeholder = self.query(Select)[1]
            placeholder.remove()

            self.mount(new_qos, after="#qos-label")

        elif isinstance(event.control, Select) and event.control.prompt == "QOS:":
            self.session.partition.qos = event.value

        if self.session.partition.name and self.session.partition.qos:
            self.query_one("#next", Button).disabled = False

    def on_button_pressed(self, event):
        if event.button.id == "next":
            self.next(__import__("tui.steps.mpi", fromlist=["MPIStep"]).MPIStep)
