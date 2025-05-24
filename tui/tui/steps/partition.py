"""
Step 2: Choose SLURM partition and QOS.
"""
from textual.widgets import Select, Button, Static
from .base import StepScreen


class PartitionStep(StepScreen):
    def compose(self):
        """Render partition dropdown, QOS placeholder, and next button."""
        yield Static("Select Partition", classes="screen-header")
        parts = list(self.session.environment.slurm["PARTITIONS"].keys())
        yield Select(
            [(p, p) for p in parts],
            prompt="Partition:",
            id="partition-select"
        )

        yield Static("Select QOS", classes="screen-header", id="qos-label")
        # Placeholder QOS dropdown (no ID)
        yield Select([], prompt="QOS:")

        # Next disabled until both partition+QOS chosen
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        """Handle partition and QOS changes and enable Next appropriately."""
        # Partition chosen: swap in real QOS dropdown
        if event.control.id == "partition-select":
            p = event.value
            part_cfg = self.session.environment.slurm["PARTITIONS"][p]
            self.session.partition.name = p
            self.session.partition.details = part_cfg

            qos_keys = [
                q for q, opts in part_cfg["QOS"].items()
                if opts.get("required", False) or q == "default"
            ]
            new_qos = Select(
                [(q, q) for q in qos_keys],
                prompt="QOS:"
            )
            # Remove placeholder (always 2nd Select)
            placeholder = self.query(Select)[1]
            placeholder.remove()
            # Mount the real dropdown after the label
            self.mount(new_qos, after="#qos-label")

        # QOS selected: store and maybe enable Next
        elif isinstance(event.control, Select) and event.control.prompt == "QOS:":
            self.session.partition.qos = event.value

        # Enable Next only when both fields set
        if self.session.partition.name and self.session.partition.qos:
            self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        """Proceed to MPI selection."""
        if event.button.id == "next":
            self.next(__import__('tui.steps.mpi', fromlist=['MPIStep']).MPIStep)
