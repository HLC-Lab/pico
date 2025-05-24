"""
Step 3: Choose MPI implementation.
"""
from textual.widgets import Button, Select, Static
from .base import StepScreen
from config_loader import list_mpi_libs, get_mpi_config


class MPIStep(StepScreen):
    def compose(self):
        yield Static("Select MPI Library", classes="screen-header")
        libs = list_mpi_libs(self.session.environment.name)
        yield Select(
            [(m, m) for m in libs],
            prompt="MPI:",
            id="mpi-select"
        )
        # Next disabled until an MPI is chosen
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        if event.control.id == "mpi-select":
            name = event.value
            self.session.mpi.name = name
            self.session.mpi.config = get_mpi_config(self.session.environment.name, name)
            # enable Next
            self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        if event.button.id == "next":
            self.next(__import__('tui.steps.summary', fromlist=['SummaryStep']).SummaryStep)
