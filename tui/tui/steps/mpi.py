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
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        from textual.widgets import Select as _Select
        if event.control.id == "mpi-select":
            if event.value is _Select.BLANK:
                self.query_one("#next").disabled = True
                return
            name = event.value
            self.session.mpi.name   = name
            self.session.mpi.config = get_mpi_config(self.session.environment.name, name)
            self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        if event.button.id == "next":
            from tui.steps.summary import SummaryStep
            self.next(SummaryStep)

    def get_help_desc(self) -> str:
        cfg = self.session.mpi.config or {}
        focused = getattr(self.focused, "id", None)
        if not cfg:
            return "First pick an MPI implementation before proceeding."
        if focused == "mpi-select":
            return cfg.get("desc", "No description.")
        return "Use arrow keys or type to select an MPI library."
