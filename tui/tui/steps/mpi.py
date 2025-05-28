"""
Step 3: Choose MPI implementation.
"""
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Select, Static
from .base import StepScreen
from config_loader import list_mpi_libs, get_mpi_config

class MPIStep(StepScreen):
    def compose(self):
        libs = list_mpi_libs(self.session.environment.name)
        yield Vertical(
            Static("Select MPI Library", classes="screen-header"),
            Select([(m, m) for m in libs], prompt="MPI:", id="mpi-select")
        )
        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Next", id="next", disabled=True),
            classes="button-row"
        )

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
        elif event.button.id == "prev":
            self.session.mpi.name = ""
            self.session.mpi.config = None
            if self.session.environment.general.get("SLURM", False):
                from tui.steps.node_config import NodeConfigStep
                self.prev(NodeConfigStep)
            else:
                from tui.steps.configure import ConfigureStep
                self.prev(ConfigureStep)

    def get_help_desc(self) -> str:
        cfg = self.session.mpi.config or {}
        focused = getattr(self.focused, "id", None)
        if not cfg:
            return "First pick an MPI implementation before proceeding."
        if focused == "mpi-select":
            return cfg.get("desc", "No description.")
        return "Use arrow keys or type to select an MPI library."
