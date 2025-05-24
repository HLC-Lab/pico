"""
Step 1: Choose the compute environment.
"""
from textual.widgets import Button, Select, Static
from .base import StepScreen
from config_loader import list_environments, get_environment_general, get_environment_slurm

class EnvironmentStep(StepScreen):
    def compose(self):
        yield Static("Select Environment", classes="screen-header")
        envs = list_environments()
        yield Select(
            [(e, e) for e in envs],
            prompt="Environment:",
            id="env-select"
        )
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        from textual.widgets import Select as _Select
        if event.control.id == "env-select":
            if event.value is _Select.BLANK:
                self.query_one("#next").disabled = True
                return
            env = event.value
            self.session.environment.name    = env
            self.session.environment.general = get_environment_general(env)
            if self.session.environment.general.get("SLURM", False):
                self.session.environment.slurm = get_environment_slurm(env)
            self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        if event.button.id == "next":
            if self.session.environment.general.get("SLURM", False):
                from tui.steps.partition import PartitionStep
                self.next(PartitionStep)
            else:
                from tui.steps.mpi import MPIStep
                self.next(MPIStep)

    def get_help_desc(self) -> str:
        gen = self.session.environment.general or {}
        focused = getattr(self.focused, "id", None)
        if not gen:
            return "First choose an environment before proceeding."
        if focused == "env-select":
            return gen.get("desc", "No description available.")
        return "Use the arrow keys or type to select an environment."
