"""
Step 1: Choose the compute environment.
"""
from textual.widgets import Button, Select, Static
from .base import StepScreen
from config_loader import list_environments, get_environment_general, get_environment_slurm


class EnvironmentStep(StepScreen):
    def compose(self):
        """Render header, dropdown, and next button."""
        yield Static("Select Environment", classes="screen-header")
        envs = list_environments()
        yield Select(
            [(env, env) for env in envs],
            prompt="Environment:",
            id="env-select"
        )
        # Next starts disabled until an environment is chosen
        yield Button("Next", id="next", disabled=True)

    def on_select_changed(self, event):
        """Enable Next and load environment when dropdown changes."""
        if event.control.id == "env-select":
            env = event.value
            self.session.environment.name = env
            self.session.environment.general = get_environment_general(env)
            if self.session.environment.general.get("SLURM", False):
                self.session.environment.slurm = get_environment_slurm(env)
            # enable Next button
            self.query_one("#next").disabled = False

    def on_button_pressed(self, event):
        """Navigate to partition or MPI step."""
        if event.button.id == "next":
            if self.session.environment.general.get("SLURM", False):
                self.next(self.__module__.replace('environment', 'partition').title() and __import__('tui.steps.partition', fromlist=['PartitionStep']).PartitionStep)
            else:
                self.next(__import__('tui.steps.mpi', fromlist=['MPIStep']).MPIStep)
