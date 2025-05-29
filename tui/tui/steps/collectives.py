# tui/steps/collectives.py

"""
Step after MPI: pick which collective algorithms to benchmark.
"""

from textual.containers import Horizontal
from textual.widgets import Static, Checkbox, Button
from textual.app import ComposeResult
from tui.steps.base import StepScreen
from config_loader import list_algorithms


class CollectiveStep(StepScreen):
    """Choose one or more collectives from config/algorithms/<MPI_LIB>."""

    def compose(self) -> ComposeResult:
        yield Static("Select collective operations", classes="screen-header")

        # List out each JSON filename (without extension)
        algs = list_algorithms(self.session.mpi.name)
        for alg in algs:
            # label and value both the alg name
            yield Checkbox(label=alg, id=f"alg-{alg}")

        # Navigation
        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Next", id="next", disabled=True),
            classes="button-row"
        )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """
        Enable Next if at least one checkbox is checked.
        """
        any_checked = any(
            cb.value
            for cb in self.query(Checkbox)
        )
        self.query_one("#next", Button).disabled = not any_checked

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle navigation buttons.
        """
        if event.button.id == "next":
            # Collect checked algorithms
            self.session.algorithms = [
                cb.label
                for cb in self.query(Checkbox)
                if cb.value
            ]
            from tui.steps.summary import SummaryStep
            self.next(SummaryStep)
        elif event.button.id == "prev":
            from tui.steps.mpi import MPIStep
            self.prev(MPIStep)


    def get_help_desc(self) -> str:
        return (
            "Toggle one or more collective operations to include in the benchmark. "
            "Use space or enter to check/uncheck, then 'n' or Next to proceed."
        )
