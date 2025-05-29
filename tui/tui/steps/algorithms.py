# tui/steps/algorithm_selection.py

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Checkbox, TabbedContent, TabPane
from tui.steps.base import StepScreen
import json
import os

class AlgorithmSelectionStep(StepScreen):
    """Screen to select algorithms for each chosen collective."""

    def compose(self) -> ComposeResult:
        yield Static("Select Algorithms for Each Collective", classes="screen-header")

        # Create TabbedContent with a TabPane for each selected collective
        with TabbedContent():
            for collective in self.session.algorithms:
                pane_id = f"tab-{collective}"
                collective_str = str(collective)
                with TabPane(title=collective_str.capitalize(), id=pane_id):
                    # Load algorithms from the corresponding JSON file
                    algos = self.load_algorithms(self.session.mpi.name, collective)
                    for key, meta in algos.items():
                        label = f"{key} ({meta.get('cvar', '')})"
                        yield Checkbox(label=label, id=f"{collective}-{key}")

        # Navigation buttons
        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Next", id="next"),
            classes="button-row"
        )

    def load_algorithms(self, mpi_name: str, collective: str) -> dict:
        """Load algorithms from the specified JSON file."""
        path = os.path.join("config", "algorithms", mpi_name, f"{collective}.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle navigation button presses."""
        if event.button.id == "next":
            # Collect selected algorithms
            selected_algorithms = {}
            for collective in self.session.algorithms:
                selected = []
                for cb in self.query(Checkbox).results():
                    if cb.id and cb.id.startswith(f"{collective}-") and cb.value:
                        selected.append(cb.label)
                selected_algorithms[collective] = selected
            self.session.selected_algorithms = selected_algorithms

            # Proceed to the next step (e.g., SummaryStep)
            from tui.steps.summary import SummaryStep
            self.next(SummaryStep)

        elif event.button.id == "prev":
            # Return to the previous step (e.g., CollectiveStep)
            from tui.steps.mpi_collectives import MPICollectivesStep
            self.prev(MPICollectivesStep)

    def get_help_desc(self) -> str:
        return (
            "Select the algorithms you wish to benchmark for each collective operation. "
            "Use the tabs to navigate between collectives and check the desired algorithms."
        )
