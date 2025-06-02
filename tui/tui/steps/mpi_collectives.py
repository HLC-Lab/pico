"""
Step 3: Choose MPI implementation and which collectives to benchmark.
"""

from textual.containers import Vertical, Horizontal
from textual.widgets import Button, Select, Static, Checkbox, Switch, Header, Footer
from textual.types import NoSelection
from textual.app import ComposeResult
from tui.steps.base import StepScreen
from models import MPILibrarySelection, CollectiveSelection
from typing import List
from config_loader import list_mpi_libs, get_mpi_config, list_algorithms


class MPICollectivesStep(StepScreen):
    """
    Unified screen to pick an MPI library and choose one or more collectives.
    """

    def compose(self) -> ComposeResult:
        # yield Static("MPI Selection & Collectives", classes="field-label")
        # MPI dropdown
        libs = list_mpi_libs(self.session.environment.name)

        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("MPI Selection", classes="field-label"),
                Select([(m, m) for m in libs], prompt="Select MPI Library", id="mpi-select"),
                classes="first-big"
            ),
            Vertical(
                Horizontal(
                    Vertical(
                        Static("Use also LibPICO?", classes="field-label"),
                        Switch(id="compile-switch")
                    ),
                    Vertical(
                        Static("Placeholder", classes="field-label"),
                        Switch(id="debug-switch", disabled=True)
                    ),
                    classes="tight-switches"
                )
            ),
            classes="row"
        )

        # Placeholder for collectives — to be filled dynamically
        self.collectives_container = Vertical(id="collectives", classes="collectives-container")
        yield self.collectives_container

        yield self.navigation_buttons()

        yield Footer()

    def on_mount(self) -> None:
        self.session.mpi = MPILibrarySelection()
        self.session.collectives = []

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mpi-select":
            self.session.mpi = MPILibrarySelection()
            self.session.collectives = []
            value = event.value
            if isinstance(value, NoSelection) or not isinstance(value, str) or value == "":
                self.query_one("#next", Button).disabled = True
                self.collectives_container.remove_children()
                return

            name = value
            self.session.mpi.name = name
            self.session.mpi.config = get_mpi_config(self.session.environment.name, name)
            self.session.mpi.type = self.session.mpi.config.get("type", "unknown")

            if not self.session.mpi.config or "type" not in self.session.mpi.config:
                self.notify("Invalid MPI configuration", severity="error")
                self.query_one("#next", Button).disabled = True
                self.collectives_container.remove_children()
                return
            

            algs = list_algorithms(self.session.mpi.type)
            self.collectives_container.remove_children()
            for alg in algs:
                checkbox_id = f"alg-{name}-{alg}".replace(" ", "_").replace(".", "_")
                self.collectives_container.mount(Checkbox(label=alg, id=checkbox_id))

            self._update_next_button_state()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._update_next_button_state()

    def _update_next_button_state(self) -> None:
        mpi_ok = bool(self.session.mpi.name)
        algs_ok = any(cb.value for cb in self.query(Checkbox))
        self.query_one("#next", Button).disabled = not (mpi_ok and algs_ok)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            # Store selected algorithms
            for cb in self.query(Checkbox):
                if cb.value:
                    self.session.collectives.append(CollectiveSelection(self.session.mpi.name, cb.label))
            from tui.steps.algorithms import AlgorithmSelectionStep
            self.next(AlgorithmSelectionStep)

        elif event.button.id == "prev":
            from tui.steps.node_config import NodeConfigStep
            self.prev(NodeConfigStep)

    def get_help_desc(self) -> str:
        focused = getattr(self.focused, "id", None)
        if focused == "mpi-select":
            return self.session.mpi.config.get("desc", "Pick an MPI implementation.")
        if focused and focused.startswith("alg-"):
            return "Check one or more collectives to benchmark."
        return "Select MPI, check collectives, then proceed."
