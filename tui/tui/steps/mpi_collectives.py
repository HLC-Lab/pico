
"""
Step 3: Choose MPI implementation and which collectives to benchmark.
"""

from textual.containers import Vertical, Horizontal
from textual.widgets import Button, Select, Static, Checkbox, Switch
from textual.types import NoSelection
from textual.app import ComposeResult
from tui.steps.base import StepScreen
from config_loader import list_mpi_libs, get_mpi_config, list_algorithms


class MPICollectivesStep(StepScreen):
    """
    Unified screen to pick an MPI library and choose one or more collectives.
    """

    def compose(self) -> ComposeResult:
        # yield Static("MPI Selection & Collectives", classes="field-label")
        # MPI dropdown
        libs = list_mpi_libs(self.session.environment.name)
        yield Horizontal(
            Vertical(
                Static("MPI Selection", classes="field-label"),
                Select([(m, m) for m in libs], prompt="Select MPI Library", id="mpi-select"),
                classes="first-big"
            ),
            Vertical(
                Horizontal(
                    Vertical(
                        Static("Collectives", classes="field-label"),
                        Switch(id="compile-switch", name="Compile Only")
                    ),
                    Vertical(
                        Static("Collectives", classes="field-label"),
                        Switch(id="debug-switch", name="Debug Mode")
                    ),
                    classes="tight-switches"
                )
            ),
            classes="row"
        )

        # Placeholder for collectives — to be filled dynamically
        self.collectives_container = Vertical(id="collectives", classes="collectives-container")
        yield self.collectives_container

        # Navigation buttons
        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Next", id="next", disabled=True),
            classes="button-row"
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mpi-select":
            value = event.value
            if isinstance(value, NoSelection) or not isinstance(value, str) or value == "":
                self.query_one("#next", Button).disabled = True
                self.collectives_container.remove_children()
                return

            name = value  # safely treat as str here
            self.session.mpi.name = name
            self.session.mpi.config = get_mpi_config(self.session.environment.name, name)

            algs = list_algorithms(name)
            self.collectives_container.remove_children()
            for alg in algs:
                self.collectives_container.mount(Checkbox(label=alg, id=f"alg-{alg}"))

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
            self.session.algorithms = [
                cb.label for cb in self.query(Checkbox) if cb.value
            ]
            from tui.steps.algorithms import AlgorithmSelectionStep
            self.next(AlgorithmSelectionStep)

        elif event.button.id == "prev":
            # Clear MPI info
            self.session.mpi.name = ""
            self.session.mpi.config = None
            self.session.algorithms = []
            if self.session.environment.general.get("SLURM", False):
                from tui.steps.node_config import NodeConfigStep
                self.prev(NodeConfigStep)
            else:
                from tui.steps.configure import ConfigureStep
                self.prev(ConfigureStep)

    def get_help_desc(self) -> str:
        focused = getattr(self.focused, "id", None)
        if focused == "mpi-select":
            return self.session.mpi.config.get("desc", "Pick an MPI implementation.")
        if focused and focused.startswith("alg-"):
            return "Check one or more collectives to benchmark."
        return "Select MPI, check collectives, then proceed."
