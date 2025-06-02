# tui/steps/algorithm_selection.py
from textual import events
from textual.app import ComposeResult
from textual.widgets import Static, Button, Checkbox, TabbedContent, TabPane, Header, Footer
from tui.steps.base import StepScreen
from config_loader import get_algorithm_config


class AlgorithmSelectionStep(StepScreen):
    """Screen to select algorithms for each chosen collective."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        yield Static("Select Algorithms for Each Collective", classes="screen-header")

        # Create TabbedContent with a TabPane for each selected collective
        with TabbedContent():
            for idx, coll in enumerate(self.session.collectives):
                collective = coll.collective_type
                pane_id = f"tab-{collective}"
                collective_str = str(collective)
                with TabPane(title=f"({idx+1}) {collective_str.capitalize()}", id=pane_id):
                    # Load algorithms from the corresponding JSON file
                    algos = get_algorithm_config(self.session.mpi.type, collective)
                    for key, meta in algos.items():
                        label = f"{key} ({meta.get('cvar', '')})"
                        yield Checkbox(label=label, id=f"{collective}-{key}")


        yield self.navigation_buttons()

        yield Footer()

    async def on_key(self, event: events.Key) -> None:
        """
        If the user presses a digit (1, 2, 3, …), switch to that tab index.

        - Gather all TabPane children via tabs.query(TabPane).
        - Convert the digit into a zero-based index.
        - If valid, set tabs.active to that TabPane’s id (string).
        - Then move keyboard focus into the first Checkbox of that pane.
        """
        if not event.key.isdigit():
            return

        idx = int(event.key) - 1
        tabs = self.query_one(TabbedContent)

        # Grab a list of all TabPane children
        panes = list(tabs.query(TabPane))
        if 0 <= idx < len(panes):
            pane = panes[idx]
            pane_id = pane.id
            if pane_id is not None:
                # 1) Activate the chosen tab
                tabs.active = pane_id

                # 2) Immediately move focus into that pane’s first Checkbox
                first_cb = pane.query_one(Checkbox)
                if first_cb:
                    first_cb.focus()

                # 3) Stop further propagation so nothing else overrides it
                event.stop()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._update_next_button_state()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle navigation button presses."""
        if event.button.id == "next":
            # Collect selected algorithms
            selected_algorithms = {}
            for coll in self.session.collectives:
                collective = coll.collective_type
                selected = []
                for cb in self.query(Checkbox).results():
                    if cb.id and cb.id.startswith(f"{collective}-") and cb.value:
                        selected.append(cb.label)
                selected_algorithms[collective] = selected
                coll.algo_list = selected_algorithms

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


    def _update_next_button_state(self) -> None:
        all_selected = True
        for coll in self.session.collectives:
            collective = coll.collective_type
            found = any(
                cb.value
                for cb in self.query(Checkbox)
                if cb.id and cb.id.startswith(f"{collective}-")
            )
            if not found:
                all_selected = False
                break

        self.query_one("#next", Button).disabled = not all_selected
