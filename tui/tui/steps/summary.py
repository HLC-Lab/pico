"""
Final step: display a summary of the selected configuration.
"""
import json
from pathlib import Path
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static
from textual.containers import Vertical, Horizontal, Container
from textual.app import ComposeResult
from tui.steps.base import StepScreen


class SavePrompt(ModalScreen[str | None]):
    """A modal screen to ask for a filename to save the configuration."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Enter filename to save configuration (e.g. config.json):", id="save-label"),
            Input(placeholder="my_config.json", id="save-filename"),
            classes="save-dialog"
        )
        yield Horizontal(
            Button("Confirm", id="confirm-save"),
            Button("Cancel", id="cancel-save"),
            classes="button-row"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-save":
            input_widget = self.query_one("#save-filename", Input)
            filename = input_widget.value.strip()

            if filename:
                if not filename.endswith(".json"):
                    filename += ".json"
                self.dismiss(filename)
            else:
                self.dismiss(None)

        elif event.button.id == "cancel-save":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SummaryStep(StepScreen):
    """Final step: shows summary and offers save option."""

    def compose(self) -> ComposeResult:
        yield Static("Summary", classes="screen-header")

        summary = self.get_summary()
        yield Container(
            Static(json.dumps(summary, indent=2), classes="summary-box"),
            classes="summary-container"
        )

        yield Horizontal(
            Button("Save", id="save-button"),
            Button("Finish", id="finish-button"),
            classes="button-row"
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-button":
            filename = await self.app.push_screen(SavePrompt())

            if filename:
                root_path = Path(__file__).parent.parent.parent.parent
                full_path = root_path / filename

                try:
                    with open(full_path, "w") as f:
                        json.dump(self.session.to_dict(), f, indent=2)
                    self.app.bell()  # Optional confirmation
                except Exception as e:
                    print(f"Failed to save file: {e}")

        elif event.button.id == "finish-button":
            self.app.exit()

    def get_summary(self) -> dict:
        if self.session.environment.general.get("SLURM") is False:
            return {
                'environment': self.session.environment.general,
                'mpi': self.session.mpi.config,
                'nodes': self.session.nodes
            }
        else:
            return {
                'environment': self.session.environment.general,
                'partition': {
                    'name': self.session.partition.name,
                    **{k: v for k, v in self.session.partition.details.items() if k != "QOS"}
                },
                'qos': {
                    'name': self.session.partition.qos,
                    **(self.session.partition.qos_details or {})
                },
                'mpi': self.session.mpi.config,
                'nodes': self.session.nodes,
                'tasks_per_node': self.session.tasks_per_node,
                'test_time': self.session.test_time or "",
            }

    def get_help_desc(self) -> str:
        return "Review the configuration, optionally save it, and press Finish to exit."
