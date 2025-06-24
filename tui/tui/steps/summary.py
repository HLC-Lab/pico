import json
from textual.widgets import Button, Static, Header, Footer
from textual.containers import Horizontal, Container
from textual.app import ComposeResult
from tui.steps.base import StepScreen

class SummaryStep(StepScreen):
    __summary: dict

    def compose(self) -> ComposeResult:

        yield Header(show_clock=True)
        yield Static("Summary", classes="screen-header")

        self.__summary = self.session.to_dict()
        yield Container(
            Static(json.dumps(self.__summary, indent=2), markup=False, classes="summary-box"),
            classes="summary-container"
        )

        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Finish", id="finish"),
            classes="button-row"
        )

        yield Footer()

    # TODO:
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev":
            from tui.steps.algorithms import AlgorithmsStep
            self.prev(AlgorithmsStep)
        elif event.button.id == "finish":
            pass


    def get_help_desc(self):
        return "a","b"
