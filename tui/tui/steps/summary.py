import json
from textual.widgets import Button, Static, Header, Footer, RichLog
from textual.containers import Horizontal, Vertical
from textual.app import ComposeResult
from tui.steps.base import StepScreen

class SummaryStep(StepScreen):
    __json: dict
    __summary: str

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        self.__json = self.session.to_dict()
        self.__summary = self.session.get_summary()

        json_log = RichLog(markup=False, classes="summary-box", id="json-log", wrap=True, auto_scroll=False)
        summary_log = RichLog(markup=False, classes="summary-box", id="summary-log", wrap=True, auto_scroll=False)

        json_log.write(json.dumps(self.__json, indent=2))
        summary_log.write(self.__summary)

        yield Horizontal(
            Vertical(
                Static("Generated Test JSON", classes="field-label"),
                json_log,
                classes="summary-container"
            ),
            Vertical(
                Static("Short Summary", classes="field-label"),
                summary_log,
                classes="summary-container"
            ),
            classes="full"
        )

        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Finish", id="finish"),
            classes="button-row"
        )

    # TODO:
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev":
            from tui.steps.algorithms import AlgorithmsStep
            self.prev(AlgorithmsStep)
        elif event.button.id == "finish":
            pass


    def get_help_desc(self):
        return "a","b"
