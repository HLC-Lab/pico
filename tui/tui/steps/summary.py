import json
from time import sleep
from textual.widgets import Button, Static, Header, Footer, RichLog, Label, Input
from textual.containers import Horizontal, Vertical
from textual.app import ComposeResult
from tui.steps.base import StepScreen
from textual.screen import Screen

SAVE_MSG =  "███████╗ █████╗ ██╗   ██╗███████╗  ██████╗ \n"\
            "██╔════╝██╔══██╗██║   ██║██╔════╝  ╚════██╗\n"\
            "███████╗███████║██║   ██║█████╗      ▄███╔╝\n"\
            "╚════██║██╔══██║╚██╗ ██╔╝██╔══╝      ▀▀══╝ \n"\
            "███████║██║  ██║ ╚████╔╝ ███████╗    ██╗   \n"\
            "╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝    ╚═╝   \n"

#TODO: Adjust rendering
class SaveScreen(Screen):
    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "request_quit", "Quit")
    ]

    __json: dict

    def __init__(self, json: dict) -> None:
        super().__init__()
        self.__json = json

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label(SAVE_MSG, id="question", classes="save-label"),
            Input(placeholder="Enter filename to save as...", id="filename-input"),
            Horizontal(
                Button("Save", id="save"),
                Button("Cancel", id="cancel"),
                classes="quit-button-row"
            ),
            id="save-dialog",
        )
        yield Footer()

    #TODO: improve data saving handling
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            filename = self.query_one("#filename-input", Input).value.strip()
            if not filename:
                self.query_one("#filename-input", Input).placeholder = "Filename cannot be empty!"
                return
            try:
                with open(filename, 'w') as f:
                    json.dump(self.__json, f, indent=2)
            except Exception as e:
                self.query_one("#filename-input", Input).placeholder = f"Error saving file: {e}"
                sleep(2)
            self.app.exit()
        else:
            self.app.pop_screen()

    def action_request_quit(self) -> None:
        self.app.pop_screen()

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
            Button("Save", id="save"),
            classes="button-row"
        )

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev":
            from tui.steps.algorithms import AlgorithmsStep
            self.prev(AlgorithmsStep)
        elif event.button.id == "save":
            self.app.push_screen(SaveScreen(self.__json))


    # TODO:
    def get_help_desc(self):
        return "a","b"
