import json
import asyncio
from pathlib import Path
from time import sleep
from textual.widgets import Button, Static, Header, Footer, RichLog, Label, Input
from textual.containers import Horizontal, Vertical
from textual.app import ComposeResult
from tui.steps.base import StepScreen
from textual.screen import Screen
from config_loader import SWING_DIR

SAVE_MSG =  "███████╗ █████╗ ██╗   ██╗███████╗  ██████╗ \n"\
            "██╔════╝██╔══██╗██║   ██║██╔════╝  ╚════██╗\n"\
            "███████╗███████║██║   ██║█████╗      ▄███╔╝\n"\
            "╚════██║██╔══██║╚██╗ ██╔╝██╔══╝      ▀▀══╝ \n"\
            "███████║██║  ██║ ╚████╔╝ ███████╗    ██╗   \n"\
            "╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝    ╚═╝   \n"

TEST_DIR = SWING_DIR / "tests"

class SaveScreen(Screen):
    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "request_quit", "Quit")
    ]

    __data: dict

    def __init__(self, json: dict) -> None:
        super().__init__()
        self.__data = json

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label(SAVE_MSG, id="question", classes="save-label"),
            Static("Files will be saved in `./tests` directory.", classes="field-label"),
            Input(placeholder="Enter filename to save as...", id="filename-input"),
            Label("", id="path-error", classes="error"),
            Horizontal(
                Button("Save", id="save", disabled=True),
                Button("Cancel", id="cancel"),
                classes="quit-button-row"
            ),
            id="save-dialog",
        )
        yield Footer()


    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            filename = self.query_one("#filename-input", Input).value.strip()
            if not filename.lower().endswith(".json"):
                filename += ".json"

            TEST_DIR.mkdir(exist_ok=True)
            target = TEST_DIR / filename
            base_stem = Path(filename).stem
            id = 1
            while target.exists():
                target = TEST_DIR / f"{base_stem}_{id}.json"
                id += 1

            try:
                await asyncio.to_thread(
                    lambda: target.write_text(json.dumps(self.__data, indent=2))
                )
            except Exception as e:
                self.query_one("#path-error", Label).update(f"Error saving file: {e}")

            await asyncio.sleep(2)
            self.app.exit()
        elif event.button.id == "cancel":
            self.app.pop_screen()


    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filename-input":
            return

        filename = event.value.strip()
        save_btn = self.query_one("#save", Button)
        error = self.query_one("#path-error", Label)

        if not filename:
            save_btn.disabled = True
            error.update("Filename cannot be empty.")
            return

        if filename.count('.') > 1:
            save_btn.disabled = True
            error.update("Filename cannot contain multiple dots.")
            return

        if '.' in filename and not filename.lower().endswith(".json"):
            save_btn.disabled = True
            error.update("Only .json extension is allowed.")
            return

        save_btn.disabled = False
        error.update("")


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
            Button("Save", id="next"),
            classes="button-row"
        )

        yield Footer()


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev":
            from tui.steps.algorithms import AlgorithmsStep
            self.prev(AlgorithmsStep)
        elif event.button.id == "next":
            self.app.push_screen(SaveScreen(self.__json))


    # TODO:
    def get_help_desc(self):
        return "a","b"
