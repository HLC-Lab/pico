from textual.app import ComposeResult
from textual.widgets import Button, Label, Footer, Static, Select, Input
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from typing import Tuple



QUIT_MSG =  " ██████╗ ██╗   ██╗██╗████████╗    ██████╗ \n"\
            "██╔═══██╗██║   ██║██║╚══██╔══╝    ╚════██╗\n"\
            "██║   ██║██║   ██║██║   ██║         ▄███╔╝\n"\
            "██║▄▄ ██║██║   ██║██║   ██║         ▀▀══╝ \n"\
            "╚██████╔╝╚██████╔╝██║   ██║         ██╗   \n"\
            " ╚══▀▀═╝  ╚═════╝ ╚═╝   ╚═╝         ╚═╝   "

HELP_MSG =  "██╗  ██╗███████╗██╗     ██████╗  \n"\
            "██║  ██║██╔════╝██║     ██╔══██╗ \n"\
            "███████║█████╗  ██║     ██████╔╝ \n"\
            "██╔══██║██╔══╝  ██║     ██╔═══╝  \n"\
            "██║  ██║███████╗███████╗██║      \n"\
            "╚═╝  ╚═╝╚══════╝╚══════╝╚═╝      "

class QuitScreen(Screen):
    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "request_quit", "Quit")
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(QUIT_MSG, id="question", classes="quit-label"),
                Horizontal(
                    Button("No", id="cancel"),
                    Button("Yes", id="quit"),
                    classes="quit-button-row"
                ),
            id="quit-dialog",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        else:
            self.app.pop_screen()

    def action_request_quit(self) -> None:
        self.app.pop_screen()



class HelpScreen(Screen):
    def __init__(self, help: Tuple[str, str]) -> None:
        super().__init__()
        self.help_title = help[0]
        self.content = help[1]

    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "request_quit", "Quit"),
        ("h", "toggle_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(HELP_MSG, id="help-header", classes="help-label"),
            Label(self.help_title, id="help-title", classes="main-label"),
            Static(self.content, id="help-box"),
            Horizontal(
                Button("Close", id="close"),
                classes="end-page-button-row"
            ),
            id="help-dialog",
        )

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.app.pop_screen()

    def action_request_quit(self) -> None:
        self.app.push_screen(QuitScreen())

    def action_toggle_help(self) -> None:
        self.app.pop_screen()


class StepScreen(Screen):

    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("p", "go_prev", "Previous Step"),
        ("P", "go_prev", "Previous Step"),
        ("n", "go_next", "Next Step"),
        ("N", "go_next", "Next Step"),
        ("h", "toggle_help", "Help"),
        ("H", "toggle_help", "Help"),
        ("?", "toggle_help", "Help"),
        ("q", "request_quit", "Quit"),
        ("Q", "request_quit", "Quit"),
    ]

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    def on_mount(self) -> None:
        self.title = "PICO"
        self.sub_title = "Performance Insights for Collective Operations"

    def navigation_buttons(self, prev_disabled: bool = False, next_disabled: bool = True) -> Horizontal:
        return Horizontal(
            Button("Prev", id="prev", disabled=prev_disabled),
            Button("Next", id="next", disabled=next_disabled),
            classes="button-row"
        )

    def reset_select(self, widget: Select, disable: bool = True, clear = True):
        """Clear out options, reset value to blank, disable."""
        if clear:
            widget._options = []
            widget._setup_variables_for_options([])
            widget._setup_options_renderables()
        widget.value = Select.BLANK
        widget.disabled = disable

    def reset_input(self, widget: Input, disable: bool = True):
        widget.value = ""
        widget.disabled = disable

    @property
    def has_slurm(self) -> bool:
        return self.session.environment.slurm if self.session.environment else False

    @property
    def lib_number(self) -> int:
        return len(self.session.libraries) if self.session.libraries else 0

    # ─── Navigation Helpers ─────────────────────────────────────────────────────

    def next(self, next_screen_cls):
        self.app.pop_screen()
        self.app.push_screen(next_screen_cls(self.session))

    def prev(self, prev_screen_cls):
        self.app.pop_screen()
        self.app.push_screen(prev_screen_cls(self.session))


    # ─── Bound Actions ──────────────────────────────────────────────────────────

    def action_request_quit(self) -> None:
        self.app.push_screen(QuitScreen())

    def action_toggle_help(self) -> None:
        self.app.push_screen(HelpScreen(self.get_help_desc()))

    def action_go_next(self) -> None:
        next_button = self.query_one("#next", Button)
        if not next_button.disabled:
            self.on_button_pressed(Button.Pressed(next_button))

    def action_go_prev(self) -> None:
        prev_button = self.query_one("#prev", Button)
        if not prev_button.disabled:
            self.on_button_pressed(Button.Pressed(prev_button))


    # ─── Subclasses Must Override ────────────────────────────────────────────────

    def on_button_pressed(self, event):
        pass

    def get_help_desc(self) -> Tuple[str, str]:
        raise NotImplementedError
