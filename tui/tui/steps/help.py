"""
A modal screen that shows a single `desc` string and closes on 'h' or the Close button.
"""
from textual.screen import Screen
from textual.widgets import Static, Button

class HelpScreen(Screen):
    BINDINGS = [
        ("h", "pop_screen", "Close Help"),
        ("q", "quit",       "Quit"),
    ]

    def __init__(self, desc: str, **kwargs):
        super().__init__(**kwargs)
        self.desc = desc

    def compose(self):
        yield Static(self.desc, expand=True)
        yield Button("Close (h)", id="close")

    def on_button_pressed(self, event):
        if event.button.id == "close":
            self.app.pop_screen()
