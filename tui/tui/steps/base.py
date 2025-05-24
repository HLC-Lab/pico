"""
Base class for step screens, handling navigation and universal help toggle.
"""
from textual.screen import Screen
from tui.steps.help import HelpScreen

class StepScreen(Screen):
    """
    Abstract screen with shared `session`, navigation helpers, and an 'h' key for help.
    """
    BINDINGS = [
        ("q", "quit",        "Quit"),
        ("h", "toggle_help", "Help"),
    ]

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    def next(self, next_screen):
        """Pop this screen and push `next_screen(session)`."""
        self.app.pop_screen()
        self.app.push_screen(next_screen(self.session))

    def action_toggle_help(self):
        """
        Toggle the help overlay in and out.
        If a HelpScreen is on top, pop it; otherwise push a new one.
        """
        stack = getattr(self.app, "screen_stack", [])
        if stack and isinstance(stack[-1], HelpScreen):
            self.app.pop_screen()
        else:
            desc = self.get_help_desc() or "No help available."
            self.app.push_screen(HelpScreen(desc))

    def get_help_desc(self) -> str:
        """
        Must be overridden by subclasses.
        Return context‐sensitive help text.
        """
        raise NotImplementedError
