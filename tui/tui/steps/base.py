# tui/steps/base.py

"""
Base class for all step screens: shared session, navigation helpers,
context‐sensitive help toggle, and keyboard focus navigation.
"""

from textual.screen import Screen
from tui.steps.help import HelpScreen


class StepScreen(Screen):
    """
    Abstract screen with shared `session`, navigation helpers, and an 'h' key for help.
    Also provides Up/Down/Left/Right and j/k for moving focus among widgets.
    """

    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "quit",        "Quit"),
        ("h", "toggle_help", "Help"),
    ]

    def __init__(self, session, **kwargs):
        """
        Args:
            session: the shared SessionConfig instance
        """
        super().__init__(**kwargs)
        self.session = session

    def on_mount(self) -> None:
        self.title = "PICO"
        self.sub_title = "Performance Evaluation for Collective Operations"
    # ─── Navigation Helpers ─────────────────────────────────────────────────────

    def next(self, next_screen_cls):
        """Pop this screen and push `next_screen_cls(session)`."""
        self.app.pop_screen()
        self.app.push_screen(next_screen_cls(self.session))

    def prev(self, prev_screen_cls):
        """Pop this screen and push `prev_screen_cls(session)`."""
        self.app.pop_screen()
        self.app.push_screen(prev_screen_cls(self.session))

    # ─── Bound Actions ──────────────────────────────────────────────────────────

    def action_quit(self):
        """Invoked when 'q' is pressed."""
        self.app.exit()

    def action_toggle_help(self):
        """
        Toggle the help overlay. Pops it if already visible, otherwise pushes a new one.
        """
        # Check if HelpScreen is already on top of the stack
        stack = getattr(self.app, "screen_stack", [])
        if stack and isinstance(stack[-1], HelpScreen):
            self.app.pop_screen()
        else:
            desc = self.get_help_desc() or "No help available."
            self.app.push_screen(HelpScreen(desc))


    # ─── Subclasses Must Override ────────────────────────────────────────────────

    def get_help_desc(self) -> str:
        """
        Return context‐sensitive help text for the current screen.
        Must be implemented by each step subclass.
        """
        raise NotImplementedError
