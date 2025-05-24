"""
Base class for step screens, handling navigation.
"""
from textual.screen import Screen


class StepScreen(Screen):
    """
    Abstract screen with `session` and `next()` helper.
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    def next(self, next_screen):
        """
        Pop this screen and push `next_screen(session)`.
        """
        self.app.pop_screen()
        self.app.push_screen(next_screen(self.session))
