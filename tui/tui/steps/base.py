from textual.screen import Screen

class StepScreen(Screen):
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    def next(self, next_screen):
        self.app.pop_screen()
        self.app.push_screen(next_screen(self.session))
