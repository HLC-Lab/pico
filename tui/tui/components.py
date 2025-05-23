from textual.widget import Widget
from textual.screen import Screen

from .steps.environment import EnvironmentStep
from models import SessionConfig

class Router(Widget):
    def on_mount(self):
        self.session = SessionConfig()
        self.app.push_screen(EnvironmentStep(self.session))
