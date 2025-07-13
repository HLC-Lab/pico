# Copyright (c) 2025 Saverio Pasqualoni
# Licensed under the MIT License

"""
Defines the Router widget and imports all step screens.
"""
from textual.widget import Widget
from models import SessionConfig
from .steps.configure import ConfigureStep

class Router(Widget):
    """
    Orchestrates the step-by-step screens.
    """
    def on_mount(self) -> None:
        """Initialize session and show first screen."""
        self.session = SessionConfig()
        # Push the environment selection screen
        self.app.push_screen(ConfigureStep(self.session))
