from textual.widgets import Static, Button
from .base import StepScreen
import json

class SummaryStep(StepScreen):
    def compose(self):
        yield Static("Configuration Summary", classes="screen-header")
        summary = {
            'environment': self.session.environment.general,
            'partition': self.session.partition.details,
            'qos': self.session.partition.qos,
            'mpi': self.session.mpi.config,
        }
        yield Static(json.dumps(summary, indent=2))
        yield Button("Finish", id="finish")

    def on_button_pressed(self, event):
        if event.button.id == "finish":
            self.app.exit()
