"""
Final step: display a summary of the selected configuration.
"""
from textual.widgets import Button, Static
from .base import StepScreen
import json

class SummaryStep(StepScreen):
    def compose(self):
        yield Static("Configuration Summary", classes="screen-header")
        if self.session.environment.general.get("SLURM") is False:
            summary = {
                'environment': self.session.environment.general,
                'mpi': self.session.mpi.config,
                'nodes': self.session.nodes
            }
        else:
            summary = {
                'environment': self.session.environment.general,
                'partition': {
                    'name': self.session.partition.name,
                    **{k: v for k, v in self.session.partition.details.items() if k != "QOS"}
                },
                'qos': {
                    'name': self.session.partition.qos,
                    **(self.session.partition.qos_details or {})
                },
                'mpi': self.session.mpi.config,
                'nodes': self.session.nodes,
                'tasks_per_node': self.session.tasks_per_node,
                'test_time': self.session.test_time or "",
            }
        yield Static(json.dumps(summary, indent=2))
        yield Button("Finish", id="finish")

    def on_button_pressed(self, event):
        if event.button.id == "finish":
            self.app.exit()

    def get_help_desc(self) -> str:
        return "Review your configuration; press Finish to apply or 'q' to quit."
