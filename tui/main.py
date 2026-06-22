# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

"""
Entry point for the benchmarking TUI.
"""
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from textual.containers import Container
from tui.components import Router
from config_loader import PICO_DIR

class PICOtui(App):
    """Main application class."""
    CSS_PATH = PICO_DIR / "tui" / "style.tcss"

    def compose(self) -> ComposeResult:
        """Compose the root layout: header, router container, footer."""
        yield Header()
        yield Container(Router(), id="router-container")
        yield Footer()


if __name__ == '__main__':
    app = PICOtui()
    app.run()
