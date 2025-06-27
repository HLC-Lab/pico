"""
Entry point for the benchmarking TUI.
"""
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from textual.containers import Container
from tui.components import Router
from config_loader import SWING_DIR

class PICOtui(App):
    """Main application class."""
    CSS_PATH = SWING_DIR / "tui" / "style.tcss"

    def compose(self) -> ComposeResult:
        """Compose the root layout: header, router container, footer."""
        yield Header()
        yield Container(Router(), id="router-container")
        yield Footer()


if __name__ == '__main__':
    app = PICOtui()
    app.run()
