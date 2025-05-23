from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from textual.containers import Container
from tui.components import Router

class BenchmarkApp(App):
    # CSS_PATH = "style.tcss"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(Router(), id="router-container")
        yield Footer()

if __name__ == '__main__':
    BenchmarkApp().run()
