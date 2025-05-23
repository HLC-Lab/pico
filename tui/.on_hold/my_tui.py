'''
textual_menu_example.py

A starting point for a Textual-based terminal user interface (TUI) application
that presents a navigable menu and allows the user to select items.

Requirements:
    - Python 3.8+
    - textual (pip install textual)

Usage:
    python textual_menu_example.py

Features:
    - Vertical menu with selectable items
    - Keyboard navigation (Up/Down arrows)
    - Enter to confirm selection
    - Displays selected item on confirmation

This file is intended as a scaffold for further development.
'''

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Label
from textual.widgets import ListView, ListItem
from textual.reactive import reactive
from textual.events import Key

class MenuSelection(Static):
    message: reactive[str] = reactive("No selection yet.")

    def render(self) -> str:
        return f"[Selected] {self.message}"

class MenuApp(App):

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        menu = ListView(
            ListItem(Label("Option 1"), id="option_1"),
            ListItem(Label("Option 2"), id="option_2"),
            ListItem(Label("Option 3"), id="option_3"),
            ListItem(Label("Option 4"), id="option_4"),
            id="menu_list"
        )
        yield menu
        self.status = MenuSelection()
        yield self.status
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    def on_key(self, event: Key) -> None:
        if event.key == "q":
            self.exit()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        selected_item = event.item
        if selected_item:
            label_widget = selected_item.query_one(Label)
            self.status.message = f"You selected: {label_widget.render()}"

if __name__ == "__main__":
    MenuApp().run()
