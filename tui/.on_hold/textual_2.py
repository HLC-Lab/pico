from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, ListView, ListItem, Static
from textual.reactive import reactive
from textual.containers import Container, Horizontal
from textual import on
from textual.events import Key

BLINK_INTERVAL = 0.4

ITEMS = ['a', 'b', 'c', 'd', 'e']
SUBITEM = {
    'a': ['a1', 'a2', 'a3'],
    'b': ['b1', 'b2', 'b3'],
    'c': ['c1', 'c2', 'c3'],
    'd': ['d1', 'd2', 'd3'],
    'e': ['e1', 'e2', 'e3'],
}
KEY_HELP = [
    ("j/k", "▼/▲"),
    ("[space]", "select"),
    ("h", "toggle help"),
    ("q", "quit")
]

class TUIApp(App):
    CSS_PATH = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="menu-container"):
            self.main_menu = ListView(*[ListItem(Static(item)) for item in ITEMS], id="main")
            yield self.main_menu
        self.help_text = "  — ".join([f"{k}: {d}" for k, d in KEY_HELP])
        self.help_widget = Static(self.help_text, id="help")
        self.help_widget.styles.offset_y = 3
        self.help_widget.styles.opacity = 0.0
        yield Footer()
        yield self.help_widget

    async def on_ready(self) -> None:
        self.focus_main = True
        self.main_menu.index = 0
        self.sub_menu = None
        self.help_visible = False

    def _current_menu(self):
        return self.main_menu if self.focus_main else self.sub_menu

    @on("key(tab)")
    def switch_focus(self) -> None:
        if self.sub_menu:
            self.focus_main = not self.focus_main

    @on("key(space)")
    async def toggle_submenu(self) -> None:
        if self.focus_main:
            sel = ITEMS[self.main_menu.index]
            items = SUBITEM.get(sel, [])
            if items and not self.sub_menu:
                self.sub_menu = ListView(*[ListItem(Static(it)) for it in items], id="sub")
                container = self.query_one("#menu-container", Horizontal)
                await container.mount(self.sub_menu)
                self.focus_main = False
        else:
            if self.sub_menu:
                await self.sub_menu.remove()
                self.sub_menu = None
                self.focus_main = True

    @on("key(h)")
    async def toggle_help(self) -> None:
        if not self.help_visible:
            await self.help_widget.animate("opacity", value=1.0, duration=0.3)
            self.help_visible = True
        else:
            await self.help_widget.animate("opacity", value=0.0, duration=0.3)
            self.help_visible = False

    @on("key(q)")
    async def quit_app(self) -> None:
        await self.action_quit()

if __name__ == "__main__":
    TUIApp().run()
