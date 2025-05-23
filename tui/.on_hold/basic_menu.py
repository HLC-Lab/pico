import curses
import time
from dataclasses import dataclass

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

@dataclass
class WindowState:
    height: int
    width: int
    begin_y: int
    begin_x: int
    items: list
    title: str | None = None
    footer: list | None = None
    idx: int = 0
    offset: int = 0
    active_key: str | None = None
    active: bool = False
    blink: bool = True
    last_blink: float = time.time()

def render_key_help(help_list):
    return "  — "+"    — ".join([f"{key}: {desc}" for key, desc in help_list])

def draw_windows(win: curses.window, state: WindowState):
    win.clear()
    win.box()

    if state.title != None:
        win.addstr(0, 2, f" {state.title} ", curses.A_BOLD | curses.A_REVERSE)
    if state.footer != None:
        win.addstr(state.height - 2, 2,
                render_key_help(state.footer).ljust(state.width - 4),
                curses.A_REVERSE)

    visible = state.items[state.offset: state.height - 2]
    for idx, item in enumerate(visible):
        y = idx + 1

        is_selected = (state.offset + idx == state.idx)
        # attr = curses.color_pair(1) if is_selected else curses.color_pair(2)
        attr = curses.color_pair(0)
        # apply manual blink only if both active _and_ blink_on flag is True
        if state.active and is_selected and state.blink:
            attr |= curses.A_REVERSE
        win.addnstr(y, 1, f" {item} ", state.width - 2, attr)

    win.refresh()


def handle_navigation(state: WindowState, key: int):
    max_idx = len(state.items) - 1
    win_height = state.height - 2  # number of lines available for items

    if key in (curses.KEY_DOWN, ord('j')):
        if state.idx < max_idx:
            state.idx += 1
            if state.idx >= state.offset + win_height:
                state.offset += 1

    elif key in (curses.KEY_UP, ord('k')):
        if state.idx > 0:
            state.idx -= 1
            if state.idx < state.offset:
                state.offset -= 1

def handle_resize(win, state, new_h, new_w):
    state.height, state.width = new_h - 2, new_w - 4
    win.resize(state.height, state.width)
    win.mvwin(state.begin_y, state.begin_x)


def draw_all(panes, active):
    for idx, (st, win) in enumerate(panes):
        st.active = (idx == active)
        if (st.active):
            now = time.time()
            if now - st.last_blink >= BLINK_INTERVAL:
                st.blink = not st.blink
                st.last_blink = now

        draw_windows(win, st)


def main(stdscr):
    # setup color pairs to use
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)

    stdscr.nodelay(True)          # make getch nonblocking
    stdscr.timeout(100)           # wake every 100ms to update blink

    # initialize screen
    stdscr.clear()
    stdscr.refresh()

    # Get window size and setup the main window
    h, w = stdscr.getmaxyx()
    main_state = WindowState(h - 2, w - 4, 1, 2, ITEMS,
                             "Main Menu", KEY_HELP)
    main_win = curses.newwin(main_state.height, main_state.width,
                             main_state.begin_y, main_state.begin_x)
    panes = [ (main_state, main_win) ]
    active = 0

    while True:
        new_h, new_w = stdscr.getmaxyx()
        if (new_h, new_w) != (h, w):
            h, w = new_h, new_w
            handle_resize(main_win, main_state, new_h, new_w)
            stdscr.clear()
            stdscr.refresh()

        draw_all(panes, active)

        stdscr.clrtoeol()
        stdscr.refresh()

        # Input handling
        key = stdscr.getch()
        if key == -1:
            continue
        if key in (curses.KEY_DOWN, ord('j'), curses.KEY_UP, ord('k')):
            handle_navigation(panes[active][0], key)
        elif key == ord('\t'):           # Tab to cycle focus
            active = (active + 1) % len(panes)
        elif key == ord(' '):
            if active == 0:
                new_state = WindowState(h - 2, (w - 4) // 2, 1, w // 2, SUBITEM[ITEMS[0]])
                new_win   = curses.newwin(new_state.height, new_state.width,
                                          new_state.begin_y, new_state.begin_x)
                panes.append( (new_state, new_win) )
                active = 1
            elif active == 1:
                panes.pop(active)
                active = 0
        elif key == ord('h'):                           pass    # TODO: show help
        elif key == ord('q'):                           break

if __name__ == '__main__':
    curses.wrapper(main)
