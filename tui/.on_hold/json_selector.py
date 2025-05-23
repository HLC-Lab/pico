"""
TUI Benchmark Selector

A curses-based text UI for selecting multiple JSON configuration files across
directories and choosing items within each JSON. Supports persistent cross-
directory file selection, item-level selection with descriptions, and a summary
output as JSON.
"""
import curses
import json
import os
import argparse
import textwrap

# Key bindings displayed at bottom of file-selection screen
KEY_HELP = [
    ("Up/Down", "move selection cursor"),
    ("Enter(dir)", "open directory"),
    ("Space(file)", "toggle file select"),
    ("f", "finish file selection"),
    ("q", "quit without selecting")
]


def render_key_help(help_list=KEY_HELP):
    return "  ".join([f"{key}: {desc}" for key, desc in help_list])


def list_dir(path):
    """
    List only subdirectories and .json files in `path`.

    Returns:
        List of tuples (display_name, full_path, type), where type is 'dir' or 'file'.
        Directories are suffixed with os.sep, listed first, then JSON files, all sorted
        case-insensitively by name.

    Errors:
        Silently ignores PermissionError when accessing restricted directories.
    """
    entries = []
    try:
        for name in os.listdir(path):
            full = os.path.join(path, name)

            if os.path.isdir(full):                 entries.append((name + os.sep, full, 'dir'))
            elif name.lower().endswith('.json'):    entries.append((name, full, 'file'))
    except PermissionError:
        pass

    entries.sort(key=lambda x: (x[2] != 'dir', x[0].lower()))
    return entries


def draw_window(win, title, entries, current_idx, offset, height, width, mode, selected_flags=None):
    """
    Render a boxed subwindow with a title and list of entries.

    Args:
        win: curses window to draw into
        title: string to display in the window's title bar
        entries: list of tuples (name, full_path, type)
        current_idx: absolute index of the highlighted entry
        offset: scroll offset into entries
        height, width: dimensions of the window
        mode: 'file_select' or 'item_select' (controls prefix display)
        selected_flags: optional list of booleans parallel to entries for checkboxes
    """
    win.clear()
    win.box()
    win.addstr(0, 2, f" {title} ", curses.A_BOLD)
    visible = entries[offset:offset + height - 2]

    for idx, (name, _, typ) in enumerate(visible):
        y = idx + 1
        prefix = ''
        if typ == 'file' and mode in ('file_select', 'item_select') and selected_flags is not None:
            sel = selected_flags[offset + idx]
            prefix = '[x] ' if sel else '[ ] '
        text = prefix + name

        attr = curses.color_pair(1) if offset + idx == current_idx else 0
        if offset + idx != current_idx and typ == 'dir':
            attr |= curses.color_pair(2)
        win.addnstr(y, 1, text, width - 2, attr)
    win.addstr(height - 2, 2, render_key_help())
    win.refresh()


def select_files(stdscr, base_dir):
    """
    Let the user navigate directories and select multiple JSON files.
    The selection persists across directory changes.

    Args:
        stdscr: main curses screen
        base_dir: root directory to start browsing

    Returns:
        List of full paths to all JSON files selected, or empty list if canceled.
    """
    curses.curs_set(0)
    stdscr.clear()
    stdscr.refresh()

    h, w = stdscr.getmaxyx()
    win = curses.newwin(h - 2, w - 4, 1, 2)
    base_real = os.path.abspath(base_dir)
    current_path = base_real
    current_idx = 0
    offset = 0
    selected_set = set()

    while True:
        new_h, new_w = stdscr.getmaxyx()
        if new_h != h or new_w != w:
            stdscr.clear()
            stdscr.refresh()
            h, w = new_h, new_w
            win.resize(h - 2, w - 4)
            win.mvwin(1, 2)
        entries = []
        if os.path.realpath(current_path) != base_real:
            entries.append(('..' + os.sep, os.path.dirname(current_path), 'dir'))
        entries.extend(list_dir(current_path))

        selected_flags = [(full in selected_set) for (_, full, _) in entries]

        current_idx = max(0, min(current_idx, len(entries) - 1))
        max_visible = h - 4
        if current_idx < offset:                                offset = current_idx
        elif current_idx >= offset + max_visible:               offset = current_idx - max_visible + 1

        draw_window(win, f"Select JSON files: {current_path}", entries, current_idx,
                    offset, h - 2, w - 4, 'file_select', selected_flags)
        stdscr.clrtoeol()
        stdscr.refresh()

        key = stdscr.getch()

        if key == ord(' '):
            name, full, typ = entries[current_idx]
            if typ == 'file':
                if full in selected_set:                        selected_set.remove(full)
                else:                                           selected_set.add(full)
        elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
            name, full, typ = entries[current_idx]
            if typ == 'dir':
                current_path = full
                current_idx = offset = 0
        elif key in [curses.KEY_UP, ord('k')]:                  current_idx -= 1
        elif key in [curses.KEY_DOWN, ord('j')]:                current_idx += 1
        elif key == ord('f'):                                   break
        elif key in [ord('q'), 27]:                             return []

    return list(selected_set)


def select_items(stdscr, data, title="Select items"):
    """
    Present keys of a loaded JSON object for multi-selection with optional descriptions.
    Controls: Up/Down or j/k to move, Space to toggle, 'h' for help popup, Enter to confirm.

    Args:
        stdscr: main curses screen
        data: dict loaded from JSON, where values may include a 'desc' field
        title: title string shown at top of selection window

    Returns:
        Sub-dict of `data` containing only the items the user left selected.
    """
    curses.curs_set(0)
    h, w = stdscr.getmaxyx()
    win = curses.newwin(h - 2, w - 4, 1, 2)
    items = list(data.keys())
    selected = [True] * len(items)
    current_idx = 0
    offset = 0
    max_visible = h - 4

    while True:
        new_h, new_w = stdscr.getmaxyx()
        if new_h != h or new_w != w:
            h, w = new_h, new_w
            win.resize(h - 2, w - 4)
            win.mvwin(1, 2)

        draw_window(win, title, [(items[i], None, 'file') for i in range(len(items))],
                    current_idx, offset, h - 2, w - 4, 'item_select', selected)
        stdscr.addstr(h - 1, 2, f"{title} - Selected {sum(selected)}/{len(items)}   Up/Down Space h:help Enter")
        stdscr.clrtoeol()
        stdscr.refresh()
        key = stdscr.getch()

        if key == ord('h'):
            desc = data[items[current_idx]].get('desc', 'No description available.')
            show_description_popup(desc, h, w)
            draw_window(win, title, [(items[i], None, 'file') for i in range(len(items))],
                        current_idx, offset, h - 2, w - 4, 'item_select', selected)
        elif key == ord(' '):                                   selected[current_idx] = not selected[current_idx]
        elif key in [curses.KEY_UP, ord('k')]:                  current_idx = max(0, current_idx - 1)
        elif key in [curses.KEY_DOWN, ord('j')]:                current_idx = min(len(items) - 1, current_idx + 1)
        elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:   break

        if current_idx < offset:                                offset = current_idx
        elif current_idx >= offset + max_visible:               offset = current_idx - max_visible + 1

    return {items[i]: data[items[i]] for i in range(len(items)) if selected[i]}


def show_description_popup(desc, h, w):
    """
    Display a centered popup window with the given description text.
    """
    lines = textwrap.wrap(desc, w - 8)
    ph = min(len(lines) + 4, h - 4)
    pw = min(max((len(line) for line in lines), default=0) + 4, w - 4)
    py = (h - ph) // 2
    px = (w - pw) // 2
    popup = curses.newwin(ph, pw, py, px)
    popup.box()
    popup.addstr(0, 2, " Description ", curses.A_BOLD)
    for i, line in enumerate(lines[:ph-3]):
        popup.addnstr(i+1, 2, line, pw-4)
    popup.addstr(ph-2, 2, "Press any key to close")
    popup.refresh()
    popup.getch()
    popup.clear()
    popup.refresh()
    del popup

def main(stdscr, base_dir):
    """
    Entry point for curses.wrapper.  Initializes colors, runs file and item selection,
    then shows a summary and prints JSON to stdout.
    """
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)

    # Stage 1: choose which JSON files to open
    json_files = select_files(stdscr, base_dir)
    if not json_files:
        return

    results = {}
    # Stage 2: for each file, let user select items inside it
    for path in json_files:
        try:
            data = json.load(open(path))
        except Exception as e:
            continue
        rel_path = os.path.relpath(path, base_dir)
        chosen = select_items(stdscr, data, title=f"Items in {rel_path}")
        results[rel_path] = chosen

    # Show summary on screen
    stdscr.clear()
    stdscr.addstr(0, 0, "Selection summary:", curses.A_BOLD)
    row = 1
    for path, items in results.items():
        stdscr.addstr(row, 2, path + ":")
        row += 1
        for key in items:
            stdscr.addstr(row, 4, key)
            row += 1
        row += 1
    stdscr.addstr(row, 0, "Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()

    # finally, output JSON mapping of file->chosen items
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TUI JSON multi-file & item selector with persistent cross-directory selection')
    parser.add_argument('base_dir', help='Base directory to browse')
    args = parser.parse_args()
    curses.wrapper(main, args.base_dir)
