# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Tuple

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from config_loader import PICO_DIR
from export_builder import json_to_exports, save_export_bundle
from summary_builder import build_effective_summary
from tui.steps.base import StepScreen


SAVE_MSG = "███████╗ █████╗ ██╗   ██╗███████╗  ██████╗ \n" \
           "██╔════╝██╔══██╗██║   ██║██╔════╝  ╚════██╗\n" \
           "███████╗███████║██║   ██║█████╗      ▄███╔╝\n" \
           "╚════██║██╔══██║╚██╗ ██╔╝██╔══╝      ▀▀══╝ \n" \
           "███████║██║  ██║ ╚████╔╝ ███████╗    ██╗   \n" \
           "╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝    ╚═╝   \n"

TEST_DIR = PICO_DIR / "tests"


class SaveScreen(Screen):
    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "request_quit", "Quit"),
    ]

    __data: dict

    def __init__(self, json_data: dict) -> None:
        super().__init__()
        self.__data = json_data

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label(SAVE_MSG, id="question", classes="save-label"),
            Static(
                "Files will be saved in `./tests` directory.",
                classes="field-label",
            ),
            Input(
                placeholder="Enter filename to save as...",
                id="filename-input",
            ),
            Label("", id="path-error", classes="error"),
            Horizontal(
                Button("Save", id="save", disabled=True),
                Button("Cancel", id="cancel"),
                classes="quit-button-row",
            ),
            id="save-dialog",
        )
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            raw = self.query_one("#filename-input", Input).value.strip()
            base_name = Path(raw).name
            stem = Path(base_name).stem
            json_name = (
                stem + ".json"
                if not base_name.lower().endswith(".json")
                else base_name
            )
            shell_name = stem + ".sh"
            TEST_DIR.mkdir(exist_ok=True)

            suffix_id = 0
            while True:
                name = (
                    f"{Path(json_name).stem}_{suffix_id}.json"
                    if suffix_id
                    else json_name
                )
                shell_candidate = (
                    f"{Path(shell_name).stem}_{suffix_id}.sh"
                    if suffix_id
                    else shell_name
                )
                target = (TEST_DIR / name).resolve()
                shell_target = (TEST_DIR / shell_candidate).resolve()

                inside_test_dir = (
                    TEST_DIR.resolve() in target.parents
                    and TEST_DIR.resolve() in shell_target.parents
                )
                if (
                    inside_test_dir
                    and not target.exists()
                    and not shell_target.exists()
                ):
                    break
                suffix_id += 1

            try:
                await asyncio.to_thread(
                    save_export_bundle,
                    self.__data,
                    target,
                    shell_target,
                )
            except Exception as error:
                self.query_one("#path-error", Label).update(
                    f"Error saving file: {error}"
                )
                return

            await asyncio.sleep(2)
            self.app.exit()
        elif event.button.id == "cancel":
            self.app.pop_screen()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filename-input":
            return

        filename = event.value.strip()
        save_button = self.query_one("#save", Button)
        error = self.query_one("#path-error", Label)

        if not filename:
            save_button.disabled = True
            error.update("Filename cannot be empty.")
            return

        if filename.count(".") > 1:
            save_button.disabled = True
            error.update("Filename cannot contain multiple dots.")
            return

        if "." in filename and not filename.lower().endswith(".json"):
            save_button.disabled = True
            error.update("Only .json extension is allowed.")
            return

        save_button.disabled = False
        error.update("")

    def action_request_quit(self) -> None:
        self.app.pop_screen()


class SummaryStep(StepScreen):
    __json: dict
    __summary: str

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        self.__json = self.session.to_dict()
        self.__summary = build_effective_summary(self.session, PICO_DIR)
        json_log = RichLog(
            markup=False,
            classes="summary-box",
            id="json-log",
            wrap=True,
            auto_scroll=False,
        )
        summary_log = RichLog(
            markup=False,
            classes="summary-box",
            id="summary-log",
            wrap=True,
            auto_scroll=False,
        )
        json_log.write(json.dumps(self.__json, indent=2))
        summary_log.write(self.__summary)

        yield Horizontal(
            Vertical(
                Static("Generated Test JSON", classes="field-label"),
                json_log,
                classes="summary-container",
            ),
            Vertical(
                Static("Effective Execution Plan", classes="field-label"),
                summary_log,
                classes="summary-container",
            ),
            classes="full",
        )

        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Save", id="next"),
            classes="button-row",
        )

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev":
            self.prev()
        elif event.button.id == "next":
            self.app.push_screen(SaveScreen(self.__json))

    def get_help_desc(self) -> Tuple[str, str]:
        focused = self.focused
        default = (
            "Review & Export",
            "Inspect the generated JSON and effective execution plan, "
            "then save the bundle into tests/.",
        )

        if not focused or not getattr(focused, "id", None):
            return default

        if focused.id == "json-log":
            return (
                "Generated Test JSON",
                "Full configuration as saved to <name>.json. Use arrow keys "
                "or PgUp/PgDn to scroll.",
            )
        if focused.id == "summary-log":
            return (
                "Effective Execution Plan",
                "Review resources, resolved libraries and selectors, "
                "benchmark case counts, output behavior, provenance, "
                "and warnings.",
            )
        if focused.id == "prev":
            return (
                "Previous Step",
                "Return to algorithm selection to make changes "
                "(shortcut: `p`).",
            )
        if focused.id == "next":
            return (
                "Save & Export",
                "Open the save dialog to choose a filename. An executable "
                ".sh export is produced alongside the JSON.",
            )

        return default
