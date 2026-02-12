# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static, Button, Checkbox, TabbedContent, TabPane, Header, Footer
from .base import StepScreen
from config_loader import alg_get_list, alg_get_algo
from models import CollectiveType, AlgorithmSelection
from typing import List, Tuple
from packaging import version


class AlgorithmsStep(StepScreen):
    __collectives: List[str]

    def compose(self) -> ComposeResult:
        self.__collectives = [str(key) for key in self.session.libraries[0].algorithms.keys()]

        yield Header(show_clock=True)

        yield Static("Select Algorithms for Each Collective", classes="field-label")

        with TabbedContent():
            for pane_num, coll in enumerate(self.__collectives):
                with TabPane(title=f"({pane_num+1}) {coll.capitalize()}", id=f"tab-{coll}"):
                    columns = []
                    for lib in self.session.libraries:
                        lib_id = lib.get_id_name()
                        lib_version = lib.version
                        std_algos = alg_get_list(str(lib.standard), str(lib.lib_type), coll)
                        regular_checks = [
                            Checkbox(
                                f"({lib.name}) {key}",
                                id=f"{coll}-{key}-{lib_id}"
                            )
                            for key, meta in std_algos.items()
                            if (ver := meta.get("version")) and version.parse(ver) <= version.parse(lib_version)
                        ]

                        pico_checks = []
                        if lib.pico_backend:
                            pico_algos = alg_get_list(str(lib.standard), "LibPico", coll)
                            pico_checks = [
                                Checkbox(
                                    f"({lib.name}) {key} (PICO custom)",
                                    id=f"{coll}-{key}-{lib_id}-pico"
                                )
                                for key in pico_algos.keys()
                            ]
                        columns.append(VerticalScroll(*regular_checks, *pico_checks))

                    yield Horizontal(*columns)

        yield self.navigation_buttons()

        yield Footer()


    def on_mount(self) -> None:
        for lib in self.session.libraries:
            for key in lib.algorithms:
                lib.algorithms[key].clear()
        self.__libs_ok = { lib.get_id_name() : False for lib in self.session.libraries }
        self.__coll_ok = { coll: False for coll in self.__collectives }


    async def on_key(self, event: events.Key) -> None:
        if not event.key.isdigit():
            return

        idx = int(event.key) - 1
        tabs = self.query_one(TabbedContent)

        panes = list(tabs.query(TabPane))
        if 0 <= idx < len(panes):
            pane = panes[idx]
            pane_id = pane.id
            if pane_id is not None:
                tabs.active = pane_id

                first_cb = pane.query_one(Checkbox)
                if first_cb:
                    first_cb.focus()

                event.stop()

    def on_checkbox_changed(self):
        self._update_next_button_state()


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            for lib in self.session.libraries:
                lib.algorithms = { 
                    CollectiveType.from_str(coll): [] 
                    for coll in self.__collectives 
                }

            checked = [
                cb for cb in self.query(Checkbox)
                if cb.id and cb.value
            ]

            for cb in checked:
                if not cb.id:
                    raise ValueError("Checkbox ID is missing. This should not happen.")

                parts = cb.id.split("-")

                # Detect PICO suffix
                if len(parts) == 4 and parts[-1] == "pico":
                    coll_str, algo_key, lib_id, _ = parts
                    pico = True
                elif len(parts) == 3:
                    coll_str, algo_key, lib_id = parts
                    pico = False
                else:
                    raise ValueError(f"Unexpected checkbox id format: {cb.id!r}")

                coll = CollectiveType.from_str(coll_str)

                library = next(
                    lib for lib in self.session.libraries
                    if lib.get_id_name() == lib_id
                )

                algo_data = alg_get_algo(str(library.standard), str(library.lib_type) if not pico else "LibPico", coll_str, algo_key )
                if not algo_data:
                    raise ValueError(f"Algorithm {algo_key} not found in {library.lib_type}/{coll_str}.json")

                library.algorithms[coll].append(
                    AlgorithmSelection.from_dict(algo_key, coll_str, algo_data)
                )

            for library in self.session.libraries:
                if not library.validate(validate_algo=True):
                    raise ValueError(f"Library {library.name} contains errors. Please check the configuration.")

            from tui.steps.summary import SummaryStep
            self.next(SummaryStep)

        elif event.button.id == "prev":
            from tui.steps.libraries import LibrariesStep
            self.prev(LibrariesStep)


    def get_help_desc(self) -> Tuple[str, str]:
        focused = self.focused
        default = (
            "Algorithm Selection",
            "Tick at least one algorithm per collective and per library. Use number keys to switch tabs quickly."
        )

        if not focused or not getattr(focused, "id", None):
            return default

        fid = focused.id

        if fid.startswith("tab-"):
            coll = fid.split("-", 1)[1]
            return (
                f"{coll.capitalize()} Tab",
                "Navigate each collective tab and choose algorithms that satisfy your requirements."
            )

        if fid.startswith("prev"):
            return (
                "Previous Step",
                "Return to library configuration (shortcut: `p`)."
            )

        if fid.startswith("next"):
            return (
                "Next Step",
                "Enabled once every selected collective/library pair has at least one algorithm (shortcut: `n`)."
            )

        if "-" not in fid:
            return default

        parts = fid.split("-")
        pico = False
        if len(parts) == 4 and parts[-1] == "pico":
            pico = True
            coll_name, algo_key, lib_id = parts[0], parts[1], parts[2]
        elif len(parts) == 3:
            coll_name, algo_key, lib_id = parts
        else:
            return default

        library = next((lib for lib in self.session.libraries if lib.get_id_name() == lib_id), None)
        if not library:
            return default

        lib_label = library.name
        lib_kind = "LibPico" if pico else str(library.lib_type)
        try:
            algo_meta = alg_get_algo(str(library.standard), lib_kind, coll_name, algo_key)
        except ValueError:
            return (
                "Algorithm Metadata",
                "Algorithm description unavailable; check config/algorithms/ files."
            )

        desc = algo_meta.get("desc", "No description provided.")
        selection = algo_meta.get("selection")
        tags = algo_meta.get("tags", [])
        constraints = algo_meta.get("constraints", [])

        extras = []
        if selection is not None:
            extras.append(f"selector value: {selection}")
        if tags:
            extras.append(f"tags: {', '.join(tags)}")
        if constraints:
            formatted = []
            for constraint in constraints:
                key = constraint.get("key", "?")
                conds = []
                for cond in constraint.get("conditions", []):
                    op = cond.get("operator", "")
                    val = cond.get("value", "")
                    conds.append(f"{op} {val}")
                if conds:
                    formatted.append(f"{key} ({' and '.join(conds)})")
            if formatted:
                extras.append(f"constraints: {', '.join(formatted)}")

        summary = desc
        if extras:
            summary += "\n" + "; ".join(extras)

        return (
            f"{coll_name.capitalize()} · {lib_label}",
            summary
        )


    def _update_next_button_state(self) -> None:
        for coll in self.__collectives:
            found = any(
                cb.value
                for cb in self.query(Checkbox)
                if cb.id and cb.id.startswith(f"{coll}-")
            )
            if found:
                self.__coll_ok[coll] = True

        for lib in self.__libs_ok:
            found = any(
                cb.value
                for cb in self.query(Checkbox)
                if cb.id and (cb.id.endswith(f"-{lib}") or cb.id.endswith(f"-{lib}-pico"))
            )
            if found:
                self.__libs_ok[lib] = True

        enable_next = all(self.__coll_ok.values()) and all(self.__libs_ok.values())
        self.query_one("#next", Button).disabled = not enable_next
