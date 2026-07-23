# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static, Button, Checkbox, TabbedContent, TabPane, Header, Footer
from .base import StepScreen
from config_loader import alg_get_list
from models import (
    AlgorithmSelection,
    CollectiveType,
    LibrarySelection,
    TestType,
    get_algorithm_constraint_issue,
    has_algorithm_coverage,
)
from typing import Dict, List, Tuple
from packaging import version


class AlgorithmsStep(StepScreen):
    __collectives: List[str]
    __algorithm_widgets: Dict[str, Tuple[LibrarySelection, str, str, bool, dict]]

    def compose(self) -> ComposeResult:
        self.__collectives = [str(key) for key in self.session.libraries[0].algorithms.keys()]
        self.__algorithm_widgets = {}
        widget_index = 0

        yield Header(show_clock=True)

        yield Static("Select Algorithms for Each Collective", classes="field-label")

        with TabbedContent():
            for pane_num, coll in enumerate(self.__collectives):
                with TabPane(title=f"({pane_num+1}) {coll.capitalize()}", id=f"tab-{coll}"):
                    columns = []
                    for lib in self.session.libraries:
                        lib_version = lib.version
                        std_algos = alg_get_list(str(lib.standard), str(lib.lib_type), coll)
                        regular_checks = []
                        for key, meta in std_algos.items():
                            required_version = meta.get("version")
                            if not required_version or version.parse(required_version) > version.parse(lib_version):
                                continue
                            checkbox_id = f"algorithm-{widget_index}"
                            widget_index += 1
                            regular_checks.append(
                                self.__make_algorithm_checkbox(
                                    checkbox_id, lib, coll, key, meta, pico=False
                                )
                            )

                        pico_checks = []
                        if lib.pico_backend:
                            pico_algos = alg_get_list(str(lib.standard), "LibPico", coll)
                            for key, meta in pico_algos.items():
                                checkbox_id = f"algorithm-{widget_index}"
                                widget_index += 1
                                pico_checks.append(
                                    self.__make_algorithm_checkbox(
                                        checkbox_id, lib, coll, key, meta, pico=True
                                    )
                                )
                        columns.append(VerticalScroll(*regular_checks, *pico_checks))

                    yield Horizontal(*columns)

        yield self.navigation_buttons()

        yield Footer()


    def on_mount(self) -> None:
        self._update_next_button_state()


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

                checkboxes = list(pane.query(Checkbox))
                if checkboxes:
                    checkboxes[0].focus()

                event.stop()

    def on_checkbox_changed(self):
        self._update_next_button_state()


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            self.__store_selections()

            for library in self.session.libraries:
                if not library.validate(validate_algo=True):
                    raise ValueError(f"Library {library.name} contains errors. Please check the configuration.")

            from tui.steps.summary import SummaryStep
            self.next(SummaryStep)

        elif event.button.id == "prev":
            self.__store_selections()
            self.prev()


    def get_help_desc(self) -> Tuple[str, str]:
        focused = self.focused
        default = (
            "Algorithm Selection",
            "Choose algorithms so every collective is covered and every library contributes at least one. Use number keys to switch tabs quickly."
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
                "Enabled once every collective has an algorithm and every library contributes at least one (shortcut: `n`)."
            )

        algorithm_info = self.__algorithm_widgets.get(fid)
        if not algorithm_info:
            return default

        library, coll_name, algo_key, _, algo_meta = algorithm_info
        lib_label = library.name

        desc = algo_meta.get("desc", "No description provided.")
        selection = algo_meta.get("selection")
        bine_imp = algo_meta.get("bine_imp")
        tags = algo_meta.get("tags", [])
        constraints = algo_meta.get("constraints", [])

        extras = []
        if selection is not None:
            extras.append(f"selector value: {selection}")
        if bine_imp is not None:
            extras.append(f"bine_imp: {bine_imp}")
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

        incompatibility = self.__constraint_issue(algo_meta, library)
        if incompatibility:
            extras.append(f"unavailable: {incompatibility}")

        summary = desc
        if extras:
            summary += "\n" + "; ".join(extras)

        return (
            f"{coll_name.capitalize()} · {lib_label}",
            summary
        )


    def _update_next_button_state(self) -> None:
        selected_pairs = {
            (id(library), collective)
            for checkbox in self.query(Checkbox)
            if checkbox.id and checkbox.value and not checkbox.disabled
            for library, collective, _, _, _ in [
                self.__algorithm_widgets[checkbox.id]
            ]
        }
        required_libraries = {id(library) for library in self.session.libraries}
        required_collectives = set(self.__collectives)
        enable_next = has_algorithm_coverage(
            required_libraries,
            required_collectives,
            selected_pairs,
        )
        self.query_one("#next", Button).disabled = not enable_next

    def __store_selections(self) -> None:
        for library in self.session.libraries:
            library.algorithms = {
                CollectiveType.from_str(collective): []
                for collective in self.__collectives
            }

        for checkbox in self.query(Checkbox):
            if not checkbox.id or not checkbox.value or checkbox.disabled:
                continue
            library, collective, algorithm_name, _, metadata = (
                self.__algorithm_widgets[checkbox.id]
            )
            collective_type = CollectiveType.from_str(collective)
            library.algorithms[collective_type].append(
                AlgorithmSelection.from_dict(
                    algorithm_name, collective, metadata
                )
            )

    def __make_algorithm_checkbox(
        self,
        checkbox_id: str,
        library: LibrarySelection,
        collective: str,
        algorithm_name: str,
        metadata: dict,
        *,
        pico: bool,
    ) -> Checkbox:
        self.__algorithm_widgets[checkbox_id] = (
            library,
            collective,
            algorithm_name,
            pico,
            metadata,
        )
        issue = self.__constraint_issue(metadata, library)
        suffix = " (PICO custom)" if pico else ""
        if issue:
            suffix += f" [unavailable: {issue}]"
        selected = self.__was_selected(
            library, collective, algorithm_name, pico
        ) and not issue
        return Checkbox(
            f"({library.name}) {algorithm_name}{suffix}",
            id=checkbox_id,
            value=selected,
            disabled=bool(issue),
        )

    @staticmethod
    def __was_selected(
        library: LibrarySelection,
        collective: str,
        algorithm_name: str,
        pico: bool,
    ) -> bool:
        selected = library.algorithms.get(CollectiveType.from_str(collective), [])
        return any(
            algorithm.name == algorithm_name
            and (algorithm.selection == "pico") == pico
            for algorithm in selected
        )

    def __constraint_issue(self, metadata: dict, library: LibrarySelection):
        communicator_sizes = sorted({
            self.session.test.number_of_nodes * tasks_per_node
            for test_type in (TestType.CPU, TestType.GPU)
            for tasks_per_node in library.tests.get(test_type, [])
        })
        return get_algorithm_constraint_issue(metadata, communicator_sizes, root=0)
