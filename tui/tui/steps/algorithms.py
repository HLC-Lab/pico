from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Button, Checkbox, TabbedContent, TabPane, Header, Footer
from .base import StepScreen
from config_loader import alg_get_list, alg_get_algo
from models import CollectiveType, AlgorithmSelection
from typing import List, Tuple
from packaging import version


class AlgorithmsStep(StepScreen):
    __libs: List[Tuple]
    __collectives: List[str]

    def compose(self) -> ComposeResult:
        self.__libs = [(lib.name, lib.get_type(), lib.get_id_name(), lib.pico_backend) for lib in self.session.libraries]
        self.__collectives = [str(key) for key in self.session.libraries[0].algorithms.keys()]

        yield Header(show_clock=True)

        yield Static("Select Algorithms for Each Collective", classes="field-label")

        with TabbedContent():
            for pane_num, coll in enumerate(self.__collectives):
                with TabPane(title=f"({pane_num+1}) {coll.capitalize()}", id=f"tab-{coll}"):
                    columns = []
                    for idx, (lib_name, lib_type, lib_name_id, pico) in enumerate(self.__libs):
                        lib_version = self.session.libraries[idx].version
                        algos = alg_get_list(lib_type, coll)
                        # BUG: Selection screen crashes, moreover chechboxes are not scrollable
                        if pico:
                            pico_algos = alg_get_list("PicoLib-MPI", coll)
                            algos.update(pico_algos)
                        columns.append(Vertical(*[
                            Checkbox(
                                f"({lib_name}) {key}",
                                id=f"{coll}-{key}-{lib_name_id}"
                            )
                            for key, meta in algos.items()
                            if (ver := meta.get("version", None))
                            and version.parse(ver) <= version.parse(lib_version)
                        ]))
                    yield Horizontal(*columns)

        yield self.navigation_buttons()

        yield Footer()


    def on_mount(self) -> None:
        for lib in self.session.libraries:
            for key in lib.algorithms:
                lib.algorithms[key].clear()
        self.__libs_ok = { lib[2] : False for lib in self.__libs }
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

                coll_str, algo_key, lib_id = cb.id.split("-", 2)
                collective = CollectiveType.from_str(coll_str)

                library = next(
                    lib for lib in self.session.libraries
                    if lib.get_id_name() == lib_id
                )

                algo_data = alg_get_algo( library.get_type(), coll_str, algo_key )
                if not algo_data:
                    raise ValueError(f"Algorithm {algo_key} not found in {library.get_type}/{coll_str}.json")

                library.algorithms[collective].append(
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


    # TODO:
    def get_help_desc(self):
        return "a","b"


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
                if cb.id and cb.id.endswith(f"-{lib}")
            )
            if found:
                self.__libs_ok[lib] = True

        enable_next = all(self.__coll_ok.values()) and all(self.__libs_ok.values())
        self.query_one("#next", Button).disabled = not enable_next
