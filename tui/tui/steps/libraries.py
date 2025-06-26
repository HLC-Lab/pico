from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Select, Static, Header, Footer, Button, Checkbox
from .base import StepScreen
from config_loader import lib_get_libraries
from textual.reactive import reactive
from models import LibrarySelection, CollectiveType


class LibRow(Horizontal):
    def __init__(self, index: int, session, libs, used_libs):
        super().__init__()
        self.index = index
        self.session = session
        self.available_libs = libs
        self.used_libs = used_libs

    def compose(self) -> ComposeResult:
        opts = [(lib, lib) for lib in self.available_libs if lib not in self.used_libs]
        yield Horizontal(
            Select(opts, prompt="Library", id=f"lib-sel-{self.index}", classes="field"),
            Horizontal(
                Button("–", id=f"remove-{self.index}", disabled=(self.index == 1)),
                Button("+", id=f"add-{self.index}", disabled=True),
                classes="button-row-task"
            ),
            classes="row-task"
        )

class LibrariesStep(StepScreen):
    __lib_counter = reactive(0)

    def compose(self) -> ComposeResult:

        yield Header(show_clock=True)

        yield Horizontal(
            Static("Library", classes="field-label"),
            Horizontal(
                Static("Remove", classes="field-label"),
                Static("Add", classes="field-label"),
                classes="button-row-task"
            ),
            classes="row-tight"
        )

        self.lib_list_container = Vertical(id="lib-list", classes="lib-list-container")
        yield self.lib_list_container

        yield Static("Select collectives to use in this test", classes='field-label')
        libs = [
            "Allgather",
            "Allreduce",
            "Alltoall",
            "Barrier",
            "Broadcast",
            "Reduce",
            "ReduceScatter",
            "Scatter"
        ]
        self.checkboxes = []
        for idx, lib in enumerate(libs):
            cb = Checkbox(lib, id=f"checkbox-{idx}")
            self.checkboxes.append(cb)

        half = (len(self.checkboxes) + 1) // 2
        left_checkboxes = self.checkboxes[:half]
        right_checkboxes = self.checkboxes[half:]

        yield Horizontal(
            Vertical(*left_checkboxes, classes="checkbox-column"),
            Vertical(*right_checkboxes, classes="checkbox-column"),
            classes="collectives-container"
        )

        yield self.navigation_buttons()

        yield Footer()

    def on_mount(self) -> None:
        self.session.libraries = []
        self.__lib_data = lib_get_libraries(self.session.environment.name)
        self.__available_libs = list(self.__lib_data.get('LIBRARY', {}).keys())
        self.__already_used = []
        self.__next_lib_id = 0
        self.__add_lib()

    def on_select_changed(self, event) -> None:
        sel = event.control
        if sel.id.startswith("lib-sel-"):
            add_button = self.query_one(f"#add-{sel.id.split('-')[-1]}", Button)
            selected_lib = sel.value
            if selected_lib is not Select.BLANK:
                add_button.disabled = False

            self.__update_already_used()

        self.__update_next_button()

    def on_checkbox_changed(self) -> None:
        self.__update_next_button()

    def on_button_pressed(self, event) -> None:
        button = event.control
        if button.id == "next":
            collectives = {CollectiveType.from_str(str(cb.label)) : [] for cb in self.checkboxes if cb.value}
            if not collectives:
                raise ValueError("At least one collective must be selected.")
            for ind, lib in enumerate(self.__already_used):
                library = LibrarySelection.from_dict(self.__lib_data.get('LIBRARY',{}).get(lib), lib)
                library.algorithms = collectives
                self.session.libraries.append(library)
                if not self.session.libraries[ind].validate():
                    raise ValueError(f"Library {lib} is not valid for the current environment.")
            from tui.steps.algorithms import AlgorithmsStep
            self.next(AlgorithmsStep)
        elif button.id == "prev":
            from tui.steps.tasks import TasksStep
            self.prev(TasksStep)
        elif button.id.startswith("add-"):
            self.__add_lib()
        elif button.id.startswith("remove-"):
            index = int(event.button.id.split("-")[1])
            self.__remove_lib(index)


    # TODO:
    def get_help_desc(self):
        return "a","b"

    def __add_lib(self) -> None:
        self.__lib_counter += 1
        self.__next_lib_id += 1
        self.lib_list_container.mount(LibRow(self.__next_lib_id, self.session, self.__available_libs, self.__already_used))

    def __remove_lib(self, index: int):
        sel = self.query_one(f"#lib-sel-{index}", Select)
        if sel.value and sel.value != Select.BLANK:
            self.__already_used.remove(sel.value)
        for child in list(self.lib_list_container.children):
            if isinstance(child, LibRow) and child.index == index:
                child.remove()
                break
        self.__lib_counter -= 1

    def __update_already_used(self) -> None:
        already_used = []
        for lib_row in self.lib_list_container.query(LibRow):
            sel = self.query_one(f"#lib-sel-{lib_row.index}", Select)
            if sel.value and sel.value != Select.BLANK:
                already_used.append(sel.value)

        self.__already_used = already_used
        # self.notify(f"Already used libraries: {self.__already_used}\n"
        #             f"Available libraries: {self.__available_libs}\n"
        #             f"Total libraries: {self.__lib_counter}",
        #             title="Libraries Update", timeout=10, markup=False)

    def __update_next_button(self) -> None:
        next_button = self.query_one("#next", Button)
        enable = False
        if self.__already_used and any(cb.value for cb in self.checkboxes):
            enable = True

        next_button.disabled = not enable

