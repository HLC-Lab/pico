# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Select, Static, Header, Footer, Button, Checkbox, Switch, Input, Label
from .base import StepScreen
from config_loader import lib_get_libraries
from textual.reactive import reactive
from models import LibrarySelection, CollectiveType, TestType
from typing import Optional, Tuple


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
                Vertical(
                    Input(id=f"cpu-tasks-{self.index}", disabled=True, placeholder="CPU tasks"),
                    Label("", id=f"cpu-error-{self.index}", classes="error"),
                    classes="field"
                ),
                Vertical(
                    Input(id=f"gpu-tasks-{self.index}", disabled=True, placeholder="GPU tasks"),
                    Label("", id=f"gpu-error-{self.index}", classes="error"),
                    classes="field"
                ),
                classes="field"
            ),
            #WARN: Visualization problems of buttons when sceen is small
            Horizontal(
                Switch(id=f"pico-backend-{self.index}", disabled=True, classes="switch-compact"),
                Button("–", id=f"remove-{self.index}", disabled=(self.index == 1), classes="btn-grow"),
                Button("+", id=f"add-{self.index}", disabled=True, classes="btn-grow"),
                classes="row-task-actions",
            ),
            classes="row-task"
        )

class LibrariesStep(StepScreen):
    __lib_counter = reactive(0)

    def compose(self) -> ComposeResult:

        yield Header(show_clock=True)

        yield Horizontal(
            Static("Library", classes="field-label field"),
            Horizontal(
                Static("CPU tasks per node", classes="field-label field"),
                Static("GPU tasks per node", classes="field-label field"),
                classes="field"
            ),
            Horizontal(
                Static("Custom collectives?", classes="field-label field"),
                Static("Remove", classes="field-label btn-grow"),
                Static("Add", classes="field-label btn-grow"),
                classes="field"
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
            "Broadcast",
            "Gather",
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
            ind = sel.id.split("-")[-1]
            cpu_input = self.query_one(f"#cpu-tasks-{ind}", Input)
            gpu_input = self.query_one(f"#gpu-tasks-{ind}", Input)
            add_button = self.query_one(f"#add-{ind}", Button)
            pico_switch = self.query_one(f"#pico-backend-{ind}", Switch)
            selected_lib = sel.value

            self.__enable_inputs(cpu_input, gpu_input, selected_lib)

            disabled = (selected_lib is Select.BLANK)
            add_button.disabled = disabled
            pico_switch.disabled = disabled
            if disabled:
                pico_switch.value = False


        self.__update_already_used()
        self.__update_next_button()

    def on_input_changed(self, event) -> None:
        inp = event.control
        if not (isinstance(inp, Input) and inp.id):
            return

        idx = inp.id.split("-")[-1]



        min, max = 1, None
        if inp.id == f"cpu-tasks-{idx}":
            error = self.query_one(f"#cpu-error-{idx}", Label)
            max = self.session.environment.partition.cpus_per_node if self.session.environment.slurm else None
        elif inp.id == f"gpu-tasks-{idx}":
            error = self.query_one(f"#gpu-error-{idx}", Label)
            max = self.session.environment.partition.gpus_per_node if self.session.environment.slurm else None
        else:
            raise ValueError(f"Unknown input ID: {inp.id}")

        _, log = self.__validate_input(inp.value, min, max)
        error.update(log)
        self.__update_next_button()


    def on_checkbox_changed(self) -> None:
        self.__update_next_button()

    def on_button_pressed(self, event) -> None:
        button = event.control
        if button.id == "next":
            collectives = {
                CollectiveType.from_str(str(cb.label)) : [] 
                for cb in self.checkboxes if cb.value
            }
            if not collectives:
                raise ValueError("At least one collective must be selected.")



            for lib_row in self.lib_list_container.query(LibRow):
                sel = lib_row.query_one(Select)
                pico_switch = lib_row.query_one(Switch)
                inputs = lib_row.query(Input)
                cpu_input, gpu_input = inputs
                if not sel.value or sel.value is Select.BLANK:
                    continue

                lib_name = str(sel.value)
                data = self.__lib_data["LIBRARY"][lib_name]
                library = LibrarySelection.from_dict(data, lib_name)
                library.algorithms = collectives
                library.pico_backend = pico_switch.value
                tests = {}
                if cpu_input.value:
                    tests[TestType.CPU] = [int(x.strip()) for x in cpu_input.value.split(',')]
                if gpu_input.value:
                    tests[TestType.GPU] = [int(x.strip()) for x in gpu_input.value.split(',')]
                library.tests = tests
                self.session.libraries.append(library)
                if not library.validate():
                    raise ValueError(f"Library {lib_name} is not valid for the current environment.")


            from tui.steps.algorithms import AlgorithmsStep
            self.next(AlgorithmsStep)

        elif button.id == "prev":
            from tui.steps.configure import ConfigureStep
            self.prev(ConfigureStep)

        elif button.id.startswith("add-"):
            self.__add_lib()

        elif button.id.startswith("remove-"):
            index = int(event.button.id.split("-")[1])
            self.__remove_lib(index)

        self.__update_next_button()


    def get_help_desc(self) -> Tuple[str, str]:
        focused = self.focused
        default = (
            "Library Setup",
            "Pick one or more MPI/NCCL libraries, specify CPU/GPU tasks per node, and enable the collectives you want to benchmark."
        )

        if not focused or not getattr(focused, "id", None):
            return default

        fid = focused.id
        env = self.session.environment
        part = env.partition if env else None
        qos = part.qos if part else None

        def partition_limits(kind: str) -> str:
            if not (env and env.slurm and part):
                return "Local mode expects comma-separated positive integers."
            if kind == "cpu":
                limit = part.cpus_per_node
                return f"Comma-separated task counts (1–{limit}) per node."
            if part.gpus_per_node is not None:
                limit = part.gpus_per_node
                return f"Comma-separated GPU task counts (1–{limit}) per node."
            return "This partition exposes no GPUs."

        if fid.startswith("lib-sel-"):
            idx = fid.split("-")[-1]
            select = self.query_one(f"#lib-sel-{idx}", Select)
            current = select.value if (select and select.value is not Select.BLANK) else None
            lib_repo = getattr(self, "_LibrariesStep__lib_data", {}).get("LIBRARY", {})
            if current:
                meta = lib_repo.get(current, {})
                desc = meta.get("desc", "No description available.")
                version = meta.get("version", "unknown")
                compiler = meta.get("compiler", "unspecified toolchain")
                gpu_support = "GPU support enabled" if meta.get("gpu", {}).get("support") else "CPU-only"
                return (
                    f"Library {current}",
                    f"{desc}\nVersion: {version} | Compiler: {compiler}\n{gpu_support}."
                )
            available_libs = getattr(self, "_LibrariesStep__available_libs", [])
            available = ", ".join(available_libs) if available_libs else "No libraries configured for this environment."
            return (
                "Select Library",
                f"Choose one of: {available}."
            )

        if fid.startswith("cpu-tasks-"):
            return (
                "CPU Tasks per Node",
                partition_limits("cpu")
            )

        if fid.startswith("gpu-tasks-"):
            return (
                "GPU Tasks per Node",
                partition_limits("gpu")
            )

        if fid.startswith("pico-backend-"):
            switch = self.query_one(f"#{fid}", Switch)
            state = "enabled" if switch and switch.value else "disabled"
            return (
                "PICO Custom Backend",
                f"PICO-specific algorithm metadata for this library is currently {state}. Enable to expose LibPico-only algorithms in the next step."
            )

        if fid.startswith("add-"):
            return (
                "Add Library Row",
                "Create another library entry to mix multiple MPI/NCCL stacks in a single test."
            )

        if fid.startswith("remove-"):
            return (
                "Remove Library Row",
                "Delete this library configuration. At least one configured library must remain."
            )

        if fid.startswith("checkbox-"):
            checkbox = focused
            label = str(getattr(checkbox, "label", "Collective"))
            selected = "enabled" if checkbox.value else "disabled"
            return (
                "Collective Selection",
                f"{label} is currently {selected}. Tick at least one collective to move forward."
            )

        if fid == "prev":
            return (
                "Previous Step",
                "Return to environment and test configuration (shortcut: `p`)."
            )

        if fid == "next":
            return (
                "Next Step",
                "Proceed to choose per-collective algorithms once every row is fully configured (shortcut: `n`)."
            )

        return default

    def __add_lib(self) -> None:
        self.__update_already_used()
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
        self.__update_already_used()

    def __update_already_used(self) -> None:
        already_used = []
        for lib_row in self.lib_list_container.query(LibRow):
            sel = self.query_one(f"#lib-sel-{lib_row.index}", Select)
            if sel.value and sel.value != Select.BLANK:
                already_used.append(sel.value)

        self.__already_used = already_used


    def __update_next_button(self) -> None:
        next_button = self.query_one("#next", Button)

        has_libs = bool(self.__already_used)
        has_coll = any(cb.value for cb in self.checkboxes)
        if not (has_libs and has_coll):
            next_button.disabled = True
            return

        for lib_row in self.lib_list_container.query(LibRow):
            sel = lib_row.query_one(Select)
            if not sel.value or sel.value is Select.BLANK:
                continue

            cpu_input, gpu_input = lib_row.query(Input)
            valid_found = False
            for inp in (cpu_input, gpu_input):
                val = inp.value or ""
                if not val or inp.disabled:
                    continue

                min_, max_ = 1, None
                if self.session.environment.slurm:
                    if inp.placeholder == "CPU tasks":
                        max_ = self.session.environment.partition.cpus_per_node
                    else:
                        max_ = self.session.environment.partition.gpus_per_node

                is_valid, _ = self.__validate_input(val, min_, max_)
                if not is_valid:
                    next_button.disabled = True
                    return

                valid_found = True

            if not valid_found:
                next_button.disabled = True
                return

        next_button.disabled = False


    def __enable_inputs(self, cpu_input: Input, gpu_input: Input, selected_lib) -> None:
        if selected_lib == Select.BLANK:
            self.reset_input(cpu_input)
            self.reset_input(gpu_input)
            return
        data = self.__lib_data["LIBRARY"][selected_lib]
        gpu_support = data.get("gpu", {}).get("support", False)
        cpu_support = True if "nccl" not in selected_lib.lower() else False
        cpu_input.disabled = not cpu_support
        gpu_input.disabled = not gpu_support

    @staticmethod
    def __validate_input(s: str, min_val: int, max_val: Optional[int] = None) -> tuple[bool, str]:
        if not s:
            return False, ""

        try:
            parts = s.split(',')
            nums = [int(p.strip()) for p in parts]
        except ValueError:
            return False, "All items must be integers."

        if any(n <= 0 for n in nums):
            return False, "All numbers must be positive integers."

        if len(nums) != len(set(nums)):
            return False, "Numbers must not repeat."

        if any(n < min_val for n in nums):
            return False, f"All numbers must be >= {min_val}."

        if max_val is not None and any(n > max_val for n in nums):
            return False, f"All numbers must be <= {max_val}."

        return True, ""
