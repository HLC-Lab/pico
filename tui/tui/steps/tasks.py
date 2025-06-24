from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, Button, Label, Header, Footer, Select, Switch
from textual.reactive import reactive
from .base import StepScreen
from models import TaskConfig, limits
from typing import List, Dict
import math

class TaskRow(Horizontal):
    def __init__(self, index: int, session):
        super().__init__()
        self.index = index
        self.session = session

    def compose(self) -> ComposeResult:
        use_gpu = self.session.compile.use_gpu_buffers
        opts = [("cpu buffers", "cpu"), ("gpu buffers", "gpu")] if use_gpu else [("cpu buffers", "cpu")]
        yield Horizontal(
            Vertical(
                Select(opts, prompt="Type of Test", id=f"task-type-{self.index}"),
                classes="field-small"
            ),
            Vertical(
                Input(placeholder="# processes", id=f"task-input-{self.index}", disabled=True),
                Label("", id=f"task-error-{self.index}", classes="error"),
                classes="field"
            ),
            Horizontal(
                Button("–", id=f"remove-{self.index}", disabled=(self.index == 1)),
                Button("+", id=f"add-{self.index}", disabled=True),
                classes="button-row-task"
            ),
            classes="row-task"
        )

class TasksStep(StepScreen):
    __task_counter = reactive(0)
    __lim : limits


    def compose(self):
        if self.has_slurm:
            part = self.session.environment.partition
            qos = part.qos
            self.__lim = limits(
                min_nodes=qos.nodes_limit.get("min"),
                max_nodes=qos.nodes_limit.get("max"),
                max_cpu_tasks=part.cpus_per_node,
                max_gpu_tasks=part.sockets_per_node,
                max_time=qos.time_limit,
            )
        else:
            self.__lim = limits(1, 1, 1, 1, "")

        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("Number of Nodes",  classes="field-label"),
                Input(placeholder=f"min {self.__lim.min_nodes}, max {self.__lim.max_nodes}" if self.has_slurm else "You have just 1 node",
                      disabled=  not self.has_slurm, id="nodes-input"),
                Label("", id="nodes-error", classes="error"),
                classes="field",
            ),
            Vertical(
                Static("Exclude Nodes", classes="field-label"),
                Switch(id="exclude-switch", value=False, disabled=not self.has_slurm),
                classes="switch-col",
            ),
            Vertical(
                Static("Excluded Nodes", classes="field-label"),
                Input(placeholder="What nodes do you want to exclude?" if self.has_slurm else "Wow buddy, you have just one node and you don't want it?",
                      id="excluded-nodes", disabled=True),
                Label("", id="excluded-nodes-error", classes="error"),
                classes="field",
            ),
            classes="row",
        )

        yield Horizontal(
            Vertical(
                Static("Test Time", classes="field-label"),
                Input(placeholder=f"HH:MM:SS (max {self.__lim.max_time})" if self.has_slurm else "No SLURM time limits on local machine",
                      id="time-input", disabled=not self.has_slurm),
                Label("", id="time-error", classes="error")
            ),
            Vertical(
                Static("Start After", classes="field-label"),
                Switch(id="dep-switch", value=False, disabled=not self.has_slurm),
                classes="switch-col",
            ),
            Vertical(
                Static("Job ID for AfterAny", classes="field-label"),
                Input(placeholder="Insert here job ID" if self.has_slurm else "No SLURM job dep on local machine",
                      id="dep-input", disabled=True),
                Label("", id="dep-error", classes="error"),
                classes="field",
            ),
            classes="row"
        )

        yield Horizontal(
            Static("Type of Task", classes="field-label-small"),
            Static("Number of Tasks per Node" if self.has_slurm else "Number of Tasks", classes="field-label-center"),
            Horizontal(
                Static("Remove", classes="field-label"),
                Static("Add", classes="field-label"),
                classes="button-row-task-static"
            ),
            classes="row-tight"
        )

        self.task_list_container = Vertical(id="task-list", classes="task-list-container")
        yield self.task_list_container

        yield self.navigation_buttons()

        yield Footer()

    def on_mount(self):
        self.__enable_next = {
            "nodes": False,
            "time": False,
            "exclude": True,
            "dep": True,
            "tasks": False,
        } if self.has_slurm else {"tasks": False}
        self.session.tasks = TaskConfig()
        self.__already_used: Dict[str, List[int]] = {}
        self.__next_task_id = 0
        self.__add_task()


    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id and event.button.id.startswith("add-"):
            self.__add_task()
        elif event.button.id and event.button.id.startswith("remove-"):
            index = int(event.button.id.split("-")[1])
            self.__remove_task(index)
        elif event.button.id == "prev":
            from tui.steps.configure import ConfigureStep
            self.prev(ConfigureStep)
        elif event.button.id == "next":
            self.session.tasks.list_tasks_from_dict(self.__already_used)
            if self.session.tasks.validate(self.session):
                from tui.steps.libraries import LibrariesStep
                self.next(LibrariesStep)
            else:
                self.notify("Invalid task configuration. Please check your inputs.", title="Error", severity="error", timeout=10)


    def on_select_changed(self, event: Select.Changed):
        sel = event.control
        val = event.value

        index = int(sel.id.split("-")[2]) if sel.id else -1
        if index == -1:
            raise ValueError("Select control ID does not contain an index.")
        task_input = self.query_one(f"#task-input-{index}", Input)

        if val is Select.BLANK:
            self.reset_input(task_input, disable=True)
            self.__enable_next["tasks"] = False
            self.__update_next_button()
            return

        task_input.disabled = False
        max_tasks = { "cpu" : self.__lim.max_cpu_tasks,
                      "gpu" : self.__lim.max_gpu_tasks }.get(str(val))
        if max_tasks is None:
            raise ValueError(f"Unknown task type: {val}")

        task_input.placeholder = f"min 1, max {max}"



    def on_input_changed(self, event):
        value = event.input.value
        if event.input.id == "nodes-input":
            error_label = self.query_one("#nodes-error", Label)
            if self.session.tasks.validate_nodes(self.session, value):
                error_label.update("")
                self.session.tasks.number_of_nodes = int(value)
                self.__enable_next["nodes"] = True
            else:
                error_label.update(f"Invalid number of nodes")
                self.session.tasks.number_of_nodes = 1
                self.__enable_next["nodes"] = False

        elif event.input.id == "time-input":
            error_label = self.query_one("#time-error", Label)
            if self.session.tasks.validate_time(self.session, value):
                error_label.update("")
                self.session.tasks.test_time = value
                self.__enable_next["time"] = True
            else:
                error_label.update(f"Invalid time (format DD-HH:MM:SS or HH:MM:SS, maximum {self.__lim.max_time})")
                self.session.tasks.test_time = ""
                self.__enable_next["time"] = False

        elif event.input.id == "excluded-nodes":
            error_label = self.query_one("#excluded-nodes-error", Label)
            if value != "":
                error_label.update("")
                self.session.tasks.exclude_nodes = value
                self.__enable_next["exclude"] = True
            else:
                error_label.update("No nodes excluded")
                self.session.tasks.exclude_nodes = None
                self.__enable_next["exclude"] = False

        elif event.input.id == "dep-input":
            error_label = self.query_one("#dep-error", Label)
            if value.isdigit():
                error_label.update("")
                self.session.tasks.job_dependency = int(value)
                self.__enable_next["dep"] = True
            else:
                error_label.update("Invalid job ID")
                self.session.tasks.job_dependency = None
                self.__enable_next["dep"] = False

        elif event.input.id and event.input.id.startswith("task-input-"):
            index = int(event.input.id.split("-")[2])
            task_type = self.query_one(f"#task-type-{index}", Select).value
            add_button = self.query_one(f"#add-{index}", Button)

            if self.__validate_task(value, index, str(task_type)):
                add_button.disabled = False
                self.__enable_next["tasks"] = True
                self.__update_already_used()
            else:
                self.__enable_next["tasks"] = False
                add_button.disabled = True

        self.__update_next_button()

    def on_switch_changed(self, event: Switch.Changed):
        if event.control.id == "exclude-switch":
            excluded_nodes_input = self.query_one("#excluded-nodes", Input)
            excluded_nodes_error = self.query_one("#excluded-nodes-error", Label)
            excluded_nodes_input.disabled = not event.value
            self.__enable_next["exclude"] = not event.value
            if not event.value:
                excluded_nodes_input.value = ""
                excluded_nodes_error.update("")
        elif event.control.id == "dep-switch":
            dep_input = self.query_one("#dep-input", Input)
            dep_error = self.query_one("#dep-error", Label)
            dep_input.disabled = not event.value
            self.__enable_next["dep"] = not event.value
            if not event.value:
                dep_input.value = ""
                dep_error.update("")

        self.__update_next_button()

    def get_help_desc(self):
        return "a","b"

    # -------------- Helper Methods for Task Row management -----------------
    def __add_task(self):
        self.__task_counter += 1
        self.__next_task_id += 1
        self.task_list_container.mount(TaskRow(self.__next_task_id, self.session))

    def __remove_task(self, index: int):
        sel = self.query_one(f"#task-type-{index}", Select)
        inp = self.query_one(f"#task-input-{index}", Input)
        if sel and inp:
            ttype = str(sel.value)
            val   = inp.value
            if val and val in self.__already_used.get(ttype, []):
                self.__already_used[ttype].remove(val)

        for child in list(self.task_list_container.children):
            if isinstance(child, TaskRow) and child.index == index:
                child.remove()
                break
        self.__task_counter -= 1

    def __update_already_used(self):
        used: dict[str, list[int]] = {}
        for child in self.task_list_container.children:
            if not child.is_attached:
                continue
            if isinstance(child, TaskRow):
                idx = child.index
                ttype = self.query_one(f"#task-type-{idx}", Select).value
                val = self.query_one(f"#task-input-{idx}", Input).value
                if val and val.isdigit():
                    used.setdefault(str(ttype), []).append(int(val))

        self.__already_used = used

    def __validate_task(self, task_input: str, index: int, task_type: str) -> bool:
        if self.has_slurm:
            task_limits = {
                "cpu": self.__lim.max_cpu_tasks,
                "gpu": self.__lim.max_gpu_tasks,
            }

            try:
                max_tasks = task_limits[task_type]
            except KeyError:
                raise ValueError(f"Unknown task type: {task_type!r}")
        else:
            max_tasks = math.inf

        error_label = self.query_one(f"#task-error-{index}", Label)

        if not task_input.isdigit():
            error_label.update("Input must be a positive integer")
            return False

        task_value = int(task_input)
        if not (1 <= task_value <= max_tasks):
            error_label.update(f"Value must be between 1 and {int(max_tasks) if max_tasks != math.inf else '∞'}")
            return False

        if task_value in self.__already_used.get(task_type, []):
            error_label.update("Number of processes already used")
            return False

        error_label.update("")
        return True

    def __update_next_button(self):
        next_button = self.query_one("#next", Button)
        is_valid = all(self.__enable_next.values())
        next_button.disabled = not is_valid
