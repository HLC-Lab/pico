# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Select, Switch, Footer, Header, Button, Input, SelectionList, Label
from textual.widgets.selection_list import Selection
from .base import StepScreen
from config_loader import conf_list_environments, conf_get_general, conf_get_slurm_opts
from models import SessionConfig, EnvironmentSelection, PartitionSelection, TestDimension, CDtype, OutputLevel
from typing import Tuple, Optional

class ConfigureStep(StepScreen):
    __buffer_sizes =  ["32  Byte", "256 Byte", "2   KiB", "16  KiB", "128 KiB", "1   MiB", "8   MiB", "64  MiB", "512 MiB"]
    __segment_sizes = ["0   Byte", "16  KiB", "128 KiB", "1   MiB"]
    def compose(self):
        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("Environment:", classes="field-label"),
                Select([(e, e) for e in conf_list_environments()], prompt="Environment:", id="env-select")
            ),
            Vertical(
                Static("Partition:", classes="field-label"),
                Select([], prompt="Partition:", id="partition-select", disabled=True)
            ),
            Vertical(
                Static("QOS:", classes="field-label"),
                Select([], prompt="QOS:", id="qos-select", disabled=True)
            ),
            classes="row"
        )


        yield Horizontal(
            Vertical(
                Static("Compile Only", classes="field-label"),
                Switch(id="compile-switch", value=False)
            ),
            Vertical(
                Static("Debug Mode", classes="field-label"),
                Switch(id="debug-switch", value=False)
            ),
            Vertical(
                Static("Dry Run Mode", classes="field-label"),
                Switch(id="dry-switch", value=False)
            ),
            Horizontal(
                Vertical(
                    Static("Number of Nodes",  classes="field-label"),
                    Input(placeholder=f"Insert number of nodes",
                        disabled=True, id="nodes-input"),
                    Label("", id="nodes-error", classes="error"),
                    classes="field",
                ),
                Vertical(
                    Static("Test Time", classes="field-label"),
                    Input(placeholder=f"Insert time in HH:MM:SS)",
                        id="time-input", disabled=True),
                    Label("", id="time-error", classes="error"),
                    classes="field"
                ),
            ),
            classes="tight-switches"
        )

        yield Horizontal(
            Vertical(
                Static("Exclude Nodes", classes="field-label"),
                Switch(id="exclude-switch", value=False, disabled=True),
                classes="switch-col",
            ),
            Vertical(
                Static(" ", classes="field-label"),
                Input(placeholder="What nodes do you want to exclude?",
                    id="excluded-nodes", disabled=True),
                Label("", id="excluded-nodes-error", classes="error"),
                classes="field",
            ),
            Vertical(
                Static("Start After", classes="field-label"),
                Switch(id="dep-switch", value=False, disabled=True),
                classes="switch-col",
            ),
            Vertical(
                Static(" ", classes="field-label"),
                Input(placeholder="Insert here job ID",
                    id="dep-input", disabled=True),
                Label("", id="dep-error", classes="error"),
                classes="field",
            ),
            Vertical(
                Static("Inject Params", classes="field-label"),
                Switch(id="inject-switch", value=False),
                classes="switch-col"
            ),
            Vertical(
                Static(" ", classes="field-label"),
                Input(placeholder="Insert here any sbatch param or env",
                      id="inject-params", disabled=True),
                classes="field",
            ),
            classes="row"
        )


        dtypes = [
            ("char", CDtype.CHAR),
            ("int8", CDtype.INT8),
            ("int16", CDtype.INT16),
            ("int32", CDtype.INT32),
            ("int64", CDtype.INT64),
            ("float", CDtype.FLOAT),
            ("double", CDtype.DOUBLE)
        ]

        output_lev = [
            ("Full", OutputLevel.FULL),
            ("Statistics", OutputLevel.STATISTICS),
            ("Minimal", OutputLevel.MINIMAL),
            ("Summary", OutputLevel.SUMMARY),
        ]

        buffer_items =  [Selection(f"{label.replace('Byte', '  B')}", self.__parse_size(label), True) for label in self.__buffer_sizes]
        segment_items = [Selection(f"{label.replace('Byte', '  B')}", self.__parse_size(label)) for label in self.__segment_sizes]
        segment_items[0] = Selection("No Segment", 0, True)

        yield Horizontal(
            Vertical(
                Vertical(
                    Static("Data Type", classes = "field-label"),
                    Select( dtypes, prompt="Select Data Type", id="data-type-select", value=CDtype.INT32)
                ),
                Vertical(
                    Static("Output Level", classes="field-label"),
                    Select( output_lev, id="output-select", prompt="Select Output Level", value=OutputLevel.STATISTICS)
                ),
                classes="field-small"
            ),
            Vertical(
                Vertical(
                    Static("Compress Res.", classes="field-label"),
                    Switch(id="compress-switch", value=True),
                ),
                Vertical(
                    Static("Delete Uncompr.", classes="field-label"),
                    Switch(id="delete-switch", value=True),
                ),
                classes="field-mini"
            ),
            Vertical(
                Static("Buffer Sizes", classes="field-label"),
                SelectionList[int](
                    *buffer_items,
                    id="buffer-size-select"
                ),
                classes="field"
            ),
            Vertical(
                Static("Segment Sizes", classes="field-label"),
                SelectionList[int](
                    *segment_items,
                    id="segment-size-select"
                ),
                classes="field"
            ),
        )

        yield self.navigation_buttons(prev_disabled=True)

        yield Footer()

    def on_mount(self):
        self.session = SessionConfig()
        self.__slurm_opts = {}

    # TODO: GPU switch does not exist anymore
    def on_select_changed(self, event):
        sel = event.control
        part_w = self.query_one("#partition-select", Select)
        qos_w = self.query_one("#qos-select", Select)

        if sel.id == "env-select":
            env = event.value
            self.reset_select(part_w)
            self.reset_select(qos_w)

            self.session.environment = EnvironmentSelection()
            self.__slurm_opts = {}

            if env is not Select.BLANK:
                env_json = conf_get_general(env)
                self.session.environment.from_dict(env_json)

                if self.has_slurm:
                    self.__slurm_opts = conf_get_slurm_opts(env)
                    part_w.set_options([(p, p) for p in self.__slurm_opts["PARTITIONS"]])
                    part_w.disabled = False

        elif sel.id == "partition-select":
            self.reset_select(qos_w)
            self.session.environment.init_partition()

            if event.value is not Select.BLANK:
                if not isinstance(self.session.environment.partition, PartitionSelection):
                    raise ValueError("Partition must be a PartitionSelection instance.")
                self.session.environment.partition.from_dict(self.__slurm_opts, event.value)
                self.session.environment.partition.init_qos()

                # Populate QOS
                qos_w.set_options([(q, q) for q in self.__slurm_opts["PARTITIONS"][event.value]["QOS"]])
                qos_w.disabled = False


        # QOS changed
        elif sel.id == "qos-select":
            if not isinstance(self.session.environment.partition, PartitionSelection):
                raise ValueError("Partition must be a PartitionSelection instance.")
            self.session.environment.partition.init_qos()

            if event.value is not Select.BLANK:
                self.session.environment.partition.qos.from_dict(self.__slurm_opts, event.value)

        # Data type changed
        elif event.select.id == "data-type-select":
            dtype = event.value
            if not self.session.test.dimensions:
                self.session.test.dimensions = TestDimension(dtype=dtype if isinstance(dtype, CDtype) else CDtype.UNKNOWN)

        #NOTE: To avoid zeroing inputs when changing output level or data types
        if event.select.id not in ("data-type-select", "output-select"):
            self.__enable_qos_dep_widgets()

        self.__update_test_selections()
        self.__label_selection_list()
        self.__update_next()

    def on_input_changed(self, event):
        self.__update_test_selections()
        value = event.input.value
        if event.input.id == "nodes-input":
            error_label = self.query_one("#nodes-error", Label)
            error_input = self.query_one("#nodes-input", Input)
            error_label.update("")
            if not self.session.test.validate_nodes(self.session, value) and not error_input.disabled:
                min, max = self.__get_nodes_limit()
                error_label.update(f"Invalid number of nodes: min {min}, max {max}.")
        elif event.input.id == "time-input":
            error_label = self.query_one("#time-error", Label)
            error_input = self.query_one("#time-input", Input)
            error_label.update("")
            if not self.session.test.validate_time(self.session, value) and not error_input.disabled:
                max_time = self.__get_time_limit()
                error_label.update(f"Invalid time (format DD-HH:MM:SS or HH:MM:SS, maximum {max_time})" if max_time else "")

        self.__update_next()



    def on_button_pressed(self, event):
        if event.button.id == 'next':
            if not self.session.test.validate(self.session):
                raise ValueError("Test configuration is not valid.")
            if not self.session.environment.validate():
                raise ValueError("Environment configuration is not valid.")
            from tui.steps.libraries import LibrariesStep
            self.next(LibrariesStep)

    # NOTE: Cannot make the Selection List change work without the decorator,
    # on_selection_list_changed does not get called.
    @on(SelectionList.SelectedChanged)
    def sel_list_handler(self):
        self.__update_test_selections()
        self.__update_next()

    def on_switch_changed(self, event):
        switch = event.control
        value = switch.value

        if switch.id == "compile-switch":
            self.__enable_compile_dep_widgets()
        elif switch.id == "dry-switch":
            comp_switch = self.query_one("#compile-switch", Switch)
            comp_switch.disabled = value
            if value:
                comp_switch.value = False
        elif switch.id == "exclude-switch":
            self.query_one("#excluded-nodes", Input).disabled = not value
        elif switch.id == "dep-switch":
            self.query_one("#dep-input", Input).disabled = not value
        elif switch.id == "inject-switch":
            self.query_one("#inject-params", Input).disabled = not value
        elif switch.id == "compress-switch":
            del_sw = self.query_one("#delete-switch", Switch)
            del_sw.disabled = not value
            if not value:
                del_sw.value = False


        self.__update_test_selections()
        self.__update_next()

    def get_help_desc(self) -> Tuple[str,str]:
        focused = self.focused
        field_desc = "Unknown Field"
        chosen_desc = "No selection"

        field_map = {
            "env-select": (
                "Select the environment for your test.",
                self.session.environment.get_help() if self.session.environment else "No environment selected."
            ),
            "partition-select": (
                "Select the partition for your test.",
                self.session.environment.partition.get_help() if self.session.environment.partition else "No partition selected."
            ),
            "qos-select": (
                "Select the QOS for your test.",
                self.session.environment.partition.qos.get_help() if self.session.environment.partition and self.session.environment.partition.qos else "No QOS selected."
            ),
            #TODO: Add info
            "compile-switch": (
                "Compile Only toggle",
                "Enables compile-only mode without running tests. Not compatible with Dry Run."
            ),
            "debug-switch": (
                "Debug Mode toggle",
                "Compiles in debug mode. \n" \
                "Debug mode:\n" \
                "    - --time is set to 00:10:00\n"\
                "    - Run only one iteration for each test instance.\n"\
                "    - Compile with -g -DDEBUG without optimization.\n"\
                "    - Do not save results (--compress and --delete are ignored)."
            ),
            "dry-switch": (
                "Dry Run Mode toggle",
                "Dry run mode. Test the script without running the actual bench tests."
            ),
            # "gpu-switch": (
            #     "GPU Buffers toggle",
            #     "Enables GPU buffers for the test. Requires a GPU partition.\n"\
            #     "This option is used also for compilation of GPU exec, so it's compatible with Compile Only"
            # ),
            "inject-params": (
                "Extra sbatch params or env to set",
                "Insert any sbatch parameter or environment variable here. \n" \
                "Example: --gres=gpu:1 or MY_ENV_VAR=value. \n"\
                "BEWARE: These will be added to the sbatch command line and there is NO CHECK for correctness done by the script over those params."
            ),
            #TODO: Add help for data type, buffer size, and segment size
        }

        if focused and focused.id in field_map:
            field_desc, chosen_desc = field_map[focused.id]

        return (field_desc, chosen_desc)

    def __parse_size(self, size_label: str) -> int:
        suffixes = {"Byte": 1, "KiB": 1024, "MiB": 1024**2}
        for suffix, factor in suffixes.items():
            if size_label.endswith(suffix):
                return int(size_label.replace(suffix, "")) * factor
        return int(size_label)

    def __label_selection_list(self):
        buf_list = self.query_one("#buffer-size-select", SelectionList)
        if not self.session.test.dimensions:
            return

        dt_size = CDtype.get_size(self.session.test.dimensions.dtype)
        if dt_size <= 0:
            return

        selected_values = buf_list.selected
        buf_list.clear_options()

        for byte_label in self.__buffer_sizes:
            raw_size = self.__parse_size(byte_label)
            text_label = byte_label.replace("Byte", "  B")
            element_count = raw_size // dt_size

            pretty_label = f"{text_label:<10} — {element_count:>10} elements"
            was_selected = raw_size in selected_values
            buf_list.add_option(Selection(pretty_label, raw_size, was_selected))


    #WARN: Full of boilerplate, but goes like this for now
    def __update_test_selections(self) -> None:
        test = self.session.test

        compile_switch = self.query_one("#compile-switch", Switch)
        compile_value = compile_switch.value if not compile_switch.disabled else False

        debug_switch = self.query_one("#debug-switch", Switch)
        debug_value = debug_switch.value if not debug_switch.disabled else False

        dry_switch = self.query_one("#dry-switch", Switch)
        dry_value = dry_switch.value if not dry_switch.disabled else False

        nodes_input = self.query_one("#nodes-input", Input)
        nodes_value = int(nodes_input.value) if nodes_input.value and nodes_input.value.isdigit() and not nodes_input.disabled else 1

        time_input = self.query_one("#time-input", Input)
        time_value = time_input.value if time_input.value and not time_input.disabled else None

        exclude_switch = self.query_one("#exclude-switch", Switch)
        exclude_switch_val = exclude_switch.value if not exclude_switch.disabled else False
        exclude_input = self.query_one("#excluded-nodes", Input)
        exclude_value = exclude_input.value if exclude_switch_val and not exclude_input.disabled else None

        dep_switch = self.query_one("#dep-switch", Switch)
        dep_switch_value = dep_switch.value if not dep_switch.disabled else False
        dep_input = self.query_one("#dep-input", Input)
        dep_value = dep_input.value if dep_switch_value and not dep_input.disabled else None

        inject_switch = self.query_one("#inject-switch", Switch)
        inject_switch_value = inject_switch.value if not inject_switch.disabled else False
        inject_input = self.query_one("#inject-params", Input)
        inject_value = inject_input.value if inject_switch_value and not inject_input.disabled else None

        compress_switch = self.query_one("#compress-switch", Switch)
        compress_value = compress_switch.value if not compress_switch.disabled else False

        delete_switch = self.query_one("#delete-switch", Switch)
        delete_value = delete_switch.value if not delete_switch.disabled else False

        output_select = self.query_one("#output-select", Select)
        output_select = output_select.value if not output_select.disabled else None

        test.compile_only = compile_value
        test.debug_mode = debug_value
        test.dry_run = dry_value
        test.number_of_nodes = nodes_value
        test.test_time = time_value
        test.exclude_nodes = exclude_value
        test.job_dependency = dep_value
        test.inject_params = inject_value
        test.compress = compress_value
        test.delete = delete_value
        test.output_level = output_select if isinstance(output_select, OutputLevel) else None

        if test.compile_only:
            test.dimensions = None
            test.output_level = None
            return

        dt_sel  = self.query_one("#data-type-select", Select).value
        buf_sel = self.query_one("#buffer-size-select", SelectionList).selected
        seg_sel = self.query_one("#segment-size-select", SelectionList).selected

        test.dimensions = TestDimension(
            dtype=dt_sel if isinstance(dt_sel, CDtype) else CDtype.UNKNOWN,
            sizes_bytes=buf_sel or [],
            segsizes_bytes=seg_sel or []
        )
        test.dimensions.fill_elements()


    def __get_time_limit(self) -> Optional[str]:
        if not self.session.environment.slurm:
            return None
        if not self.session.environment.partition:
            return None
        return self.session.environment.partition.qos.time_limit

    def __get_nodes_limit(self):
        if not self.session.environment.slurm:
            return 1,1

        part = self.session.environment.partition
        if not part or not part.qos:
            return 1, 1
        nodes_limit = part.qos.nodes_limit
        min_nodes = nodes_limit.get('min', 2)
        max_nodes = nodes_limit.get('max', 2)
        return min_nodes, max_nodes



    #BUG: When selecting lumi or leonardo, after qos is selected, if you then select local, the widgets are not disabled.
    def __enable_qos_dep_widgets(self) -> None:
        qos_dep_wid = [
            self.query_one("#nodes-input", Input),
            self.query_one("#time-input", Input),
            self.query_one("#exclude-switch", Switch),
            self.query_one("#dep-switch", Switch),
        ]

        env = self.session.environment
        part = getattr(env, "partition", None)
        qos = getattr(part, "qos", None)

        should_enable = bool(part and qos and env.validate())
        #WARN: Hardcoded bug fix, to debug after
        if env.name == "local":
            should_enable = False
        # self.notify("Should enable is " + str(should_enable))

        for wid in qos_dep_wid:
            if isinstance(wid, Input):
                self.reset_input(wid, disable=not should_enable)
            elif isinstance(wid, Switch):
                wid.disabled = not should_enable
                wid.value = False

    def __enable_compile_dep_widgets(self) -> None:
        compile_dep_wid = [
            self.query_one("#dry-switch", Switch),
            self.query_one("#nodes-input", Input),
            self.query_one("#time-input", Input),
            self.query_one("#exclude-switch", Switch),
            self.query_one("#dep-switch", Switch),
            self.query_one("#data-type-select", Select),
            self.query_one("#buffer-size-select", SelectionList),
            self.query_one("#segment-size-select", SelectionList),
            self.query_one("#compress-switch", Switch),
            self.query_one("#delete-switch", Switch),
            self.query_one("#output-select", Select)
        ]
        
        compile_only = self.query_one("#compile-switch", Switch).value
        for wid in compile_dep_wid:
            if isinstance(wid, Input):
                self.reset_input(wid, disable=compile_only)
            elif isinstance(wid, Switch):
                wid.disabled = compile_only
                wid.value = False
            else:
                wid.disabled = compile_only


    def __update_next(self) -> None:
        next_b = self.query_one("#next", Button)
        env_validate = self.session.environment.validate()
        test_validate = self.session.test.validate(self.session)
        if env_validate and test_validate:
            next_b.disabled = False
        else:
            next_b.disabled = True


