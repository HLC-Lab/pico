# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Select, Switch, Footer, Header, Button, Input, SelectionList, Label
from textual.widgets.selection_list import Selection
from .base import StepScreen
from config_loader import conf_list_environments, conf_get_general, conf_get_slurm_opts
from models import EnvironmentSelection, PartitionSelection, TestDimension, CDtype, OutputLevel
from typing import Tuple, Optional

class ConfigureStep(StepScreen):
    __buffer_sizes =  ["32  Byte", "256 Byte", "2   KiB", "16  KiB", "128 KiB", "1   MiB", "8   MiB", "64  MiB", "512 MiB"]
    __segment_sizes = ["0   Byte", "16  KiB", "128 KiB", "1   MiB"]
    def compose(self):
        self.__hydrating = True
        environment = self.session.environment
        test = self.session.test
        dimensions = test.dimensions
        partition = environment.partition
        qos = partition.qos if partition else None

        self.__slurm_opts = {}
        if environment.name and environment.slurm:
            self.__slurm_opts = conf_get_slurm_opts(environment.name)

        environment_select_args = {}
        if environment.name:
            environment_select_args["value"] = environment.name

        partition_options = []
        partition_select_args = {}
        if self.__slurm_opts:
            partition_options = [
                (name, name) for name in self.__slurm_opts.get("PARTITIONS", {})
            ]
        if partition and partition.name:
            partition_select_args["value"] = partition.name

        qos_options = []
        qos_select_args = {}
        if partition and partition.name:
            qos_options = [
                (name, name)
                for name in self.__slurm_opts.get("PARTITIONS", {})
                .get(partition.name, {})
                .get("QOS", {})
            ]
        if qos and qos.name:
            qos_select_args["value"] = qos.name

        has_saved_configuration = bool(environment.name)
        slurm_ready = bool(environment.slurm and environment.validate())
        runtime_enabled = slurm_ready and not test.compile_only

        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("Environment:", classes="field-label"),
                Select(
                    [(e, e) for e in conf_list_environments()],
                    prompt="Environment:",
                    id="env-select",
                    **environment_select_args,
                )
            ),
            Vertical(
                Static("Partition:", classes="field-label"),
                Select(
                    partition_options,
                    prompt="Partition:",
                    id="partition-select",
                    disabled=not bool(environment.slurm and environment.name),
                    **partition_select_args,
                )
            ),
            Vertical(
                Static("QOS:", classes="field-label"),
                Select(
                    qos_options,
                    prompt="QOS:",
                    id="qos-select",
                    disabled=not bool(partition and partition.name),
                    **qos_select_args,
                )
            ),
            classes="row"
        )


        yield Horizontal(
            Vertical(
                Static("Compile Only", classes="field-label"),
                Switch(id="compile-switch", value=test.compile_only, disabled=test.dry_run)
            ),
            Vertical(
                Static("Debug Mode", classes="field-label"),
                Switch(id="debug-switch", value=test.debug_mode)
            ),
            Vertical(
                Static("Dry Run Mode", classes="field-label"),
                Switch(id="dry-switch", value=test.dry_run, disabled=test.compile_only)
            ),
            Horizontal(
                Vertical(
                    Static("Number of Nodes",  classes="field-label"),
                    Input(placeholder=f"Insert number of nodes",
                        value=str(test.number_of_nodes) if slurm_ready else "",
                        disabled=not runtime_enabled, id="nodes-input"),
                    Label("", id="nodes-error", classes="error"),
                    classes="field",
                ),
                Vertical(
                    Static("Test Time", classes="field-label"),
                    Input(placeholder=f"Insert time in HH:MM:SS",
                        value=test.test_time or "",
                        id="time-input", disabled=not runtime_enabled),
                    Label("", id="time-error", classes="error"),
                    classes="field"
                ),
            ),
            classes="tight-switches"
        )

        yield Horizontal(
            Vertical(
                Static("Exclude Nodes", classes="field-label"),
                Switch(
                    id="exclude-switch",
                    value=bool(test.exclude_nodes),
                    disabled=not runtime_enabled,
                ),
                classes="switch-col",
            ),
            Vertical(
                Static(" ", classes="field-label"),
                Input(placeholder="What nodes do you want to exclude?",
                    value=test.exclude_nodes or "",
                    id="excluded-nodes",
                    disabled=not (runtime_enabled and test.exclude_nodes)),
                Label("", id="excluded-nodes-error", classes="error"),
                classes="field",
            ),
            Vertical(
                Static("Start After", classes="field-label"),
                Switch(
                    id="dep-switch",
                    value=bool(test.job_dependency),
                    disabled=not runtime_enabled,
                ),
                classes="switch-col",
            ),
            Vertical(
                Static(" ", classes="field-label"),
                Input(placeholder="Insert here job ID",
                    value=str(test.job_dependency) if test.job_dependency else "",
                    id="dep-input",
                    disabled=not (runtime_enabled and test.job_dependency)),
                Label("", id="dep-error", classes="error"),
                classes="field",
            ),
            Vertical(
                Static("Inject Params", classes="field-label"),
                Switch(id="inject-switch", value=bool(test.inject_params)),
                classes="switch-col"
            ),
            Vertical(
                Static(" ", classes="field-label"),
                Input(placeholder="Insert here any sbatch param or env",
                      value=test.inject_params or "",
                      id="inject-params", disabled=not bool(test.inject_params)),
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
            ("Summarized", OutputLevel.SUMMARIZED),
        ]

        selected_buffers = (
            set(dimensions.sizes_bytes)
            if dimensions and dimensions.sizes_bytes
            else {self.__parse_size(label) for label in self.__buffer_sizes}
        )
        selected_segments = (
            set(dimensions.segsizes_bytes)
            if dimensions and dimensions.segsizes_bytes
            else {self.__parse_size(label) for label in self.__segment_sizes[1:]}
        )
        buffer_items = [
            Selection(
                label.replace("Byte", "  B"),
                self.__parse_size(label),
                self.__parse_size(label) in selected_buffers,
            )
            for label in self.__buffer_sizes
        ]
        segment_items = [
            Selection(
                "No Segment" if index == 0 else label.replace("Byte", "  B"),
                self.__parse_size(label),
                self.__parse_size(label) in selected_segments,
            )
            for index, label in enumerate(self.__segment_sizes)
        ]

        dtype = (
            dimensions.dtype
            if dimensions and dimensions.dtype != CDtype.UNKNOWN
            else CDtype.INT32
        )
        output_level = test.output_level or OutputLevel.MINIMAL
        compress = test.compress if has_saved_configuration else True
        delete = test.delete if has_saved_configuration else True

        yield Horizontal(
            Vertical(
                Vertical(
                    Static("Data Type", classes = "field-label"),
                    Select(
                        dtypes,
                        prompt="Select Data Type",
                        id="data-type-select",
                        value=dtype,
                        disabled=test.compile_only,
                    )
                ),
                Vertical(
                    Static("Output Level", classes="field-label"),
                    Select(
                        output_lev,
                        id="output-select",
                        prompt="Select Output Level",
                        value=output_level,
                        disabled=test.compile_only,
                    )
                ),
                classes="field-small"
            ),
            Vertical(
                Vertical(
                    Static("Compress Res.", classes="field-label"),
                    Switch(
                        id="compress-switch",
                        value=compress,
                        disabled=test.compile_only,
                    ),
                ),
                Vertical(
                    Static("Delete Uncompr.", classes="field-label"),
                    Switch(
                        id="delete-switch",
                        value=delete,
                        disabled=test.compile_only or not compress,
                    ),
                ),
                classes="field-mini"
            ),
            Vertical(
                Static("Buffer Sizes", classes="field-label"),
                SelectionList[int](
                    *buffer_items,
                    id="buffer-size-select",
                    disabled=test.compile_only,
                ),
                classes="field"
            ),
            Vertical(
                Static("Segment Sizes", classes="field-label"),
                SelectionList[int](
                    *segment_items,
                    id="segment-size-select",
                    disabled=test.compile_only,
                ),
                classes="field"
            ),
        )

        yield self.navigation_buttons(prev_disabled=True)

        yield Footer()

    def on_mount(self):
        super().on_mount()
        self.__label_selection_list()
        self.__update_next()
        self.call_after_refresh(self.__finish_hydration)

    def __finish_hydration(self) -> None:
        self.__hydrating = False

    # TODO: GPU switch does not exist anymore
    def on_select_changed(self, event):
        if self.__hydrating:
            return
        sel = event.control
        part_w = self.query_one("#partition-select", Select)
        qos_w = self.query_one("#qos-select", Select)

        if sel.id == "env-select":
            env = event.value
            previous_environment = self.session.environment.name
            self.reset_select(part_w)
            self.reset_select(qos_w)

            self.session.environment = EnvironmentSelection()
            self.__slurm_opts = {}
            if env != previous_environment:
                self.session.libraries = []

            if not sel.is_blank():
                env_json = conf_get_general(env)
                self.session.environment.from_dict(env_json)

                if self.has_slurm:
                    self.__slurm_opts = conf_get_slurm_opts(env)
                    part_w.set_options([(p, p) for p in self.__slurm_opts["PARTITIONS"]])
                    part_w.disabled = False

        elif sel.id == "partition-select":
            self.reset_select(qos_w)
            self.session.environment.init_partition()

            if not sel.is_blank():
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

            if not sel.is_blank():
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
        if self.__hydrating:
            return
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
        if self.__hydrating:
            return
        self.__update_test_selections()
        self.__update_next()

    def on_switch_changed(self, event):
        if self.__hydrating:
            return
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
        default_desc = (
            "Configuration Overview",
            "Pick an environment, resources, and data sizes to describe your benchmark run."
        )

        if not focused or not getattr(focused, "id", None):
            return default_desc

        fid = focused.id

        env = self.session.environment
        part = env.partition if env else None
        qos = part.qos if part else None
        test = self.session.test
        dims = test.dimensions if test else None

        def nodes_limits() -> str:
            if env and env.slurm and part and qos:
                mn = qos.nodes_limit.get("min", 1)
                mx = qos.nodes_limit.get("max", mn)
                return f"Allowed: {mn}–{mx} nodes for {part.name} / {qos.name}."
            return "Local executions require exactly 1 node."

        def time_limit() -> str:
            if env and env.slurm and qos:
                limit = qos.time_limit or "00:00:00"
                return f"Format DD-HH:MM:SS (or HH:MM:SS). Max allowed: {limit}."
            return "Debug/local runs do not accept a custom time limit."

        def dtype_help() -> str:
            if not dims:
                return "Select the datatype used when computing element counts."
            if dims.dtype == CDtype.UNKNOWN:
                return "Pick a datatype to enable automatic conversion between bytes and elements."
            return f"Current dtype: {dims.dtype}"

        def buffers_help(get_segments: bool = False) -> str:
            if not dims:
                return "Toggle one or more segment sizes." if get_segments else "Toggle one or more message sizes; element counts follow the datatype."
            values = dims.get_printable_sizes(get_segment_sizes=get_segments)
            if not values:
                return "Select at least one segment size or leave 'No Segment' to rely on library defaults." if get_segments else "Select at least one message size to generate workloads."
            label = "Segments" if get_segments else "Buffers"
            return f"{label} currently enabled: {', '.join(values)}"

        field_map: dict[str, Tuple[str, str]] = {
            "env-select": (
                "Test Environment",
                env.get_help() if env and env.name else "Select the target platform (loads JSON from config/environment/)."
            ),
            "partition-select": (
                "SLURM Partition",
                part.get_help() if part and part.name else "Choose a partition to unlock QoS and resource limits."
            ),
            "qos-select": (
                "Quality of Service",
                qos.get_help() if qos and qos.name else "Select a QoS profile to set node/time constraints."
            ),
            "compile-switch": (
                "Compile Only",
                "Build binaries only. Forces 1 node and disables runtime options."
            ),
            "debug-switch": (
                "Debug Mode",
                "Fast debug recipe: short timeout, single iteration, -g -DDEBUG, skips compression."
            ),
            "dry-switch": (
                "Dry Run",
                "Generate scripts without launching jobs. Incompatible with compile-only."
            ),
            "nodes-input": (
                "Number of Nodes",
                nodes_limits()
            ),
            "time-input": (
                "Test Time",
                time_limit()
            ),
            "exclude-switch": (
                "Exclude Nodes",
                "Toggle to supply a comma-separated list of node names to avoid."
            ),
            "excluded-nodes": (
                "Nodes to Exclude",
                "Comma-separated hostnames passed to the job launcher."
            ),
            "dep-switch": (
                "Start After",
                "Enable to wait for another SLURM job id before running."
            ),
            "dep-input": (
                "Dependency Job ID",
                "Provide a numeric SLURM job ID that must finish before this test."
            ),
            "inject-switch": (
                "Inject Parameters",
                "Enable to add custom sbatch arguments or env vars."
            ),
            "inject-params": (
                "Custom sbatch/env parameters",
                "Comma or space separated tokens appended to the sbatch command."
            ),
            "data-type-select": (
                "Datatype",
                dtype_help()
            ),
            "output-select": (
                "Output Level",
                "Full saves every rank for every iteration; Statistics saves "
                "cross-rank statistics per iteration; Minimal saves the slowest "
                "rank per iteration; Summarized saves one aggregate row after "
                "discarding the first 20% of samples."
            ),
            "compress-switch": (
                "Compress Results",
                "Enable gzip compression of raw CSV logs."
            ),
            "delete-switch": (
                "Delete Uncompressed",
                "If compression is enabled, remove original files after packaging."
            ),
            "buffer-size-select": (
                "Message Sizes",
                buffers_help(get_segments=False)
            ),
            "segment-size-select": (
                "Segment Sizes",
                buffers_help(get_segments=True)
            ),
            "prev": (
                "Previous Step",
                "Shortcut: press `p` to go back without leaving the TUI."
            ),
            "next": (
                "Next Step",
                "Press `n` to continue once all required fields are valid."
            ),
        }

        if fid in field_map:
            return field_map[fid]

        return default_desc

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
