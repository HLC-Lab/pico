from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Select, Switch, Footer, Header, Button, Input, SelectionList
from textual.widgets.selection_list import Selection
from .base import StepScreen
from config_loader import conf_list_environments, conf_get_general, conf_get_slurm_opts
from models import SessionConfig, EnvironmentSelection, PartitionSelection, TestDimension, CDtype
from typing import Tuple

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
            classes="row"
        )

        yield Horizontal(
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
                Vertical(
                    Static("GPU Buffers", classes="field-label"),
                    Switch(id="gpu-switch", value=False, disabled=True)
                ),
                Vertical(
                    Static("Inject params or ENV?", classes="field-label"),
                    Input(placeholder="Insert here any sbatch param or env", id="inject-params")
                ),
                classes="tight-switches"
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

        buffer_items =  [Selection(f"{label.replace("Byte", "  B")}", self.__parse_size(label), True) for label in self.__buffer_sizes]
        segment_items = [Selection(f"{label.replace("Byte", "  B")}", self.__parse_size(label)) for label in self.__segment_sizes]
        segment_items[0] = Selection("No Segment", 0, True)

        yield Horizontal(
            Vertical(
                Static("Data Type", classes = "field-label"),
                Select( dtypes, prompt="Select Data Type", id="data-type-select", value=CDtype.INT32),
                classes="field-small"
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

    def on_select_changed(self, event):
        sel = event.control

        part_w = self.query_one("#partition-select", Select)
        qos_w = self.query_one("#qos-select", Select)
        gpu_sw = self.query_one("#gpu-switch", Switch)
        next_b = self.query_one("#next", Button)

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

        self.__update_selections()
        self.__label_selection_list()

        next_b.disabled = not (self.session.environment.validate() and self.session.test.validate())
        val = True
        if isinstance(self.session.environment.partition, PartitionSelection):
            val = not self.session.environment.partition.is_gpu

        gpu_sw.disabled = val
        if val:
            gpu_sw.value = False
            self.session.test.use_gpu_buffers = False

    def on_input_changed(self, event):
        input_w = event.control
        if input_w.id == "inject-params":
            self.session.test.inject_params = input_w.value.strip()
        else:
            raise ValueError(f"Unexpected input control: {input_w.id}")


    def on_button_pressed(self, event):
        if event.button.id == 'next':
            if not self.session.test.validate():
                raise ValueError("Test configuration is not valid.")
            if not self.session.environment.validate():
                raise ValueError("Environment configuration is not valid.")
            from tui.steps.tasks import TasksStep
            self.next(TasksStep)

    # NOTE: Cannot make the Selection List change work without the decorator,
    # on_selection_list_changed does not get called.
    @on(SelectionList.SelectedChanged)
    def sel_list_handler(self):
        self.__update_selections()
        next_b = self.query_one("#next", Button)
        next_b.disabled = not (self.session.environment.validate() and self.session.test.validate())

    def on_switch_changed(self, event):
        compile_switch = self.query_one("#compile-switch", Switch)
        dry_switch = self.query_one("#dry-switch", Switch)

        cid = event.control.id
        val = event.value

        if cid == "compile-switch":
            dt_select = self.query_one("#data-type-select", Select)
            buf_list = self.query_one("#buffer-size-select", SelectionList)
            seg_list = self.query_one("#segment-size-select", SelectionList)
            self.session.test.compile_only = val
            if val:
                self.session.test.dry_run = False
                dry_switch.value = False

            dry_switch.disabled = val
            dt_select.disabled = val
            buf_list.disabled = val
            seg_list.disabled = val
        elif cid == "debug-switch":
            self.session.test.debug_mode = val
        elif cid == "dry-switch":
            self.session.test.dry_run = val
            if val:
                self.session.test.compile_only = False
                compile_switch.value = False
            compile_switch.disabled = val
        elif cid == "gpu-switch":
            if not self.session.environment.partition:
                raise ValueError("Partition must be selected before enabling GPU buffers.")
            self.session.test.use_gpu_buffers = val

        self.__update_selections()
        next_b = self.query_one("#next", Button)
        next_b.disabled = not (self.session.environment.validate() and self.session.test.validate())

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
            "gpu-switch": (
                "GPU Buffers toggle",
                "Enables GPU buffers for the test. Requires a GPU partition.\n"\
                "This option is used also for compilation of GPU exec, so it's compatible with Compile Only"
            ),
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
        """Parses human-readable sizes into bytes."""
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

    def __update_selections(self):
        test_opt = self.session.test
        if test_opt.compile_only:
            test_opt.dimensions = None
            return

        dt_sel = self.query_one("#data-type-select", Select).value
        buf_sel = self.query_one("#buffer-size-select", SelectionList).selected
        seg_sel = self.query_one("#segment-size-select", SelectionList).selected

        test_opt.dimensions = TestDimension(
            dtype=dt_sel if isinstance(dt_sel, CDtype) else CDtype.UNKNOWN,
            sizes_bytes=buf_sel if buf_sel else [],
            segsizes_bytes=seg_sel if seg_sel else []
        )

        test_opt.dimensions.fill_elements()



