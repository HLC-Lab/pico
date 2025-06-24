from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Select, Switch, Footer, Header, Button, Input
from .base import StepScreen
from config_loader import conf_list_environments, conf_get_general, conf_get_slurm_opts
from models import SessionConfig, EnvironmentSelection, PartitionSelection
from typing import Tuple

class ConfigureStep(StepScreen):
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

        next_b.disabled = not self.session.environment.validate()
        val = True
        if isinstance(self.session.environment.partition, PartitionSelection):
            val = not self.session.environment.partition.is_gpu

        gpu_sw.disabled = val
        if val:
            gpu_sw.value = False
            self.session.compile.use_gpu_buffers = False

    def on_input_changed(self, event):
        input_w = event.control
        if input_w.id == "inject-params":
            self.session.compile.inject_params = input_w.value.strip()
        else:
            raise ValueError(f"Unexpected input control: {input_w.id}")

    def on_button_pressed(self, event):
        if event.button.id == 'next':
            if not self.session.compile.validate():
                raise ValueError("Compile configuration is not valid.")
            if not self.session.environment.validate():
                raise ValueError("Environment configuration is not valid.")
            from tui.steps.tasks import TasksStep
            self.next(TasksStep)

    def on_switch_changed(self, event):
        compile_switch = self.query_one("#compile-switch", Switch)
        dry_switch = self.query_one("#dry-switch", Switch)

        cid = event.control.id
        val = event.value

        if cid == "compile-switch":
            self.session.compile.compile_only = val
            if val:
                self.session.compile.dry_run = False
                dry_switch.value = False
            dry_switch.disabled = val
        elif cid == "debug-switch":
            self.session.compile.debug_mode = val
        elif cid == "dry-switch":
            self.session.compile.dry_run = val
            if val:
                self.session.compile.compile_only = False
                compile_switch.value = False
            compile_switch.disabled = val
        elif cid == "gpu-switch":
            if not self.session.environment.partition:
                raise ValueError("Partition must be selected before enabling GPU buffers.")
            self.session.compile.use_gpu_buffers = val

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
        }

        if focused and focused.id in field_map:
            field_desc, chosen_desc = field_map[focused.id]

        return (field_desc, chosen_desc)


