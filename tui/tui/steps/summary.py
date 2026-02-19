# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations
import json
import asyncio
import os
import stat
import re
from pathlib import Path
from textual.widgets import Button, Static, Header, Footer, RichLog, Label, Input
from textual.containers import Horizontal, Vertical
from textual.app import ComposeResult
from tui.steps.base import StepScreen
from textual.screen import Screen
from config_loader import PICO_DIR
from typing import Any, Dict, Iterable, List, Tuple, Union

JsonLike = Union[Dict[str, Any], str, Path]


# WARN: This function needs to be rewritten completely, this is just a quick working prototype
def json_to_exports(config: JsonLike, sh_path: Union[str, Path]) -> str:
    """
    Parse the given JSON config (dict or file path) and write a shell script that exports variables
    per the agreed specification. Returns the output .sh path as a string.
    """
    # --- Load JSON ---
    data: Dict[str, Any]
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = config

    out_path = Path(sh_path)
    lines: List[str] = []

    # helpers -------------
    def is_number_like(v: Any) -> bool:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
        if isinstance(v, str):
            return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", v))
        return False

    def bash_escape_double_quoted(s: str) -> str:
        return s.replace("\\", "\\\\").replace("$", "\\$").replace('"', '\\"')

    def dq(s: str) -> str:
        return f"\"{bash_escape_double_quoted(str(s))}\""

    def yesno(b: Any) -> str:
        return "yes" if bool(b) else "no"

    def csv(values: Iterable[Any]) -> str:
        return ",".join(str(v) for v in values)

    def write_export(name: str, value: Any, *, quote: bool) -> None:
        if quote:
            lines.append(f'export {name}={dq(value)}')
        else:
            lines.append(f'export {name}={value}')

    def mpi_lib_tag(lib_type: str) -> str:
        lt = lib_type.strip().lower()
        if "open" in lt and "mpi" in lt:
            return "OMPI"
        if "cray" in lt:
            return "CRAY_MPICH"
        if "mpich" in lt and "cray" not in lt:
            return "MPICH"
        # Fallback: uppercase and normalize separators
        return re.sub(r"[^A-Za-z0-9]+", "_", lib_type).upper().strip("_")

    def parse_module_name_version(mod: str) -> Tuple[str, str]:
        parts = str(mod).split("/", 1)
        name = parts[0]
        version = parts[1] if len(parts) > 1 else ""
        return name, version

    # --- Begin script ---
    lines.append("#!/bin/bash")

    # ================= environment =================
    env = data.get("environment")
    if not isinstance(env, dict):
        lines.append("# skipped: environment section not found")
    else:
        # LOCATION
        if "name" in env:
            write_export("LOCATION", env["name"], quote=True)
        else:
            lines.append("# skipped: environment.name missing")

        # RUN
        write_export("RUN", "srun" if env.get("slurm") else "mpirun", quote=False)

        # other_var (export all key-value pairs)
        other_var = env.get("other_var")
        if isinstance(other_var, dict):
            for k, v in other_var.items():
                if is_number_like(v):
                    write_export(str(k), v, quote=False)
                else:
                    write_export(str(k), v, quote=True)

        # PARTITION and optional QOS (+ extras)
        part = env.get("partition")
        if isinstance(part, dict):
            if "name" in part:
                write_export("PARTITION", part["name"], quote=True)
            else:
                lines.append("# skipped: environment.partition.name missing")
            qos = part.get("qos")
            if isinstance(qos, dict):
                if qos.get("is_required", False):
                    qname = qos.get("name")
                    if qname:
                        write_export("QOS", qname, quote=True)
                    else:
                        lines.append("# skipped: QOS required but name missing")
                    extra_reqs = qos.get("extra_requirements")
                    if isinstance(extra_reqs, dict):
                        for k, v in extra_reqs.items():
                            if is_number_like(v):
                                write_export(f"QOS_{k.upper()}", v, quote=False)
                            else:
                                write_export(f"QOS_{k.upper()}", v, quote=True)
        else:
            lines.append("# skipped: environment.partition missing")

        # ------- GENERAL_MODULES (env-level only, not from libraries) -------
        general_modules: List[str] = []
        py_mod = env.get("python_module")
        if py_mod:
            general_modules.append(str(py_mod))
        # (If you later add more env-level modules under environment, add them here)
        if general_modules:
            write_export("GENERAL_MODULES", csv(general_modules), quote=True)

    # ================= test =================
    test = data.get("test")
    if not isinstance(test, dict):
        lines.append("# skipped: test section not found")
    else:
        # booleans
        for src_key, out_name in [
            ("compile_only", "COMPILE_ONLY"),
            ("debug_mode", "DEBUG_MODE"),
            ("dry_run", "DRY_RUN"),
            ("delete", "DELETE"),
            ("compress", "COMPRESS"),
        ]:
            if src_key in test:
                write_export(out_name, yesno(test[src_key]), quote=True)
            else:
                lines.append(f"# skipped: test.{src_key} missing")

        # numbers / strings
        if "number_of_nodes" in test:
            write_export("N_NODES", test["number_of_nodes"], quote=False)
        else:
            lines.append("# skipped: test.number_of_nodes missing")

        if "output_level" in test:
            write_export("OUTPUT_LEVEL", test["output_level"], quote=True)
        else:
            lines.append("# skipped: test.output_level missing")

        if "test_time" in test:
            write_export("TEST_TIME", test["test_time"], quote=True)
        else:
            lines.append("# skipped: test.test_time missing")

        # dimensions
        dims = test.get("dimensions")
        if not isinstance(dims, dict):
            lines.append("# skipped: test.dimensions missing")
        else:
            if "dtype" in dims:
                write_export("TYPES", dims["dtype"], quote=True)
            else:
                lines.append("# skipped: test.dimensions.dtype missing")

            if "sizes_elements" in dims and isinstance(dims["sizes_elements"], list):
                write_export("SIZES", csv(dims["sizes_elements"]), quote=True)
            else:
                lines.append("# skipped: test.dimensions.sizes_elements missing or not a list")

            if "segsizes_bytes" in dims and isinstance(dims["segsizes_bytes"], list):
                write_export("SEGMENT_SIZES", csv(dims["segsizes_bytes"]), quote=True)
            else:
                lines.append("# skipped: test.dimensions.segsizes_bytes missing or not a list")

    # ================= libraries (0-based, per-library) =================
    libs = data.get("libraries")
    if not isinstance(libs, list) or len(libs) == 0:
        write_export("LIB_COUNT", 0, quote=False)
        lines.append("# skipped: libraries not found or empty")
    else:
        write_export("LIB_COUNT", len(libs), quote=False)

        for i, lib in enumerate(libs):
            prefix = f"LIB_{i}_"

            if not isinstance(lib, dict):
                lines.append(f"# skipped: libraries[{i}] not a dict")
                continue

            # Identity & basics
            name = lib.get("name")
            if name: write_export(prefix + "NAME", name, quote=True)

            version = lib.get("version")
            if version: write_export(prefix + "VERSION", version, quote=True)

            standard = lib.get("standard")
            if standard: write_export(prefix + "STANDARD", standard, quote=True)

            lib_type = lib.get("lib_type")
            if lib_type:
                write_export(prefix + "MPI_LIB", mpi_lib_tag(str(lib_type)), quote=True)

            # Compiler / MPI lib version
            comp = lib.get("compiler")
            if comp: write_export(prefix + "PICOCC", comp, quote=True)

            if version:
                write_export(prefix + "MPI_LIB_VERSION", version, quote=True)

            # Per-library tests: tasks / gpu
            tests_lib = lib.get("tests", {})
            cpu_list = tests_lib.get("cpu") if isinstance(tests_lib, dict) else None
            if isinstance(cpu_list, list) and len(cpu_list) > 0:
                write_export(prefix + "TASKS_PER_NODE", csv(cpu_list), quote=True)
            # else: omit if missing/empty

            gpu_list = tests_lib.get("gpu") if isinstance(tests_lib, dict) else None
            gpu_awareness_yes = False
            gpu_support = lib.get("gpu_support", {})
            if not isinstance(gpu_support, dict):
                gpu_support = {}
            if isinstance(gpu_list, list) and len(gpu_list) > 0:
                write_export(prefix + "GPU_PER_NODE", csv(gpu_list), quote=True)
                # Awareness only if any non-zero
                gpu_awareness_yes = any(is_number_like(v) and int(v) != 0 for v in gpu_list)
                if gpu_awareness_yes:
                    write_export(prefix + "GPU_AWARENESS", "yes", quote=True)
                    native_gpu_support = bool(gpu_support.get("gpu_support_native", False))
                    write_export(prefix + "GPU_NATIVE_SUPPORT", "yes" if native_gpu_support else "no", quote=True)
                # else: omit GPU_AWARENESS when empty or all zeroes
            # else: omit GPU_PER_NODE

            # Per-library load mechanism: module | set_env | default
            lib_modules: List[str] = []
            lib_load = lib.get("lib_load", {})
            load_type = None
            if isinstance(lib_load, dict):
                lt = str(lib_load.get("type", "default")).strip().lower()
                if lt in ("module", "set_env", "default"):
                    load_type = lt
                else:
                    load_type = "default"
            else:
                load_type = "default"

            write_export(prefix + "LOAD_TYPE", load_type, quote=True)

            if load_type == "module":
                mod = lib_load.get("module")
                if mod:
                    lib_modules.append(str(mod))
            elif load_type == "set_env":
                env_var = lib_load.get("env_var", {})
                if isinstance(env_var, dict) and env_var:
                    keys_out: List[str] = []
                    for k, raw in env_var.items():
                        # raw may be like "/opt/xxx/bin:$PATH" or just "/opt/xxx/bin"
                        s = str(raw)
                        # split by ":" and collect prefixes until we hit a ref to $KEY
                        parts = s.split(":")
                        prefixes: List[str] = []
                        key_uc = str(k).upper()
                        stopper_tokens = {f"${key_uc}", f"${{{key_uc}}}"}
                        for p in parts:
                            p_stripped = p.strip()
                            if p_stripped in stopper_tokens or p_stripped.endswith(key_uc) and "$" in p_stripped:
                                break
                            prefixes.append(p_stripped)
                        # if we didn’t see a stopper, and the value had no "$KEY", take the full string
                        if not prefixes and "$" not in s:
                            prefixes = [s.strip()]
                        # write per-key export if we have anything to prepend
                        if prefixes:
                            keys_out.append(str(k))
                            write_export(prefix + f"ENV_PREPEND_{str(k).upper()}", ":".join(prefixes), quote=True)
                    if keys_out:
                        write_export(prefix + "ENV_PREPEND_VARS", ",".join(keys_out), quote=True)
            # load_type == "default": nothing to add here

            if gpu_awareness_yes:
                gpu_meta = gpu_support.get("metadata", {})
                if not isinstance(gpu_meta, dict):
                    gpu_meta = {}

                meta_gpu_lib = gpu_meta.get("GPU_LIB")
                meta_gpu_lib_version = gpu_meta.get("GPU_LIB_VERSION")
                if meta_gpu_lib:
                    write_export(prefix + "GPU_LIB", str(meta_gpu_lib), quote=True)
                if meta_gpu_lib_version:
                    write_export(prefix + "GPU_LIB_VERSION", str(meta_gpu_lib_version), quote=True)

                gpu_load = gpu_support.get("gpu_load", {})

                if isinstance(gpu_load, dict) and gpu_load.get("type") == "module" and gpu_load.get("module"):
                    mod = str(gpu_load["module"])
                    name_m, ver_m = parse_module_name_version(mod)
                    if not meta_gpu_lib:
                        write_export(prefix + "GPU_LIB", name_m, quote=True)
                    if ver_m and not meta_gpu_lib_version:
                        write_export(prefix + "GPU_LIB_VERSION", ver_m, quote=True)
                    lib_modules.append(mod)
                elif not meta_gpu_lib:
                    lines.append(f"# skipped: libraries[{i}].gpu_support.metadata.GPU_LIB missing for GPU-aware lib")

            if lib_modules:
                write_export(prefix + "MODULES", csv(lib_modules), quote=True)

            # Algorithms / Collectives (per library)
            algos = lib.get("algorithms")
            if not isinstance(algos, dict):
                lines.append(f"# skipped: libraries[{i}].algorithms missing")
            else:
                coll_keys = list(algos.keys())
                if coll_keys:
                    write_export(prefix + "COLLECTIVES", csv(coll_keys), quote=True)
                else:
                    lines.append(f"# skipped: libraries[{i}].algorithms empty")

                # MPICH-family detection for CVARS
                is_mpich_family = bool(lib_type and "mpich" in str(lib_type).lower())

                for coll_key in coll_keys:
                    entries = algos.get(coll_key)
                    if not isinstance(entries, list) or not entries:
                        lines.append(f"# skipped: libraries[{i}].algorithms.{coll_key} empty")
                        continue

                    # names
                    names = [str(e.get("name", "")) for e in entries]
                    write_export(prefix + f"{coll_key.upper()}_ALGORITHMS", csv(names), quote=True)

                    # skip (names with a 'count' constraint)
                    skip_names: List[str] = []
                    for e in entries:
                        cons = e.get("constraints", [])
                        has_count = False
                        if isinstance(cons, list):
                            for c in cons:
                                if isinstance(c, dict) and c.get("key") == "count":
                                    has_count = True
                                    break
                        if has_count:
                            nm = str(e.get("name", ""))
                            if nm:
                                skip_names.append(nm)
                    write_export(prefix + f"{coll_key.upper()}_ALGORITHMS_SKIP", csv(skip_names), quote=True)

                    # is_segmented flags from tags
                    seg_flags: List[str] = []
                    for e in entries:
                        tags = e.get("tags", [])
                        seg_flags.append("yes" if isinstance(tags, list) and "is_segmented" in tags else "no")
                    write_export(prefix + f"{coll_key.upper()}_ALGORITHMS_IS_SEGMENTED", csv(seg_flags), quote=True)

                    # CVARS (MPICH-family only)
                    if is_mpich_family:
                        cvars: List[str] = []
                        for e in entries:
                            sel = e.get("selection", "auto")
                            cvars.append("auto" if sel == "pico" else str(sel))
                        write_export(prefix + f"{coll_key.upper()}_ALGORITHMS_CVARS", csv(cvars), quote=True)

    # --- Write file and chmod +x ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    mode = os.stat(out_path).st_mode
    os.chmod(out_path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(out_path)


SAVE_MSG =  "███████╗ █████╗ ██╗   ██╗███████╗  ██████╗ \n"\
            "██╔════╝██╔══██╗██║   ██║██╔════╝  ╚════██╗\n"\
            "███████╗███████║██║   ██║█████╗      ▄███╔╝\n"\
            "╚════██║██╔══██║╚██╗ ██╔╝██╔══╝      ▀▀══╝ \n"\
            "███████║██║  ██║ ╚████╔╝ ███████╗    ██╗   \n"\
            "╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝    ╚═╝   \n"

TEST_DIR = PICO_DIR / "tests"

class SaveScreen(Screen):
    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "request_quit", "Quit")
    ]

    __data: dict

    def __init__(self, json: dict) -> None:
        super().__init__()
        self.__data = json

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label(SAVE_MSG, id="question", classes="save-label"),
            Static("Files will be saved in `./tests` directory.", classes="field-label"),
            Input(placeholder="Enter filename to save as...", id="filename-input"),
            Label("", id="path-error", classes="error"),
            Horizontal(
                Button("Save", id="save", disabled=True),
                Button("Cancel", id="cancel"),
                classes="quit-button-row"
            ),
            id="save-dialog",
        )
        yield Footer()


    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            raw = self.query_one("#filename-input", Input).value.strip()
            base_name = Path(raw).name
            stem = Path(base_name).stem
            json_name = stem + ".json" if not base_name.lower().endswith(".json") else base_name
            sh_name   = stem + ".sh"
            TEST_DIR.mkdir(exist_ok=True)

            suffix_id = 0
            while True:
                name = f"{Path(json_name).stem}_{suffix_id}.json" if suffix_id else json_name
                shn  = f"{Path(sh_name).stem}_{suffix_id}.sh"     if suffix_id else sh_name

                target = (TEST_DIR / name).resolve()
                sh_target = (TEST_DIR / shn).resolve()

                if TEST_DIR.resolve() in target.parents and TEST_DIR.resolve() in sh_target.parents:
                    if not target.exists() and not sh_target.exists():
                        break
                suffix_id += 1

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    lambda: target.write_text(
                        json.dumps(self.__data, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                )
                await asyncio.to_thread(json_to_exports, self.__data, sh_target)

            except Exception as e:
                self.query_one("#path-error", Label).update(f"Error saving file: {e}")
                return  # stay on screen so user can fix filename / retry

            await asyncio.sleep(2)
            self.app.exit()

        elif event.button.id == "cancel":
            self.app.pop_screen()


    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filename-input":
            return

        filename = event.value.strip()
        save_btn = self.query_one("#save", Button)
        error = self.query_one("#path-error", Label)

        if not filename:
            save_btn.disabled = True
            error.update("Filename cannot be empty.")
            return

        if filename.count('.') > 1:
            save_btn.disabled = True
            error.update("Filename cannot contain multiple dots.")
            return

        if '.' in filename and not filename.lower().endswith(".json"):
            save_btn.disabled = True
            error.update("Only .json extension is allowed.")
            return

        save_btn.disabled = False
        error.update("")


    def action_request_quit(self) -> None:
        self.app.pop_screen()


class SummaryStep(StepScreen):
    __json: dict
    __summary: str

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        self.__json = self.session.to_dict()
        self.__summary = self.session.get_summary()
        json_log = RichLog(markup=False, classes="summary-box", id="json-log", wrap=True, auto_scroll=False)
        summary_log = RichLog(markup=False, classes="summary-box", id="summary-log", wrap=True, auto_scroll=False)
        json_log.write(json.dumps(self.__json, indent=2))
        summary_log.write(self.__summary)

        yield Horizontal(
            Vertical(
                Static("Generated Test JSON", classes="field-label"),
                json_log,
                classes="summary-container"
            ),
            Vertical(
                Static("Short Summary", classes="field-label"),
                summary_log,
                classes="summary-container"
            ),
            classes="full"
        )

        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Save", id="next"),
            classes="button-row"
        )

        yield Footer()


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev":
            from tui.steps.algorithms import AlgorithmsStep
            self.prev(AlgorithmsStep)
        elif event.button.id == "next":
            self.app.push_screen(SaveScreen(self.__json))


    def get_help_desc(self) -> Tuple[str, str]:
        focused = self.focused
        default = (
            "Review & Export",
            "Inspect the generated JSON and short summary, then save the bundle into tests/."
        )

        if not focused or not getattr(focused, "id", None):
            return default

        fid = focused.id

        if fid == "json-log":
            return (
                "Generated Test JSON",
                "Full configuration as saved to <name>.json. Use arrow keys or PgUp/PgDn to scroll."
            )
        if fid == "summary-log":
            return (
                "Short Summary",
                "Condensed view of environment, nodes, dimensions, and libraries."
            )
        if fid == "prev":
            return (
                "Previous Step",
                "Return to algorithm selection to make changes (shortcut: `p`)."
            )
        if fid == "next":
            return (
                "Save & Export",
                "Open the save dialog to choose a filename. An executable .sh export is produced alongside the JSON."
            )

        return default
