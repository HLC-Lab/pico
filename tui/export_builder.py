# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

JsonLike = Union[Mapping[str, Any], str, Path]

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LAUNCHER_RE = re.compile(r"^[A-Za-z0-9_./+-]+$")
_NODE_LIST_RE = re.compile(r"^[A-Za-z0-9_.\-,\[\]]+$")
_JOB_DEP_RE = re.compile(r"^[0-9]+(?::[0-9]+)*$")
_UNSAFE_FRAGMENT_RE = re.compile(r"[\x00\r\n;&|`]|\$\(")


class ExportValidationError(ValueError):
    """Raised when a TUI descriptor cannot produce a complete shell export."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)
        super().__init__(
            "Invalid TUI descriptor:\n"
            + "\n".join(f"- {error}" for error in self.errors)
        )


def _load_config(config: JsonLike) -> Dict[str, Any]:
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
    else:
        loaded = dict(config)
    if not isinstance(loaded, dict):
        raise ExportValidationError(["descriptor root must be an object"])
    return loaded


def _shell_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _csv(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def _mpi_lib_tag(lib_type: str) -> str:
    normalized = lib_type.strip().lower()
    if "open" in normalized and "mpi" in normalized:
        return "OMPI"
    if "cray" in normalized:
        return "CRAY_MPICH"
    if "mpich" in normalized:
        return "MPICH"
    return re.sub(r"[^A-Za-z0-9]+", "_", lib_type).upper().strip("_")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_int_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_positive_int(item) for item in value)
    )


def _validate_fragment(value: Any, field: str, errors: List[str]) -> None:
    if value is None or value == "":
        return
    if not isinstance(value, str):
        errors.append(f"{field} must be a string")
    elif _UNSAFE_FRAGMENT_RE.search(value):
        errors.append(f"{field} contains unsafe shell control characters")


def validate_export_config(data: Mapping[str, Any]) -> None:
    """Validate fields required by scripts/submit_wrapper.sh's TUI path."""

    errors: List[str] = []
    environment = data.get("environment")
    test = data.get("test")
    libraries = data.get("libraries")

    if not isinstance(environment, dict):
        errors.append("environment must be an object")
        environment = {}
    if not _nonempty_string(environment.get("name")):
        errors.append("environment.name is required")
    if not isinstance(environment.get("slurm"), bool):
        errors.append("environment.slurm must be a boolean")

    launcher = environment.get("launcher")
    if launcher is not None and (
        not _nonempty_string(launcher) or not _LAUNCHER_RE.fullmatch(launcher)
    ):
        errors.append("environment.launcher must be a command name or path")
    _validate_fragment(
        environment.get("launcher_flags"),
        "environment.launcher_flags",
        errors,
    )

    other_var = environment.get("other_var", {})
    if other_var is not None and not isinstance(other_var, dict):
        errors.append("environment.other_var must be an object")
    elif isinstance(other_var, dict):
        for name in other_var:
            if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
                errors.append(
                    f"environment.other_var contains invalid variable name {name!r}"
                )

    slurm = environment.get("slurm") is True
    partition = environment.get("partition")
    if slurm:
        if not isinstance(partition, dict):
            errors.append("environment.partition is required for Slurm")
        elif not _nonempty_string(partition.get("name")):
            errors.append("environment.partition.name is required for Slurm")
    if isinstance(partition, dict):
        qos = partition.get("qos")
        if qos is not None and not isinstance(qos, dict):
            errors.append("environment.partition.qos must be an object")
        elif isinstance(qos, dict):
            if qos.get("is_required") and not _nonempty_string(qos.get("name")):
                errors.append(
                    "environment.partition.qos.name is required by this QoS"
                )
            extra_requirements = qos.get("extra_requirements", {})
            if (
                extra_requirements is not None
                and not isinstance(extra_requirements, dict)
            ):
                errors.append(
                    "environment.partition.qos.extra_requirements must be an object"
                )
            elif isinstance(extra_requirements, dict):
                for name in extra_requirements:
                    normalized_name = str(name).upper()
                    if not _ENV_NAME_RE.fullmatch(normalized_name):
                        errors.append(
                            "environment.partition.qos.extra_requirements "
                            f"contains invalid name {name!r}"
                        )

    if not isinstance(test, dict):
        errors.append("test must be an object")
        test = {}

    boolean_fields = (
        "compile_only",
        "debug_mode",
        "dry_run",
        "delete",
        "compress",
    )
    for field in boolean_fields:
        if not isinstance(test.get(field), bool):
            errors.append(f"test.{field} must be a boolean")

    compile_only = test.get("compile_only") is True
    if not _positive_int(test.get("number_of_nodes")):
        errors.append("test.number_of_nodes must be a positive integer")
    if test.get("delete") is True and test.get("compress") is not True:
        errors.append("test.delete requires test.compress")

    dependency = test.get("job_dependency")
    if dependency is not None and dependency != "":
        dependency_string = str(dependency)
        if not _JOB_DEP_RE.fullmatch(dependency_string):
            errors.append(
                "test.job_dependency must contain numeric job IDs separated by ':'"
            )

    excluded_nodes = test.get("exclude_nodes")
    if excluded_nodes is not None and excluded_nodes != "":
        if not isinstance(excluded_nodes, str) or not _NODE_LIST_RE.fullmatch(
            excluded_nodes
        ):
            errors.append("test.exclude_nodes contains an invalid node expression")

    _validate_fragment(test.get("inject_params"), "test.inject_params", errors)

    if not compile_only:
        if test.get("output_level") not in (
            "full",
            "statistics",
            "minimal",
            "summarized",
        ):
            errors.append(
                "test.output_level must be full, statistics, minimal or summarized"
            )
        if slurm and not _nonempty_string(test.get("test_time")):
            errors.append("test.test_time is required for Slurm")

        dimensions = test.get("dimensions")
        if not isinstance(dimensions, dict):
            errors.append("test.dimensions is required")
        else:
            if dimensions.get("dtype") not in (
                "char",
                "float",
                "double",
                "int8",
                "int16",
                "int32",
                "int64",
            ):
                errors.append("test.dimensions.dtype is invalid")
            if not _positive_int_list(dimensions.get("sizes_elements")):
                errors.append(
                    "test.dimensions.sizes_elements must be a non-empty "
                    "positive-integer list"
                )
            segment_sizes = dimensions.get("segsizes_bytes")
            if (
                not isinstance(segment_sizes, list)
                or not segment_sizes
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in segment_sizes
                )
            ):
                errors.append(
                    "test.dimensions.segsizes_bytes must be a non-empty "
                    "non-negative-integer list"
                )

    if not isinstance(libraries, list) or not libraries:
        errors.append("libraries must be a non-empty list")
        libraries = []

    for index, library in enumerate(libraries):
        label = f"libraries[{index}]"
        if not isinstance(library, dict):
            errors.append(f"{label} must be an object")
            continue

        for field in ("name", "version", "standard", "lib_type", "compiler"):
            if not _nonempty_string(library.get(field)):
                errors.append(f"{label}.{field} is required")

        load = library.get("lib_load")
        if not isinstance(load, dict):
            errors.append(f"{label}.lib_load must be an object")
        else:
            load_type = load.get("type")
            if load_type not in ("default", "module", "set_env"):
                errors.append(
                    f"{label}.lib_load.type must be default, module or set_env"
                )
            elif load_type == "module" and not _nonempty_string(
                load.get("module")
            ):
                errors.append(f"{label}.lib_load.module is required")
            elif load_type == "set_env":
                env_vars = load.get("env_var")
                if not isinstance(env_vars, dict) or not env_vars:
                    errors.append(f"{label}.lib_load.env_var is required")
                else:
                    for name, value in env_vars.items():
                        if (
                            not isinstance(name, str)
                            or not _ENV_NAME_RE.fullmatch(name)
                        ):
                            errors.append(
                                f"{label}.lib_load.env_var contains invalid "
                                f"variable name {name!r}"
                            )
                        if not isinstance(value, str):
                            errors.append(
                                f"{label}.lib_load.env_var.{name} must be a string"
                            )

        if compile_only:
            continue

        library_tests = library.get("tests")
        if not isinstance(library_tests, dict):
            errors.append(f"{label}.tests must be an object")
            library_tests = {}

        cpu_tasks = library_tests.get("cpu", [])
        gpu_tasks = library_tests.get("gpu", [])
        if cpu_tasks and not _positive_int_list(cpu_tasks):
            errors.append(f"{label}.tests.cpu must contain positive integers")
        if gpu_tasks and not _positive_int_list(gpu_tasks):
            errors.append(f"{label}.tests.gpu must contain positive integers")
        if not cpu_tasks and not gpu_tasks:
            errors.append(f"{label}.tests must select CPU or GPU tasks")

        if gpu_tasks:
            metadata = library.get("metadata")
            if not isinstance(metadata, dict):
                errors.append(f"{label}.metadata is required for GPU tests")
            else:
                for field in ("GPU_LIB", "GPU_LIB_VERSION"):
                    if not _nonempty_string(metadata.get(field)):
                        errors.append(
                            f"{label}.metadata.{field} is required for GPU tests"
                        )

        algorithms = library.get("algorithms")
        if not isinstance(algorithms, dict) or not algorithms:
            errors.append(f"{label}.algorithms must be a non-empty object")
            continue

        selected_count = 0
        for collective, entries in algorithms.items():
            if (
                not isinstance(collective, str)
                or not _ENV_NAME_RE.fullmatch(collective.upper())
            ):
                errors.append(f"{label}.algorithms contains an invalid collective")
                continue
            if not isinstance(entries, list):
                errors.append(
                    f"{label}.algorithms.{collective} must be a list"
                )
                continue
            selected_count += len(entries)
            for algorithm_index, algorithm in enumerate(entries):
                algorithm_label = (
                    f"{label}.algorithms.{collective}[{algorithm_index}]"
                )
                if not isinstance(algorithm, dict):
                    errors.append(f"{algorithm_label} must be an object")
                    continue
                if not _nonempty_string(algorithm.get("name")):
                    errors.append(f"{algorithm_label}.name is required")
                if algorithm.get("selection") in (None, ""):
                    errors.append(f"{algorithm_label}.selection is required")
        if selected_count == 0:
            errors.append(f"{label}.algorithms must select at least one algorithm")

    if errors:
        raise ExportValidationError(errors)


def render_shell_exports(config: JsonLike) -> str:
    """Render a validated TUI descriptor as a sourceable Bash script."""

    data = _load_config(config)
    validate_export_config(data)
    lines: List[str] = ["#!/bin/bash"]

    def export(name: str, value: Any) -> None:
        lines.append(f"export {name}={_shell_quote(value)}")

    environment = data["environment"]
    slurm = environment["slurm"]
    export("LOCATION", environment["name"])
    export("RUN", environment.get("launcher") or ("srun" if slurm else "mpirun"))
    export("RUNFLAGS", environment.get("launcher_flags") or "")

    for name, value in (environment.get("other_var") or {}).items():
        export(name, value)

    partition = environment.get("partition")
    if isinstance(partition, dict):
        export("PARTITION", partition["name"])
        qos = partition.get("qos")
        if isinstance(qos, dict):
            if qos.get("is_required"):
                export("QOS", qos["name"])
            for name, value in (qos.get("extra_requirements") or {}).items():
                export(f"QOS_{str(name).upper()}", value)

    export("GENERAL_MODULES", environment.get("python_module") or "")

    test = data["test"]
    for source, target in (
        ("compile_only", "COMPILE_ONLY"),
        ("debug_mode", "DEBUG_MODE"),
        ("dry_run", "DRY_RUN"),
        ("delete", "DELETE"),
        ("compress", "COMPRESS"),
    ):
        export(target, "yes" if test[source] else "no")

    export("N_NODES", test["number_of_nodes"])
    optional_test_exports = (
        ("output_level", "OUTPUT_LEVEL"),
        ("test_time", "TEST_TIME"),
    )
    for source, target in optional_test_exports:
        if test.get(source) not in (None, ""):
            export(target, test[source])
    for source, target in (
        ("inject_params", "INJECT_PARAMS"),
        ("exclude_nodes", "EXCLUDE_NODES"),
        ("job_dependency", "JOB_DEP"),
    ):
        export(target, test.get(source) or "")

    dimensions = test.get("dimensions")
    if isinstance(dimensions, dict):
        export("TYPES", dimensions["dtype"])
        export("SIZES", _csv(dimensions["sizes_elements"]))
        export("SEGMENT_SIZES", _csv(dimensions["segsizes_bytes"]))

    libraries = data["libraries"]
    export("LIB_COUNT", len(libraries))
    for index, library in enumerate(libraries):
        prefix = f"LIB_{index}_"
        export(prefix + "NAME", library["name"])
        export(prefix + "VERSION", library["version"])
        export(prefix + "STANDARD", library["standard"])
        export(prefix + "MPI_LIB", _mpi_lib_tag(library["lib_type"]))
        export(prefix + "PICOCC", library["compiler"])
        export(prefix + "MPI_LIB_VERSION", library["version"])

        library_tests = library.get("tests") or {}
        cpu_tasks = library_tests.get("cpu") or []
        gpu_tasks = library_tests.get("gpu") or []
        if cpu_tasks:
            export(prefix + "TASKS_PER_NODE", _csv(cpu_tasks))
        if gpu_tasks:
            export(prefix + "GPU_PER_NODE", _csv(gpu_tasks))
            export(prefix + "GPU_AWARENESS", "yes")
            gpu_support = library.get("gpu_support") or {}
            export(
                prefix + "GPU_NATIVE_SUPPORT",
                "yes" if gpu_support.get("gpu_support_native") else "no",
            )

        modules: List[str] = []
        library_load = library["lib_load"]
        load_type = library_load["type"]
        export(prefix + "LOAD_TYPE", load_type)
        if load_type == "module":
            modules.append(library_load["module"])
        elif load_type == "set_env":
            env_vars = library_load["env_var"]
            prepend_names: List[str] = []
            for name, raw_value in env_vars.items():
                value = str(raw_value)
                parts = value.split(":")
                stop_tokens = {f"${name}", f"${{{name}}}"}
                prefixes: List[str] = []
                for part in parts:
                    stripped = part.strip()
                    if stripped in stop_tokens:
                        break
                    prefixes.append(stripped)
                if prefixes:
                    prepend_names.append(name)
                    export(
                        prefix + f"ENV_PREPEND_{name.upper()}",
                        ":".join(prefixes),
                    )
            if prepend_names:
                export(prefix + "ENV_PREPEND_VARS", _csv(prepend_names))

        if gpu_tasks:
            metadata = library["metadata"]
            export(prefix + "GPU_LIB", metadata["GPU_LIB"])
            export(prefix + "GPU_LIB_VERSION", metadata["GPU_LIB_VERSION"])
            gpu_load = (library.get("gpu_support") or {}).get("gpu_load") or {}
            if gpu_load.get("type") == "module" and gpu_load.get("module"):
                gpu_module = str(gpu_load["module"])
                modules.append(gpu_module)

        if modules:
            export(prefix + "MODULES", _csv(modules))

        algorithms = library.get("algorithms") or {}
        collective_names = list(algorithms)
        if collective_names:
            export(prefix + "COLLECTIVES", _csv(collective_names))
        is_mpich = "mpich" in library["lib_type"].lower()
        for collective, entries in algorithms.items():
            if not entries:
                continue
            collective_prefix = prefix + collective.upper() + "_ALGORITHMS"
            export(collective_prefix, _csv(entry["name"] for entry in entries))
            skipped = []
            segmented = []
            cvars = []
            for entry in entries:
                has_count_constraint = any(
                    constraint.get("key") == "count"
                    for constraint in entry.get("constraints") or []
                    if isinstance(constraint, dict)
                )
                if has_count_constraint:
                    skipped.append(entry["name"])
                segmented.append(
                    "yes"
                    if "is_segmented" in (entry.get("tags") or [])
                    else "no"
                )
                if is_mpich:
                    selection = entry["selection"]
                    cvars.append("auto" if selection == "pico" else selection)
            export(collective_prefix + "_SKIP", _csv(skipped))
            export(collective_prefix + "_IS_SEGMENTED", _csv(segmented))
            if is_mpich:
                export(collective_prefix + "_CVARS", _csv(cvars))

    return "\n".join(lines) + "\n"


def _write_atomic(
    path: Path,
    content: str,
    mode: int,
    *,
    replace: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_to_exports(config: JsonLike, sh_path: Union[str, Path]) -> str:
    """Validate and atomically write an executable shell export."""

    output = Path(sh_path)
    rendered = render_shell_exports(config)
    _write_atomic(
        output,
        rendered,
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH,
    )
    return str(output)


def save_export_bundle(
    config: Mapping[str, Any],
    json_path: Union[str, Path],
    shell_path: Union[str, Path],
) -> Tuple[str, str]:
    """Validate first, then atomically write the JSON and shell bundle."""

    rendered_shell = render_shell_exports(config)
    rendered_json = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    json_output = Path(json_path)
    shell_output = Path(shell_path)

    _write_atomic(
        json_output,
        rendered_json,
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
        replace=False,
    )
    try:
        _write_atomic(
            shell_output,
            rendered_shell,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH,
            replace=False,
        )
    except BaseException:
        json_output.unlink(missing_ok=True)
        raise

    return str(json_output), str(shell_output)
