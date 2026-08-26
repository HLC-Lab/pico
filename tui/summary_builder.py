from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from models import (
    AlgorithmSelection,
    LibType,
    LoadType,
    SessionConfig,
    TestType,
    get_algorithm_constraint_issue,
)


@dataclass(frozen=True)
class CaseCounts:
    planned: int = 0
    runnable: int = 0
    runtime_skipped: int = 0
    preflight_invalid: int = 0


def _has_count_skip(algorithm: AlgorithmSelection) -> bool:
    for constraint in algorithm.constraints or []:
        if constraint.get("key") != "count":
            continue
        for condition in constraint.get("conditions") or []:
            if (
                condition.get("operator") == ">="
                and condition.get("value") == "comm_sz"
            ):
                return True
    return False


def _nccl_protocol_multiplier(
    library_type: LibType, algorithm_name: str
) -> int:
    if library_type != LibType.NCCL:
        return 1

    name = algorithm_name.lower()
    if name.endswith(("_simple", "_ll", "_ll128")):
        return 1
    if name.endswith("_nccl"):
        name = name[:-5]
    return 3 if name == "ring" else 1


def calculate_case_counts(session: SessionConfig) -> CaseCounts:
    test = session.test
    dimensions = test.dimensions
    if test.compile_only or not dimensions:
        return CaseCounts()

    planned = 0
    runnable = 0
    runtime_skipped = 0
    preflight_invalid = 0

    for library in session.libraries:
        for test_type in (TestType.GPU, TestType.CPU):
            for tasks_per_node in library.tests.get(test_type, []):
                communicator_size = test.number_of_nodes * tasks_per_node
                for algorithms in library.algorithms.values():
                    for algorithm in algorithms:
                        segmented = "is_segmented" in (algorithm.tags or [])
                        segment_multiplier = (
                            len(dimensions.segsizes_bytes) if segmented else 1
                        )
                        protocol_multiplier = _nccl_protocol_multiplier(
                            library.lib_type, algorithm.name
                        )
                        case_multiplier = (
                            segment_multiplier * protocol_multiplier
                        )
                        constraint_issue = get_algorithm_constraint_issue(
                            {"constraints": algorithm.constraints or []},
                            [communicator_size],
                            root=0,
                        )

                        for count in dimensions.sizes_elements:
                            planned += case_multiplier
                            if constraint_issue:
                                preflight_invalid += case_multiplier
                            elif (
                                _has_count_skip(algorithm)
                                and count < communicator_size
                            ):
                                runtime_skipped += case_multiplier
                            else:
                                runnable += case_multiplier

    return CaseCounts(
        planned=planned,
        runnable=runnable,
        runtime_skipped=runtime_skipped,
        preflight_invalid=preflight_invalid,
    )


def _git_description(pico_dir: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(pico_dir),
                "describe",
                "--tags",
                "--always",
                "--dirty",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _csv(values: Iterable[object]) -> str:
    return ", ".join(map(str, values)) or "none"


def _algorithm_lines(session: SessionConfig) -> List[str]:
    lines: List[str] = []
    for library in session.libraries:
        lines.append(f"{library.name}:")
        for collective, algorithms in library.algorithms.items():
            rendered = []
            for algorithm in algorithms:
                backend = (
                    "LibPico"
                    if algorithm.selection == "pico"
                    else f"selector={algorithm.selection}"
                )
                rendered.append(f"{algorithm.name} ({backend})")
            lines.append(
                f"  {collective}: {', '.join(rendered) if rendered else 'none'}"
            )
    return lines


def _warnings(session: SessionConfig, counts: CaseCounts) -> List[str]:
    warnings: List[str] = []
    test = session.test

    if counts.preflight_invalid:
        warnings.append(
            f"{counts.preflight_invalid} case(s) violate algorithm constraints."
        )
    if not session.environment.launcher:
        warnings.append(
            "The environment does not define a launcher; the compatibility "
            "fallback will be used."
        )
    if test.dry_run and session.environment.slurm:
        warnings.append(
            "Remote dry-run currently reaches sbatch; review the generated command carefully."
        )
    if test.debug_mode:
        warnings.append(
            "TUI debug mode does not currently apply all CLI debug overrides."
        )
    if not session.environment.slurm:
        warnings.append(
            "Local execution currently derives mpirun without site-specific launcher flags."
        )

    for library in session.libraries:
        if library.tests.get(TestType.GPU):
            missing_gpu_metadata = [
                key
                for key in ("GPU_LIB", "GPU_LIB_VERSION")
                if not library.metadata.get(key)
            ]
            if missing_gpu_metadata:
                warnings.append(
                    f"{library.name}: missing GPU metadata: "
                    f"{', '.join(missing_gpu_metadata)}."
                )
        if library.lib_type == LibType.NCCL:
            for algorithms in library.algorithms.values():
                if any(algorithm.selection == "pico" for algorithm in algorithms):
                    warnings.append(
                        f"{library.name}: LibPico algorithm names are not yet separated "
                        "from NCCL runtime selectors."
                    )
                    break
        if (
            library.lib_load.type == LoadType.SET_ENV
            and library.lib_load.env_var
            and any("$" in value for value in library.lib_load.env_var.values())
        ):
            warnings.append(
                f"{library.name}: environment paths contain variable references "
                "that the current shell exporter may preserve literally."
            )

    warnings.append(
        "TUI library runs currently capture the full process environment."
    )
    warnings.append("Descriptor schema is currently unversioned.")
    return warnings


def build_effective_summary(
    session: SessionConfig,
    pico_dir: Path,
    *,
    generated_at: Optional[datetime] = None,
    git_description: Optional[str] = None,
) -> str:
    generated_at = generated_at or datetime.now().astimezone()
    revision = git_description or _git_description(pico_dir)
    counts = calculate_case_counts(session)
    environment = session.environment
    partition = environment.partition
    qos = partition.qos if partition else None
    test = session.test
    dimensions = test.dimensions

    lines = [
        "EFFECTIVE EXECUTION PLAN",
        "",
        "Provenance",
        f"  Generated: {generated_at.isoformat(timespec='seconds')}",
        f"  Repository revision: {revision}",
        "  Descriptor schema: legacy / unversioned",
        "",
        "Environment & resources",
        f"  Site: {environment.name}",
        f"  Launcher: "
        f"{environment.launcher or ('srun' if environment.slurm else 'mpirun')}",
        f"  Nodes: {test.number_of_nodes}",
    ]
    if partition:
        lines.append(f"  Partition: {partition.name}")
    if environment.launcher_flags:
        lines.append(f"  Launcher flags: {environment.launcher_flags}")
    if qos:
        lines.append(f"  QoS: {qos.name}")
    if test.test_time:
        lines.append(f"  Time limit: {test.test_time}")
    if test.exclude_nodes:
        lines.append(f"  Excluded nodes: {test.exclude_nodes}")
    if test.job_dependency:
        lines.append(f"  Dependency: afterany:{test.job_dependency}")
    if test.inject_params:
        lines.append(f"  Additional parameters: {test.inject_params}")

    lines.extend(["", "Workload"])
    if test.compile_only:
        lines.append("  Compile-only: no benchmark cases will run")
    elif dimensions:
        lines.extend([
            f"  Datatype: {dimensions.dtype}",
            f"  Message sizes (elements): {_csv(dimensions.sizes_elements)}",
            f"  Message sizes (bytes): {_csv(dimensions.sizes_bytes)}",
            f"  Segment sizes (bytes): {_csv(dimensions.segsizes_bytes)}",
            "  Root: 0",
            "  Reduction operation: MPI_SUM",
        ])

    lines.extend(["", "Libraries & process mapping"])
    for index, library in enumerate(session.libraries):
        cpu_tasks = library.tests.get(TestType.CPU, [])
        gpu_tasks = library.tests.get(TestType.GPU, [])
        load = str(library.lib_load)
        lines.extend([
            f"  [{index}] {library.name} {library.version}",
            f"      Standard/type: {library.standard} / {library.lib_type}",
            f"      Compiler: {library.compiler}",
            f"      Library metadata: "
            f"{_csv(f'{key}={value}' for key, value in library.metadata.items())}",
            f"      Load: {load}",
            f"      CPU tasks/node: {_csv(cpu_tasks)}",
            f"      GPU tasks/node: {_csv(gpu_tasks)}",
            f"      CPU communicator sizes: "
            f"{_csv(test.number_of_nodes * value for value in cpu_tasks)}",
            f"      GPU communicator sizes: "
            f"{_csv(test.number_of_nodes * value for value in gpu_tasks)}",
            f"      LibPico backend: {'yes' if library.pico_backend else 'no'}",
        ])

    lines.extend(["", "Algorithms", *_algorithm_lines(session)])

    lines.extend([
        "",
        "Benchmark matrix",
        f"  Planned invocations: {counts.planned}",
        f"  Runtime count-skipped: {counts.runtime_skipped}",
        f"  Preflight-invalid: {counts.preflight_invalid}",
        f"  Runnable invocations: {counts.runnable}",
        "",
        "Measurement & output",
        "  Iterations: automatic, based on message size",
        "  Root: 0",
        "  Correctness comparison: enabled",
        f"  Debug mode: {'yes' if test.debug_mode else 'no'}",
        f"  Dry run: {'yes' if test.dry_run else 'no'}",
        f"  Output level: {test.output_level or 'not applicable'}",
        f"  Compression: {'yes' if test.compress else 'no'}",
        f"  Delete after compression: {'yes' if test.delete else 'no'}",
        "",
        "Run after saving",
        "  scripts/submit_wrapper.sh -f tests/<saved-name>.sh",
    ])

    warnings = _warnings(session, counts)
    if warnings:
        lines.extend(["", "Warnings"])
        lines.extend(f"  - {warning}" for warning in warnings)

    return "\n".join(lines)
