import copy
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TUI_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = TUI_DIR.parent
sys.path.insert(0, str(TUI_DIR))

from export_builder import (  # noqa: E402
    ExportValidationError,
    render_shell_exports,
    save_export_bundle,
)


def base_config(*, slurm=False, gpu=False):
    environment = {
        "name": "leonardo" if slurm else "local",
        "desc": "Test environment",
        "slurm": slurm,
        "launcher": "srun" if slurm else "mpiexec",
        "launcher_flags": None if slurm else "--map-by :OVERSUBSCRIBE",
    }
    if slurm:
        environment["partition"] = {
            "name": "boost_usr_prod",
            "qos": {
                "name": "default",
                "is_required": False,
                "extra_requirements": None,
            },
        }

    tests = {"gpu": [4]} if gpu else {"cpu": [8]}
    metadata = (
        {
            "MPI_LIB_COMPILER": "gcc",
            "MPI_LIB_COMPILER_VERSION": "12.2.0",
            "GPU_LIB": "CUDA",
            "GPU_LIB_VERSION": "12.1",
        }
        if gpu
        else {
            "MPI_LIB_COMPILER": "gcc",
            "MPI_LIB_COMPILER_VERSION": "12.2.0",
        }
    )
    return {
        "environment": environment,
        "test": {
            "compile_only": False,
            "debug_mode": False,
            "dry_run": False,
            "compress": True,
            "delete": False,
            "number_of_nodes": 2 if slurm else 1,
            "output_level": "summarized",
            "test_time": "00:30:00" if slurm else None,
            "dimensions": {
                "dtype": "int32",
                "sizes_elements": [8, 64],
                "sizes_bytes": [32, 256],
                "segsizes_bytes": [0, 16384],
            },
        },
        "libraries": [
            {
                "name": "Open MPI",
                "desc": "Test MPI",
                "metadata": metadata,
                "tests": tests,
                "standard": "MPI",
                "lib_type": "Open-MPI",
                "version": "4.1.6",
                "compiler": "mpicc",
                "gpu_support": {
                    "gpu": gpu,
                    "gpu_support_native": False,
                    "gpu_load": (
                        {"type": "module", "module": "cuda/12.1"}
                        if gpu
                        else None
                    ),
                },
                "lib_load": {
                    "type": "module",
                    "module": "openmpi/4.1.6",
                },
                "algorithms": {
                    "allgather": [
                        {
                            "name": "default",
                            "selection": "0",
                            "constraints": [],
                            "tags": [],
                        }
                    ]
                },
            }
        ],
    }


def source_exports(script):
    names = (
        "RUN",
        "RUNFLAGS",
        "EXCLUDE_NODES",
        "JOB_DEP",
        "LIB_0_GPU_LIB",
        "LIB_0_GPU_LIB_VERSION",
    )
    command = (
        "source \"$1\"\n"
        + "\n".join(f'printf "%s\\\\0" "${{{name}-}}"' for name in names)
    )
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(script)],
        check=True,
        capture_output=True,
    )
    values = result.stdout.decode().split("\0")[:-1]
    return dict(zip(names, values))


class ExportBuilderTests(unittest.TestCase):
    def test_local_launcher_and_flags_are_exported(self):
        rendered = render_shell_exports(base_config())
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "test.sh"
            script.write_text(rendered)
            exports = source_exports(script)

        self.assertEqual(exports["RUN"], "mpiexec")
        self.assertEqual(exports["RUNFLAGS"], "--map-by :OVERSUBSCRIBE")

    def test_slurm_options_are_exported(self):
        config = base_config(slurm=True)
        config["test"]["exclude_nodes"] = "node[01-04],node07"
        config["test"]["job_dependency"] = "1234:5678"
        rendered = render_shell_exports(config)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "test.sh"
            script.write_text(rendered)
            exports = source_exports(script)

        self.assertEqual(exports["RUN"], "srun")
        self.assertEqual(exports["EXCLUDE_NODES"], "node[01-04],node07")
        self.assertEqual(exports["JOB_DEP"], "1234:5678")

    def test_gpu_metadata_is_exported(self):
        rendered = render_shell_exports(base_config(slurm=True, gpu=True))
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "test.sh"
            script.write_text(rendered)
            exports = source_exports(script)

        self.assertEqual(exports["LIB_0_GPU_LIB"], "CUDA")
        self.assertEqual(exports["LIB_0_GPU_LIB_VERSION"], "12.1")

    def test_lumi_rocm_metadata_is_exported(self):
        general_path = (
            REPOSITORY_DIR
            / "config/environment/lumi/lumi_general.json"
        )
        libraries_path = (
            REPOSITORY_DIR
            / "config/environment/lumi/lumi_libraries.json"
        )
        general = json.loads(general_path.read_text())
        raw_library = json.loads(libraries_path.read_text())["LIBRARY"][
            "Cray MPICH"
        ]
        config = base_config(slurm=True, gpu=True)
        config["environment"].update(general)
        config["environment"]["partition"]["name"] = "standard-g"
        library = config["libraries"][0]
        library.update(
            {
                "name": "Cray MPICH",
                "standard": "MPI",
                "lib_type": raw_library["lib_type"],
                "version": raw_library["version"],
                "compiler": raw_library["compiler"],
                "metadata": raw_library["metadata"],
                "lib_load": {
                    "type": raw_library["load"]["type"],
                    "module": raw_library["load"]["module"],
                },
                "gpu_support": {
                    "gpu": True,
                    "gpu_support_native": False,
                    "gpu_load": {"type": "default"},
                },
            }
        )

        rendered = render_shell_exports(config)

        self.assertIn("export RUN='srun'", rendered)
        self.assertIn("export LIB_0_GPU_LIB='ROCM'", rendered)
        self.assertIn("export LIB_0_GPU_LIB_VERSION='6.0.3'", rendered)

    def test_multiple_libraries_and_collectives_are_exported(self):
        config = base_config()
        second_library = copy.deepcopy(config["libraries"][0])
        second_library["name"] = "MPICH"
        second_library["lib_type"] = "MPICH"
        second_library["version"] = "4.3.0"
        second_library["algorithms"] = {
            "allgather": [],
            "allreduce": [
                {
                    "name": "recursive_doubling_mpich",
                    "selection": "recursive_doubling",
                }
            ],
        }
        config["libraries"].append(second_library)

        rendered = render_shell_exports(config)

        self.assertIn("export LIB_COUNT='2'", rendered)
        self.assertIn(
            "export LIB_1_COLLECTIVES='allgather,allreduce'",
            rendered,
        )
        self.assertIn(
            "export LIB_1_ALLREDUCE_ALGORITHMS='recursive_doubling_mpich'",
            rendered,
        )

    def test_compile_only_does_not_require_workload(self):
        config = base_config()
        config["test"] = {
            "compile_only": True,
            "debug_mode": False,
            "dry_run": False,
            "compress": False,
            "delete": False,
            "number_of_nodes": 1,
        }
        config["libraries"][0].pop("tests")
        config["libraries"][0].pop("algorithms")

        rendered = render_shell_exports(config)

        self.assertIn("export COMPILE_ONLY='yes'", rendered)
        self.assertNotIn("export TYPES=", rendered)

    def test_incomplete_descriptor_is_rejected(self):
        config = base_config()
        del config["libraries"][0]["algorithms"]

        with self.assertRaisesRegex(
            ExportValidationError,
            r"libraries\[0\]\.algorithms",
        ):
            render_shell_exports(config)

    def test_unsafe_scheduler_inputs_are_rejected(self):
        config = base_config(slurm=True)
        config["test"]["inject_params"] = "--constraint=gpu; touch /tmp/pwned"
        config["test"]["exclude_nodes"] = "node01\nnode02"
        config["test"]["job_dependency"] = "123:bad"

        with self.assertRaises(ExportValidationError) as context:
            render_shell_exports(config)

        message = str(context.exception)
        self.assertIn("test.inject_params", message)
        self.assertIn("test.exclude_nodes", message)
        self.assertIn("test.job_dependency", message)

    def test_legacy_launcher_fallback_is_retained(self):
        config = base_config()
        del config["environment"]["launcher"]
        del config["environment"]["launcher_flags"]

        self.assertIn("export RUN='mpirun'", render_shell_exports(config))

    def test_bundle_is_validated_before_either_file_is_written(self):
        config = base_config()
        del config["test"]["dimensions"]

        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "test.json"
            shell_path = Path(directory) / "test.sh"
            with self.assertRaises(ExportValidationError):
                save_export_bundle(config, json_path, shell_path)

            self.assertFalse(json_path.exists())
            self.assertFalse(shell_path.exists())

    def test_bundle_writes_sourceable_json_and_executable_shell(self):
        config = base_config(slurm=True)
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "test.json"
            shell_path = Path(directory) / "test.sh"
            save_export_bundle(config, json_path, shell_path)

            self.assertEqual(json.loads(json_path.read_text()), config)
            self.assertTrue(shell_path.stat().st_mode & stat.S_IXUSR)
            self.assertEqual(source_exports(shell_path)["RUN"], "srun")

    def test_bundle_does_not_overwrite_existing_outputs(self):
        config = base_config()
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "test.json"
            shell_path = Path(directory) / "test.sh"
            json_path.write_text("keep me")

            with self.assertRaises(FileExistsError):
                save_export_bundle(config, json_path, shell_path)

            self.assertEqual(json_path.read_text(), "keep me")
            self.assertFalse(shell_path.exists())


if __name__ == "__main__":
    unittest.main()
