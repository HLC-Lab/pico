# Environment Configuration Guide

PICO supports two configuration systems:
- **TUI format** (recommended) — `config/environment/<name>/` with JSON files
- **Legacy CLI format** — `config/environments/<name>.sh` (see existing files for reference)

---

## Directory Structure

```
config/environment/<name>/
├── <name>_general.json       # Required
├── <name>_libraries.json     # Required
└── <name>_slurm.json         # Optional (only if slurm: true)
```

---

## Schema Reference

### `<name>_general.json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Environment identifier |
| `desc` | string | yes | Human-readable description |
| `slurm` | bool | yes | `true` for SLURM clusters, `false` for local execution |
| `python_module` | string | no | Module to load for Python (e.g. `"python/3.11"`) |
| `other_var` | object | no | Extra environment variables to export (e.g. `{"UCX_IB_SL": "1"}`) |

### `<name>_libraries.json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `NETWORK_LIB` | string | yes | Network library name (e.g. `"ucx"`) |
| `NETWORK_LIB_VERSION` | string | yes | Network library version |
| `LIBRARY` | object | yes | Map of library name to library entry |

**Library entry fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `desc` | string | yes | Human-readable description |
| `standard` | string | yes | `"mpi"` or `"nccl"` |
| `lib_type` | string | yes | `"Open-MPI"`, `"MPICH"`, `"Cray-MPICH"`, `"Intel-MPI"`, or `"nccl"` |
| `version` | string | yes | Library version |
| `compiler` | string | yes | Compiler wrapper (e.g. `"mpicc"`, `"cc"`, `"nvcc"`) |
| `load` | object | yes | Loading mechanism (see below) |
| `gpu` | object | no | GPU support: `{ "support": bool, "load": { ... } }` |
| `metadata` | object | no | Key-value pairs for result metadata CSV |

**Load mechanism:**

| Type | Fields | Description |
|------|--------|-------------|
| `"default"` | `{ "type": "default" }` | No special setup needed |
| `"module"` | `{ "type": "module", "module": "..." }` | Load via `module load` |
| `"env_var"` | `{ "type": "env_var", "vars": { "PATH": "...", ... } }` | Set environment variables |

### `<name>_slurm.json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `PARTITIONS` | object | yes | Map of partition name to partition entry |

**Partition entry fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `desc` | string | yes | Partition description |
| `is_gpu` | bool | yes | Whether this is a GPU partition |
| `gpus_per_node` | int | if GPU | Number of GPUs per node |
| `cpus_per_node` | int | yes | Number of CPUs per node |
| `sockets_per_node` | int | no | Sockets per node |
| `QOS` | object | yes | Map of QoS name to QoS entry |

**QoS entry fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `required` | bool | yes | Whether the QoS must be explicitly selected |
| `desc` | string | yes | QoS description |
| `nodes_limit` | object | yes | `{ "min": int, "max": int }` — node count range |
| `time_limit` | string | yes | Wall time limit (format `"HH:MM:SS"` or `"D-HH:MM:SS"`) |
| `extra_requirements` | object | no | Extra SLURM requirements (e.g. `{"tasks_per_node": 32, "gres": "gpu:4"}`) |

---

## Examples

### Local machine (no SLURM)

`mybox_general.json`:
```json
{
  "name": "mybox",
  "desc": "My development machine.",
  "slurm": false
}
```

`mybox_libraries.json`:
```json
{
  "NETWORK_LIB": "ucx",
  "NETWORK_LIB_VERSION": "1.14.0",
  "LIBRARY": {
    "Open MPI": {
      "desc": "Default system Open MPI.",
      "standard": "mpi",
      "lib_type": "Open-MPI",
      "version": "5.0.7",
      "compiler": "mpicc",
      "gpu": { "support": false },
      "load": { "type": "default" },
      "metadata": {
        "MPI_LIB_COMPILER": "gcc",
        "MPI_LIB_COMPILER_VERSION": "14.2.1"
      }
    },
    "Custom MPICH": {
      "desc": "Locally built MPICH.",
      "standard": "mpi",
      "lib_type": "MPICH",
      "version": "4.2.0",
      "compiler": "mpicc",
      "gpu": { "support": false },
      "load": {
        "type": "env_var",
        "vars": {
          "PATH": "/opt/mpich/bin:$PATH",
          "LD_LIBRARY_PATH": "/opt/mpich/lib:$LD_LIBRARY_PATH"
        }
      },
      "metadata": {
        "MPI_LIB_COMPILER": "gcc",
        "MPI_LIB_COMPILER_VERSION": "14.2.1"
      }
    }
  }
}
```

### SLURM cluster with GPU support

`supercloud_general.json`:
```json
{
  "name": "supercloud",
  "desc": "SuperCloud HPC cluster.",
  "slurm": true,
  "python_module": "python/3.12.0",
  "other_var": {
    "UCX_MAX_RNDV_RAILS": "2"
  }
}
```

`supercloud_libraries.json`:
```json
{
  "NETWORK_LIB": "ucx",
  "NETWORK_LIB_VERSION": "1.15.0",
  "LIBRARY": {
    "Open MPI 5.0": {
      "desc": "Open MPI 5.0 with CUDA support.",
      "standard": "mpi",
      "lib_type": "Open-MPI",
      "version": "5.0.0",
      "compiler": "mpicc",
      "load": {
        "type": "module",
        "module": "openmpi/5.0.0--gcc--12.3.0"
      },
      "gpu": {
        "support": true,
        "load": {
          "type": "module",
          "module": "cuda/12.2"
        }
      },
      "metadata": {
        "MPI_LIB_COMPILER": "gcc",
        "MPI_LIB_COMPILER_VERSION": "12.3.0",
        "GPU_LIB": "CUDA",
        "GPU_LIB_VERSION": "12.2"
      }
    },
    "NCCL 2.20": {
      "desc": "NVIDIA NCCL for GPU collectives.",
      "standard": "nccl",
      "lib_type": "nccl",
      "version": "2.20.5",
      "compiler": "nvcc",
      "load": {
        "type": "module",
        "module": "nccl/2.20.5--cuda-12.2"
      },
      "gpu": {
        "support": true,
        "load": { "type": "default" }
      },
      "metadata": {
        "GPU_LIB": "CUDA",
        "GPU_LIB_VERSION": "12.2"
      }
    }
  }
}
```

`supercloud_slurm.json`:
```json
{
  "PARTITIONS": {
    "gpu_prod": {
      "desc": "Production GPU partition.",
      "is_gpu": true,
      "gpus_per_node": 4,
      "cpus_per_node": 64,
      "sockets_per_node": 2,
      "QOS": {
        "default": {
          "required": false,
          "desc": "Default QoS.",
          "nodes_limit": { "min": 2, "max": 64 },
          "time_limit": "24:00:00"
        },
        "long": {
          "required": true,
          "desc": "Long-running jobs.",
          "nodes_limit": { "min": 2, "max": 8 },
          "time_limit": "7-00:00:00",
          "extra_requirements": {
            "tasks_per_node": 64,
            "gres": "gpu:4"
          }
        }
      }
    },
    "cpu_prod": {
      "desc": "Production CPU partition.",
      "is_gpu": false,
      "cpus_per_node": 128,
      "sockets_per_node": 2,
      "QOS": {
        "default": {
          "required": false,
          "desc": "Default QoS.",
          "nodes_limit": { "min": 2, "max": 256 },
          "time_limit": "48:00:00"
        }
      }
    }
  }
}
```

---

## Step-by-step: Adding a New Environment

1. **Create the directory:** `mkdir config/environment/<name>/`
2. **Create `general.json`:** Set name, description, and whether the cluster uses SLURM
3. **Create `libraries.json`:** List all available MPI and NCCL libraries with their loading mechanism
4. **If SLURM, create `slurm.json`:** Define partitions and QoS options
5. **Verify:** Launch the TUI — the new environment will appear in the "Configure" step

See `config/environment/local/`, `config/environment/leonardo/`, and `config/environment/lumi/` for complete real-world examples.

---

## Legacy CLI Format

For the CLI workflow (being phased out), environments are shell scripts under `config/environments/<name>.sh`. Required variables include:

- `PICOCC` — MPI compiler wrapper
- `RUN` — job launcher (`mpiexec`, `srun`)
- `MPI_LIB` / `MPI_LIB_VERSION`
- `PARTITION_CPUS_PER_NODE` / `PARTITION_GPUS_PER_NODE`
- (non-local only) `PARTITION` / `PICO_ACCOUNT`
- `load_other_env_var()` — exported function for dynamic setup

See `config/environments/local.sh` for a working example.
