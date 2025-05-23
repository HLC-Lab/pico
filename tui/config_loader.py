import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"

def load_json(*path_parts):
    path = CONFIG_DIR.joinpath(*path_parts)
    with open(path, 'r') as f:
        return json.load(f)

# Helpers to list available environments, partitions, MPIs

def list_environments():
    return [p.name for p in (CONFIG_DIR / 'environments_new').iterdir() if p.is_dir()]

def get_environment_general(env_name):
    return load_json('environments_new', env_name, f"{env_name}_general.json")

def get_environment_slurm(env_name):
    return load_json('environments_new', env_name, f"{env_name}_slurm.json")


def list_mpi_libs(env_name):
    mpi_cfg = load_json('environments_new', env_name, f"{env_name}_mpi.json")
    return list(mpi_cfg.get('MPI', {}).keys())


def get_mpi_config(env_name, mpi_name):
    mpi_cfg = load_json('environments_new', env_name, f"{env_name}_mpi.json")
    return mpi_cfg['MPI'][mpi_name]
