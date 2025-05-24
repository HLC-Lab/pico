"""
Configuration loader utilities.
Provides functions to read JSON config files for environments and MPI libraries.
"""
import json
from pathlib import Path

# Base config directory (assumes this file sits next to `config/`)
CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_json(*path_parts) -> dict:
    """
    Load a JSON file from CONFIG_DIR / path_parts.

    Args:
        *path_parts: path segments under config/ (e.g. ('environments', 'lumi', 'lumi_general.json')).
    Returns:
        Parsed JSON as a dict.
    """
    path = CONFIG_DIR.joinpath(*path_parts)
    with open(path, 'r') as f:
        return json.load(f)


def list_environments() -> list[str]:
    """
    List all available environment names (subdirectories under config/environments).
    """
    env_dir = CONFIG_DIR / 'environments_new'
    return [p.name for p in env_dir.iterdir() if p.is_dir()]


def get_environment_general(env_name: str) -> dict:
    """
    Get the general settings for a given environment.
    """
    filename = f"{env_name}_general.json"
    return load_json('environments_new', env_name, filename)


def get_environment_slurm(env_name: str) -> dict:
    """
    Get the SLURM-specific settings for an environment.
    """
    filename = f"{env_name}_slurm.json"
    return load_json('environments_new', env_name, filename)


def list_mpi_libs(env_name: str) -> list[str]:
    """
    List available MPI implementations for a given environment.
    """
    mpi_cfg = load_json('environments_new', env_name, f"{env_name}_mpi.json")
    return list(mpi_cfg.get('MPI', {}).keys())


def get_mpi_config(env_name: str, mpi_name: str) -> dict:
    """
    Retrieve the configuration for a specific MPI library.
    """
    mpi_cfg = load_json('environments_new', env_name, f"{env_name}_mpi.json")
    return mpi_cfg['MPI'][mpi_name]
