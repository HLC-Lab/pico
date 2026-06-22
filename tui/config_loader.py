# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

"""
Configuration loader utilities.
Provides functions to read JSON config files for environments and MPI libraries.
"""
import json
from pathlib import Path
from typing import List

# Base config directory (assumes this file sits next to `config/`)
PICO_DIR = Path(__file__).parent.parent
CONFIG_DIR = PICO_DIR / "config"
ALG_DIR = CONFIG_DIR / "algorithms"
ENV = 'environment'
ENV_DIR = CONFIG_DIR / ENV

def load_json(*path_parts, panic=True) -> dict:
    path = CONFIG_DIR.joinpath(*path_parts)
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        if panic:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return {}


# --------------- ConfigureStep utilities --------------- #

def conf_list_environments() -> List[str]:
    return [p.name for p in ENV_DIR.iterdir() if p.is_dir()]

def conf_get_general(env_name: str) -> dict:
    filename = f"{env_name}_general.json"
    return load_json(ENV, env_name, filename)

def conf_get_slurm_opts(env_name: str) -> dict:
    filename = f"{env_name}_slurm.json"
    return load_json(ENV, env_name, filename)


# --------------- LibrariesStep utilities --------------- #

def lib_get_libraries(env_name: str) -> dict:
    filename = f"{env_name}_libraries.json"
    return load_json(ENV, env_name, filename)

# --------------- AlgorithmsStep utilities --------------- #

def alg_get_list(std_type: str, lib_name: str, coll_name: str) -> dict:
    return load_json("algorithms", std_type, lib_name, f"{coll_name}.json", panic=False)

# WARN: I think this is slow, as it loads the whole file each time.
# Consider caching when using get_alg_list() in AlgorithmStep and delete this.
def alg_get_algo(std_type: str, lib_name: str, coll_name: str, algo_name: str) -> dict:
    coll_algos = alg_get_list(std_type, lib_name, coll_name)
    if algo_name not in coll_algos:
        raise ValueError(f"Algorithm {algo_name} not found in {std_type}/{lib_name}/{coll_name}.json")

    return coll_algos[algo_name]


