"""
Data model definitions for the session configuration.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvironmentSelection:
    """
    User’s environment choice and loaded data.
    """
    name: str = ''
    general: Optional[dict] = None  # general settings
    slurm: Optional[dict] = None    # slurm-specific


@dataclass
class PartitionSelection:
    """
    User’s partition and QOS selection.
    """
    name: str = ''
    qos: str = ''
    details: Optional[dict] = None       # partition‐level keys (desc, CPUs/GPU per node, etc.)
    qos_details: Optional[dict] = None   # ONLY the selected QOS block

@dataclass
class MPILibrarySelection:
    """
    User’s chosen MPI library and its config.
    """
    name: str = ''
    config: Optional[dict] = None


@dataclass
class SessionConfig:
    """
    Full session state carrying all selections.
    """
    environment: EnvironmentSelection = field(default_factory=EnvironmentSelection)
    partition:   PartitionSelection   = field(default_factory=PartitionSelection)
    mpi:         MPILibrarySelection  = field(default_factory=MPILibrarySelection)
    nodes: int = 0
    tasks_per_node: int = 1
    test_time: str = ""
