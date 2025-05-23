from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EnvironmentSelection:
    name: str = ''
    general: Optional[dict] = None
    slurm: Optional[dict] = None

@dataclass
class PartitionSelection:
    name: str = ''
    qos: str = ''
    details: Optional[dict] = None

@dataclass
class MPILibrarySelection:
    name: str = ''
    config: Optional[dict] = None

@dataclass
class SessionConfig:
    environment: EnvironmentSelection = field(default_factory=EnvironmentSelection)
    partition: PartitionSelection = field(default_factory=PartitionSelection)
    mpi: MPILibrarySelection = field(default_factory=MPILibrarySelection)
