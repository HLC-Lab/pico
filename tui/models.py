# Copyright (c) 2025 Saverio Pasqualoni
# Licensed under the MIT License

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Union, Optional, Dict, List, Any, cast
from datetime import timedelta
import re

_duration_re = re.compile(r'''
    ^\s*
    (?:(\d+)-)?                       # optional days (group 1) + dash
    (?:
        (?:[01]\d|2[0-3])             # hours 00–23
        :([0-5]\d)                    # minutes (group 2)
        :([0-5]\d)                    # seconds (group 3)
      | 24:00:00                      # OR exactly 24:00:00
    )
    \s*$
''', re.VERBOSE)

def parse_duration(s: str):
    m = _duration_re.match(s)
    if not m:
        return False

    days = int(m.group(1)) if m.group(1) is not None else 0

    hours, minutes, seconds = map(int, s.split('-')[-1].split(':'))

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


@dataclass
class QosSelection:
    """
    Represents a single QoS entry under a given partition & environment.
    """
    partition: str
    environment: str
    desc: str = ''
    name: str = ''
    is_required: bool = False
    nodes_limit: Dict[str, int] = field(default_factory=dict)
    time_limit: str = ''
    extra_requirements: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"{self.name} (partition={self.partition}, env={self.environment})"

    def get_help(self) -> str:
        return self.desc

    def validate(self) -> bool:
        if not (self.partition and self.environment and self.desc and self.name and self.time_limit):
            return False

        min_n = self.nodes_limit.get('min', 0)
        max_n = self.nodes_limit.get('max', 0)
        if 0 in (min_n, max_n) or min_n <= 0 or max_n <= 0 or min_n > max_n:
            return False

        if self.is_required and self.name in ['default', '']:
            return False

        return True

    def from_dict(self, slurm_json: Dict[str, Any], qos: str) -> None:
        qos_options = slurm_json.get('PARTITIONS', {}).get(self.partition, {}).get('QOS', {}).get(qos, {})
        if not qos_options:
            raise ValueError(f"QoS '{qos}' not found in SLURM config for partition '{self.partition}'.")
        self.name = qos
        self.desc = qos_options.get('desc', '')
        self.is_required = qos_options.get('required', False)
        self.time_limit = qos_options.get('time_limit', '')
        self.nodes_limit = qos_options.get('nodes_limit', {})
        self.extra_requirements = qos_options.get('extra_requirements', None)

@dataclass
class PartitionSelection:
    """
    Represents a single partition under an environment.
    Holds metadata and can produce QosSelection objects.
    """
    environment: str
    desc: str = ''
    name: str = ''
    is_gpu: bool = False
    cpus_per_node: int = 0
    sockets_per_node: int = 0
    gpus_per_node: Optional[int] = None
    qos: QosSelection = field(default_factory=lambda: QosSelection(partition='', environment=''))

    PART_REQUIRED_FIELDS = ['desc', 'name', 'is_gpu', 'cpus_per_node', 'sockets_per_node']

    def __str__(self) -> str:
        return f"{self.name} (env={self.environment})"

    def get_help(self) -> str:
        msg = f"{self.name} ({self.environment}):\n{self.desc}\n" \
            f"CPUs per node: {self.cpus_per_node}\n" \
            f"Sockets per node: {self.sockets_per_node}"
        if self.is_gpu:
            msg += f"GPUs per node: {self.gpus_per_node}"
        return msg

    def from_dict(self, slurm_json: Dict[str, Any], partition: str) -> None:
        partition_options = slurm_json.get('PARTITIONS', {}).get(partition, {})
        if not partition_options:
            raise ValueError(f"Partition '{partition}' not found in SLURM config.")

        self.name = partition
        self.desc = partition_options.get('desc', '')
        self.is_gpu = partition_options.get('is_gpu', False)
        self.cpus_per_node = partition_options.get('cpus_per_node', 0)
        self.sockets_per_node = partition_options.get('sockets_per_node', 0)
        if self.is_gpu:
            self.gpus_per_node = partition_options.get('gpus_per_node', 0)

    def validate(self) -> bool:
        missing = [f for f in self.PART_REQUIRED_FIELDS if getattr(self, f) in (None, '')]
        if missing:
            return False
        if not self.__validate_cpu_config():
            return False
        if self.is_gpu and not self.__validate_gpu_config():
            return False
        if not self.__validate_qos_consistency():
            return False
        return True

    def __validate_cpu_config(self) -> bool:
        if (self.sockets_per_node * self.cpus_per_node) <= 0:
            return False
        if (self.cpus_per_node % self.sockets_per_node) != 0:
            return False
        return True

    def __validate_gpu_config(self) -> bool:
        if not isinstance(self.gpus_per_node, int) or self.gpus_per_node <= 0:
            return False
        else:
            return True

    def __validate_qos_consistency(self) -> bool:
        if self.name != self.qos.partition:
            return False
        if self.environment != self.qos.environment:
            return False
        return True

    def init_qos(self) -> None:
        self.qos = QosSelection(environment=self.environment, partition=self.name)

@dataclass
class EnvironmentSelection:
    desc: str = ''
    name: str = ''
    slurm: bool = False
    python_module: Optional[str] = None
    other_var: Optional[Dict[str, Any]] = None
    partition: Optional[PartitionSelection] = None

    def __str__(self) -> str:
        return self.name

    def get_help(self) -> str:
        return f"{self.name}:\n{self.desc}"

    def from_dict(self, env_json: Dict[str, Any]) -> None:
        self.desc = env_json.get('desc', '')
        self.name = env_json.get('name', '')
        self.slurm = env_json.get('slurm', False)
        self.python_module = env_json.get('python_module')
        self.other_var = env_json.get('other_var')

    def init_partition(self) -> None:
        self.partition = PartitionSelection(environment=self.name)

    def __validate(self) -> bool:
        if not (self.name and self.desc):
            return False
        if self.slurm and not self.python_module:
            return False
        return True

    def validate(self) -> bool:
        if not self.slurm:
            return self.__validate()

        if not (self.partition and self.partition.qos):
            return False

        return (self.__validate() and self.partition.validate() and self.partition.qos.validate())

    def get_summary(self) -> str:
        if self.partition and self.partition.qos:
            return f"{self.name}, {self.partition.name} (QOS: {self.partition.qos.name})"
        return self.name

class CDtype(Enum):
    UNKNOWN = 'unknown'
    CHAR = 'char'
    FLOAT = 'float'
    DOUBLE = 'double'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'

    def __str__(self) -> str:
        return self.value

    def get_size(self) -> int:
        sizes = {
            CDtype.UNKNOWN: 0,
            CDtype.CHAR: 1,
            CDtype.FLOAT: 4,
            CDtype.DOUBLE: 8,
            CDtype.INT8: 1,
            CDtype.INT16: 2,
            CDtype.INT32: 4,
            CDtype.INT64: 8
        }
        return sizes[self]


@dataclass
class TestDimension:
    dtype: CDtype = CDtype.UNKNOWN
    sizes_bytes: List[int] = field(default_factory=list)
    sizes_elements: List[int] = field(default_factory=list)
    segsizes_bytes: List[int] = field(default_factory=list)

    def get_printable_sizes(self, get_segment_sizes=False) -> List[str]:
        sizes = []
        source_sizes = self.segsizes_bytes if get_segment_sizes else self.sizes_bytes

        for size in source_sizes:
            if size < 1024:
                sizes.append(f"{size} B")
            elif size < 1024**2:
                sizes.append(f"{size / 1024:.2f} KiB")
            else:
                sizes.append(f"{size / 1024**2:.2f} MiB")
        return sizes


    def validate(self) -> bool:
        dtype_size = self.dtype.get_size()
        if self.dtype == CDtype.UNKNOWN or dtype_size == 0:
            return False

        expected_elements = [size // dtype_size for size in self.sizes_bytes]
        if self.sizes_elements != expected_elements:
            return False

        size_lists = [
            self.sizes_elements,
            self.sizes_bytes,
            self.segsizes_bytes
        ]

        for size_list in size_lists:
            if not size_list:
                return False
            if any(size <= 0 for size in size_list) and size_list != self.segsizes_bytes:
                return False
            if len(size_list) != len(set(size_list)):
                return False

        if len(self.sizes_elements) != len(self.sizes_bytes):
            return False

        return True

    def fill_elements(self) -> None:
        if self.sizes_bytes:
            dtype_size = self.dtype.get_size()
            if dtype_size == 0:
                raise ValueError(f"Invalid dtype size: {dtype_size}")
            self.sizes_elements = [size // dtype_size for size in self.sizes_bytes]


class OutputLevel(Enum):
    FULL = 'full'
    STATISTICS = 'statistics'
    MINIMAL = 'minimal'
    SUMMARY = 'summary'

    @classmethod
    def from_str(cls, value: str):
        value = value.lower()
        if value == 'full':
            return cls.FULL
        elif value == 'statistics':
            return cls.STATISTICS
        elif value == 'minimal':
            return cls.MINIMAL
        elif value == 'summary':
            return cls.SUMMARY
        else:
            raise ValueError(f"Unknown output level: {value}")

    def __str__(self) -> str:
        return self.value

    def get_help(self) -> str:
        if self == OutputLevel.FULL:
            return "Full output with detailed results. Every test time of each task for each iteration is saved."
        elif self == OutputLevel.STATISTICS:
            return "Saves only max, min, mean and stddev for each iteration."
        elif self == OutputLevel.MINIMAL:
            return "Saves only max for each iteration."
        elif self == OutputLevel.SUMMARY:
            return "Saves only statistics summary of the whole test, no iteration data is saved."
        else:
            return "Unknown output level."

@dataclass
class TestConfig:
    compile_only: bool = False
    debug_mode: bool = False
    dry_run: bool = False
    compress: bool = False
    delete: bool = False
    number_of_nodes: int = 1
    output_level: Optional[OutputLevel] = None
    inject_params: Optional[str] = None
    dimensions: Optional[TestDimension] = field(default_factory=lambda: TestDimension())
    test_time: Optional[str] = None
    exclude_nodes: Optional[str] = None
    job_dependency: Optional[Union[str, int]] = None

    def validate(self, session) -> bool:
        if self.compile_only:
            if self.number_of_nodes != 1:
                return False
            if (self.test_time or self.exclude_nodes or
                self.job_dependency or self.dimensions or
                self.dry_run or self.compress or
                self.delete or self.output_level):
                return False
            return True

        if not (self.output_level and isinstance(self.output_level, OutputLevel)):
            return False

        if self.delete and not self.compress:
            return False

        if not (self.dimensions and self.dimensions.validate()):
            return False
        if session.environment.slurm:
            if not self.validate_nodes(session, self.number_of_nodes):
                return False
            if not self.validate_time(session, self.test_time):
                return False
        else:
            if self.number_of_nodes != 1:
                return False
            if self.test_time or self.exclude_nodes or self.job_dependency:
                return False

        return True

    @staticmethod
    def validate_nodes(session, n_nodes) -> bool:
        if isinstance(n_nodes, str):
            if not (n_nodes and n_nodes.isdigit()):
                return False
            n_nodes = int(n_nodes)

        if session.environment.slurm:
            if not (session.environment.partition and session.environment.partition.qos):
                return False
            min_nodes = session.environment.partition.qos.nodes_limit.get('min', 1)
            max_nodes = session.environment.partition.qos.nodes_limit.get('max', 1)
            if not (min_nodes <= n_nodes <= max_nodes):
                return False
        elif n_nodes != 1:
            return False

        return True

    @staticmethod
    def validate_time(session, test_time) -> bool:
        if test_time is None:
            return False
        if session.environment.slurm:
            min_time = parse_duration('00:00:01')
            max_time = parse_duration(session.environment.partition.qos.time_limit)
            this_time = parse_duration(test_time)

            if False in (min_time, max_time, this_time):
                return False

            # Not necessary to cast, here to silence type checker
            min_time = cast(timedelta, min_time)
            max_time = cast(timedelta, max_time)
            this_time = cast(timedelta, this_time)
            if not (min_time <= this_time <= max_time):
                return False

        elif test_time is not None:
            return False

        return True

    def get_summary(self) -> str:
        summary = []
        summary.append(f"Nodes: {self.number_of_nodes}")
        if self.dry_run:
            summary.append("Dry Run")
        if self.compile_only:
            summary.append("Compile Only")
        if self.debug_mode:
            summary.append("Debug Mode")
        if self.test_time:
            summary.append(f"Test Time: {self.test_time}")
        if self.exclude_nodes:
            summary.append(f"Exclude Nodes: {self.exclude_nodes}")
        if self.job_dependency:
            summary.append(f"Job Dependency: {self.job_dependency}")
        if self.inject_params:
            summary.append(f"Inject Params: {self.inject_params}")
        if self.dimensions:
            sizes = self.dimensions.get_printable_sizes()
            seg_sizes = self.dimensions.get_printable_sizes(get_segment_sizes=True)
            summary.append(f"Dimensions: {', '.join(sizes)}")
            summary.append(f"Segment Sizes: {', '.join(seg_sizes)}")
            summary.append(f"Data Type: {self.dimensions.dtype}")
        if self.output_level:
            summary.append(f"Output Level: {self.output_level}")
        if self.compress:
            summary.append("Compression Enabled")
            if self.delete:
                summary.append("Delete Unompressed files after Test")
        return ', '.join(summary) if summary else "No test configuration set"

class LoadType(Enum):
    DEFAULT = 'default'
    MODULE = 'module'
    SET_ENV = 'set_env'

    @classmethod
    def from_str(cls, value: str):
        value = value.lower()
        if value == 'default':
            return cls.DEFAULT
        elif value == 'module':
            return cls.MODULE
        elif value == 'env_var':
            return cls.SET_ENV
        else:
            raise ValueError(f"Unknown load type: {value}")

    def __str__(self) -> str:
        return self.value

class StdType(Enum):
    MPI = 'MPI'
    NCCL = 'NCCL'
    UNKNOWN = 'unknown'

    @classmethod
    def from_str(cls, value: str):
        value = value.lower()
        if value == 'mpi':
            return cls.MPI
        elif value == 'nccl':
            return cls.NCCL
        else:
            return cls.UNKNOWN

    def __str__(self) -> str:
        return self.value


class LibType(Enum):
    OPEN_MPI = 'Open-MPI'
    MPICH = 'MPICH'
    CRAY_MPICH = 'Cray-MPICH'
    NCCL = 'NCCL'
    UNKNOWN = 'unknown'

    @classmethod
    def from_str(cls, value: str):
        value = value.lower()
        if value == 'open-mpi':
            return cls.OPEN_MPI
        elif value == 'mpich':
            return cls.MPICH
        elif value == 'cray-mpich':
            return cls.CRAY_MPICH
        elif value == 'nccl':
            return cls.NCCL
        else:
            return cls.UNKNOWN

    def __str__(self) -> str:
        return self.value



@dataclass
class LibraryLoad:
    type: LoadType = LoadType.DEFAULT
    module: Optional[str] = None
    env_var: Optional[Dict[str, str]] = None

    def validate(self) -> bool:
        if self.type == LoadType.MODULE and not self.module:
            return False
        if self.type == LoadType.SET_ENV and not self.env_var:
            return False
        return True

    def from_dict(self, lib_load: Dict[str, Any]) -> None:
        self.type = LoadType.from_str(lib_load.get('type', 'default'))
        if self.type == LoadType.MODULE:
            self.module = lib_load.get('module')
            if not self.module:
                raise ValueError("Module name is required for LoadType.MODULE")
        elif self.type == LoadType.SET_ENV:
            self.env_var = lib_load.get('vars', {})
            if not isinstance(self.env_var, dict):
                raise ValueError("vars must be a dictionary for LoadType.SET_ENV")
            for key, value in self.env_var.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError(f"env_var key and value must be strings: {key}={value}")

    #TODO: Improve this to work well in the json generation
    def __str__(self) -> str:
        if self.type == LoadType.MODULE:
            return f"MODULE: {self.module}"
        elif self.type == LoadType.SET_ENV:
            if not self.env_var:
                raise ValueError("env_var must be set for LoadType.SET_ENV")
            return f"SET_ENV: {', '.join(f'{k}={v}' for k, v in self.env_var.items())}"
        else:
            return "DEFAULT"

@dataclass
class GPUSupport:
    gpu: bool = False
    gpu_load: Optional[LibraryLoad] = None

    def validate(self) -> bool:
        if self.gpu and not (self.gpu_load and self.gpu_load.validate()):
            return False

        return True

    @classmethod
    def from_dict(cls, gpu_json: Dict[str, Any]) -> "GPUSupport":
        gpu = gpu_json.get('support', False)
        gpu_load = None
        if gpu:
            gpu_load_data = gpu_json.get('load', {})
            if not isinstance(gpu_load_data, dict):
                raise ValueError("GPU load data must be a dictionary")
            gpu_load = LibraryLoad()
            gpu_load.from_dict(gpu_load_data)
        return cls(gpu=gpu, gpu_load=gpu_load)

class CollectiveType(Enum):
    UNKNOWN = 'unknown'
    ALLTOALL = 'alltoall'
    ALLREDUCE = 'allreduce'
    ALLGATHER = 'allgather'
    BARRIER = 'barrier'
    BCAST = 'bcast'
    GATHER = 'gather'
    SCATTER = 'scatter'
    REDUCE = 'reduce'
    REDUCE_SCATTER = 'reduce_scatter'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str):
        value = value.lower()
        if value == 'alltoall':
            return cls.ALLTOALL
        elif value == 'allreduce':
            return cls.ALLREDUCE
        elif value == 'allgather':
            return cls.ALLGATHER
        elif value == 'barrier':
            return cls.BARRIER
        elif value in ('bcast', 'broadcast'):
            return cls.BCAST
        elif value == 'gather':
            return cls.GATHER
        elif value == 'scatter':
            return cls.SCATTER
        elif value == 'reduce':
            return cls.REDUCE
        elif value in ('reduce_scatter', 'reducescatter'):
            return cls.REDUCE_SCATTER
        else:
            raise ValueError(f"Unknown collective type: {value}")

@dataclass
class AlgorithmSelection:
    name: str = ''
    coll: CollectiveType = CollectiveType.UNKNOWN
    desc: str = ''
    version: str = ''
    selection: Union[str, int] = ''
    constraints: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None

    def validate(self) -> bool:
        if ('' in (self.name, self.desc, self.version, self.selection) or
            self.coll == CollectiveType.UNKNOWN):
            return False
        return True

    @classmethod
    def from_dict(cls, algo_name: str, coll: str, algo_dict: dict) -> "AlgorithmSelection":
        collective = CollectiveType.from_str(coll)
        desc = algo_dict.get('desc', '')
        version = algo_dict.get('version', '')
        selection = algo_dict.get('selection', '')
        constraints = algo_dict.get('constraints')
        tags = algo_dict.get('tags')

        if '' in (desc, version, selection):
            raise ValueError(f"Algorithm {algo_name} for collective {coll} is missing required fields.")

        if collective == CollectiveType.UNKNOWN:
            raise ValueError(f"Unknown collective type: {collective}")

        return cls(
            name=algo_name,
            desc=desc,
            coll=collective,
            version=version,
            selection=selection,
            constraints=constraints,
            tags=tags
        )

class TestType(Enum):
    CPU = 'cpu'
    GPU = 'gpu'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str):
        if value.lower() == 'cpu':
            return cls.CPU
        elif value.lower() == 'gpu':
            return cls.GPU
        else:
            raise ValueError(f"Unknown test type: {value}")


@dataclass
class LibrarySelection:
    name: str = ''
    desc: str = ''
    tests: Dict[TestType, List[int]] = field(default_factory=dict)
    standard: StdType = StdType.UNKNOWN
    lib_type: LibType = LibType.UNKNOWN
    version: str = ''
    compiler: str = ''
    gpu_support: GPUSupport = field(default_factory=lambda: GPUSupport())
    lib_load: LibraryLoad = field(default_factory=lambda: LibraryLoad())
    pico_backend: bool = False
    algorithms: Dict[CollectiveType, List[AlgorithmSelection]] = field(default_factory=dict)

    def get_summary(self) -> str:
        summary = f"Library: {self.name}\n" 
        if self.tests.get(TestType.CPU, False):
            summary += f"CPU Tests: {', '.join(map(str, self.tests.get(TestType.CPU, [])))}\n"
        if self.tests.get(TestType.GPU, False):
            summary += f"GPU Tests: {', '.join(map(str, self.tests.get(TestType.GPU, [])))}\n"
        if self.algorithms:
            algos = ', '.join(f"{coll}: {[algo.name for algo in algos]}" for coll, algos in self.algorithms.items())
            summary += f"Algorithms: {algos}\n"
        return summary

    @classmethod
    def from_dict(cls, lib_json: Dict[str, Any], name: str) -> "LibrarySelection":
        desc = lib_json.get('desc', '')
        version = lib_json.get('version', '')
        compiler = lib_json.get('compiler', '')
        standard = StdType.from_str(lib_json.get('standard', ''))
        library = LibType.from_str(lib_json.get('lib_type', ''))
        gpu_support = GPUSupport.from_dict(lib_json.get('gpu', {}))

        lib_load = LibraryLoad()
        lib_load.from_dict(lib_json.get('load', {}))

        return cls(
            name=name,
            desc=desc,
            standard=standard,
            lib_type=library,
            version=version,
            compiler=compiler,
            gpu_support=gpu_support,
            lib_load=lib_load
        )

    # def validate(self, validate_algo=False, r=False) -> bool:
    def validate(self, validate_algo=False) -> bool:
        if not (self.name and self.desc and self.version and self.compiler and 
                self.standard != StdType.UNKNOWN and self.lib_type != LibType.UNKNOWN):
            #if r: raise  ValueError("1")
            return False
        if self.standard == StdType.NCCL and not self.gpu_support.gpu:
            #if r: raise  ValueError("2")
            return False
        if not (self.lib_load.validate() and self.gpu_support.validate()):
            #if r: raise  ValueError("3")
            return False
        if not self.tests:
            #if r: raise  ValueError("4")
            return False
        if self.tests.get(TestType.CPU, []) and self.standard == StdType.NCCL:
            #if r: raise  ValueError("5")
            return False
        if self.tests.get(TestType.GPU, []):
            if not self.gpu_support.gpu:
                #if r: raise  ValueError("6")
                return False
            if not self.gpu_support.gpu_load or not self.gpu_support.gpu_load.validate():
                #if r: raise  ValueError("7")
                return False

        if validate_algo:
            to_delete = []
            for coll, algos in self.algorithms.items():
                if not algos:
                    to_delete.append(coll)
                    continue
            for coll in to_delete:
                del self.algorithms[coll]
                if not self.algorithms:
                    #if r: raise  ValueError("8")
                    return False

            for coll, algos in self.algorithms.items():
                for algo in algos:
                    if algo.coll != coll or not algo.validate():
                        #if r: raise  ValueError("9")
                        return False
        return True


    def get_id_name(self) -> str:
        return self.name.replace(' ','_').replace('.','_').replace('-','_').lower()


#TODO: Maybe is better to modify libraries to be a dict with library names as keys?
@dataclass
class SessionConfig:
    environment: EnvironmentSelection = field(default_factory=lambda: EnvironmentSelection())
    test: TestConfig = field(default_factory=lambda: TestConfig())
    libraries: List[LibrarySelection] = field(default_factory=list)

    @staticmethod
    def _merge_dicts(base: Dict[str, Any], partial: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in partial.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = SessionConfig._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result


    @staticmethod
    def _convert(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return str(obj)
        if isinstance(obj, dict):
            return { SessionConfig._convert(k): SessionConfig._convert(v) for k, v in obj.items() }
        if isinstance(obj, list):
            return [ SessionConfig._convert(item) for item in obj ]
        return obj

    @staticmethod
    def _prune_none(obj: Any) -> Any:
        if isinstance(obj, dict):
            pruned = {}
            for k, v in obj.items():
                pv = SessionConfig._prune_none(v)
                # Exclude None or empty lists/dicts
                if pv is None:
                    continue
                if isinstance(pv, dict) and not pv:
                    continue
                if isinstance(pv, list) and not pv:
                    continue
                pruned[k] = pv
            return pruned
        if isinstance(obj, list):
            return [SessionConfig._prune_none(item) for item in obj if SessionConfig._prune_none(item) is not None]
        return obj

    def to_dict(self, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = asdict(self)
        base = self._convert(raw)
        if partial:
            base = self._merge_dicts(base, partial)

        return self._prune_none(base)

    def get_summary(self) -> str:
        env = self.environment.get_summary()
        test = self.test.get_summary()
        libs = "\n".join(lib.get_summary() for lib in self.libraries)

        return f"Environment: {env}\n\n" \
               f"Test Configuration:\n{test}\n\n" \
               f"Libraries:\n{libs}"


