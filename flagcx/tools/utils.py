import json
import os
from typing import *
from enum import *
from itertools import count


class Collective(Enum):
    P2P = 0
    Custom = 1
    Broadcast = 2
    Gather = 3
    Scatter = 4
    Reduce = 5
    AllReduce = 6
    AllGather = 7
    ReduceScatter = 8
    AlltoAll = 9
    AlltoAllv = 10
    Nop = 11


class RedOp(Enum):
    sum = 0
    prod = 1
    max = 2
    min = 3
    avg = 4
    nop = 5


class Primitive(Enum):
    P2P = 0
    Custom = 1
    Broadcast = 2
    Gather = 3
    Scatter = 4
    Reduce = 5
    AllReduce = 6
    AllGather = 7
    ReduceScatter = 8
    AlltoAll = 9
    AlltoAllv = 10
    Nop = 11
    LocCpy = 12
    LocRed = 13


class DataType(Enum):
    int8 = 0
    uint8 = 1
    int32 = 2
    uint32 = 3
    int64 = 4
    uint64 = 5
    float16 = 6
    float32 = 7
    float = 7
    float64 = 8
    bfloat16 = 9


class BufferKind(str, Enum):
    none = "none"
    input = "input"
    output = "output"
    scratch = "scratch"


class P2pType(str, Enum):
    send = "send"
    recv = "recv"
    term = "term"
    wait = "wait"
    put = "put"
    signal = "signal"
    barrier_signal = "barrier_signal"
    wait_signal = "wait_signal"
    put_value = "put_value"
    put_signal = "put_signal"
    get = "get"


class AlgoType(str, Enum):
    dummy = "dummy"
    loc_red = "loc_red"
    grouped_ag = "grouped_ag"
    ring_ag = "ring_ag"
    ring_ar = "ring_ar"
    sliced_ar = "sliced_ar"
    ring_rs = "ring_rs"
    tree_red = "tree_red"
    ipc_ar = "ipc_ar"
    grouped_ar = "grouped_ar"
    ipc_ring_ar = "ipc_ring_ar"
    hierarchical_sliced_ar = "hierarchical_sliced_ar"


FORMAT_VERSION = 7
ALGORITHM_HASH_SCHEMA_VERSION = 2
FNV_OFFSET_BASIS = 14695981039346656037
FNV_PRIME = 1099511628211
DEFAULT_RED_SLICE_SIZE = 65536
SIZE_T_MASK = (1 << 64) - 1
HASH_CONSTANT = 0x9E3779B97F4A7C15
SIZE_COUNT_SUFFIXES = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
}

DATA_TYPE_SIZES = {
    DataType.int8: 1,
    DataType.uint8: 1,
    DataType.int32: 4,
    DataType.uint32: 4,
    DataType.int64: 8,
    DataType.uint64: 8,
    DataType.float16: 2,
    DataType.float32: 4,
    DataType.float64: 8,
    DataType.bfloat16: 2,
}

ALGO_TO_INT = {
    "dummy": 0,
    "loc_red": 1,
    "grouped_ag": 2,
    "ring_ag": 3,
    "ring_ar": 4,
    "sliced_ar": 5,
    "ring_rs": 6,
    "tree_red": 7,
    "ipc_ar": 8,
    "grouped_ar": 9,
    "ipc_ring_ar": 10,
    "hierarchical_sliced_ar": 11,
}

COMM_OP_TO_INT = {
    "send": 0,
    "recv": 1,
    "broadcast": 2,
    "gather": 3,
    "scatter": 4,
    "reduce": 5,
    "all_reduce": 6,
    "all_gather": 7,
    "reduce_scatter": 8,
    "all_to_all": 9,
    "all_to_allv": 10,
    "noop": 11,
}

COLLECTIVE_TO_COMM_OP = {
    Collective.P2P: "send",
    Collective.Custom: "noop",
    Collective.Broadcast: "broadcast",
    Collective.Gather: "gather",
    Collective.Scatter: "scatter",
    Collective.Reduce: "reduce",
    Collective.AllReduce: "all_reduce",
    Collective.AllGather: "all_gather",
    Collective.ReduceScatter: "reduce_scatter",
    Collective.AlltoAll: "all_to_all",
    Collective.AlltoAllv: "all_to_allv",
    Collective.Nop: "noop",
}


def _enum_value(value):
    return value.value if isinstance(value, Enum) else value


def _datatype_value(datatype: Union[DataType, int]) -> int:
    return int(_enum_value(datatype))


def _datatype_size(datatype: Union[DataType, int]) -> int:
    if isinstance(datatype, DataType):
        return DATA_TYPE_SIZES[datatype]
    for candidate, size in DATA_TYPE_SIZES.items():
        if candidate.value == datatype:
            return size
    raise ValueError(f"Unsupported datatype value {datatype}")


def parse_size_count(value: Union[str, int], name: str = "value") -> int:
    """Parse a size/count value, accepting optional binary K/M/G suffixes."""
    if isinstance(value, int):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{name} must not be empty")
        suffix = text[-1].upper()
        multiplier = SIZE_COUNT_SUFFIXES.get(suffix, 1)
        number_text = (
            text[:-1].strip() if suffix in SIZE_COUNT_SUFFIXES else text
        )
        if not number_text:
            raise ValueError(f"{name} has no numeric part: {value!r}")
        try:
            parsed = int(number_text, 0) * multiplier
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def parse_env_size_count(
    env_name: str, default: Optional[Union[str, int]] = None
) -> Optional[int]:
    """Read an environment variable as a size/count with optional K/M/G suffix."""
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None if default is None else parse_size_count(default, env_name)
    return parse_size_count(value, env_name)


def count_from_size_bytes(
    size_bytes: Union[str, int], datatype: Union[DataType, int]
) -> int:
    """Convert a byte size such as '1M' into an element count for a datatype."""
    parsed_size = parse_size_count(size_bytes, "size_bytes")
    type_size = _datatype_size(datatype)
    if parsed_size % type_size != 0:
        raise ValueError(
            f"size_bytes={parsed_size} is not divisible by datatype size {type_size}"
        )
    return parsed_size // type_size


def _red_op_value(red_op: Union[RedOp, int]) -> int:
    return int(_enum_value(red_op))


def _red_op_from_value(red_op: Union[RedOp, int]) -> RedOp:
    if isinstance(red_op, RedOp):
        return red_op
    return RedOp(red_op)


def _comm_op_name(collective: Collective, comm_op: Optional[str]) -> str:
    if comm_op is not None:
        return comm_op
    return COLLECTIVE_TO_COMM_OP[collective]


def _hash_combine(seed: int, value: int) -> int:
    return (
        seed
        ^ ((value & SIZE_T_MASK) + HASH_CONSTANT + ((seed << 6) & SIZE_T_MASK) + (seed >> 2))
    ) & SIZE_T_MASK


def _algorithm_hash(algo: str, config: Mapping[str, int]) -> int:
    """Match getUniRunnerDagAlgorithmHash using normalized builder inputs."""
    fields = {
        "loc_red": ("num_slices", "buffer_mode"),
        "ring_ag": ("num_slices", "buffer_mode"),
        "ring_ar": ("num_slices", "buffer_mode"),
        "grouped_ag": ("group_size", "buffer_mode"),
        "sliced_ar": ("num_slices", "num_red_slices", "buffer_mode"),
        "ring_rs": ("num_slices", "num_red_slices", "buffer_mode"),
        "tree_red": ("num_slices", "num_red_slices", "buffer_mode"),
        "ipc_ar": ("num_slices", "num_red_slices", "topology_hash", "buffer_mode"),
        "ipc_ring_ar": ("num_channels", "simple_buffer_bytes", "nthreads",
                        "chunk_steps", "slice_steps", "topology_hash", "buffer_mode"),
        "hierarchical_sliced_ar": ("num_slices", "num_red_slices", "group_size", "buffer_mode"),
        "dummy": (),
    }
    if algo == "grouped_ar":
        names = ["num_slices", "num_local_slices", "num_red_slices"]
        if config.get("pipeline_version", 0):
            names.append("pipeline_version")
        names.extend(("group_size", "buffer_mode"))
    else:
        names = fields[algo]
    values = [ALGORITHM_HASH_SCHEMA_VERSION, ALGO_TO_INT[algo]]
    values.extend(config.get(name, 0) for name in names)
    result = FNV_OFFSET_BASIS
    for value in values:
        if not isinstance(value, int) or value < 0 or value > SIZE_T_MASK:
            raise ValueError("algorithm configuration values must fit uint64")
        for byte in range(8):
            result ^= (value >> (8 * byte)) & 0xff
            result = (result * FNV_PRIME) & SIZE_T_MASK
    return result


def _write_json_file(path: str, payload: Mapping):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _rank_chunk(count: int, nranks: int, chunk: int) -> Tuple[int, int]:
    base = count // nranks
    remainder = count % nranks
    chunk_count = base + (1 if chunk < remainder else 0)
    offset = chunk * base + min(chunk, remainder)
    return offset, chunk_count


def _slice_chunk(count: int, num_slices: int, slice_idx: int) -> Tuple[int, int]:
    base = count // num_slices
    remainder = count % num_slices
    slice_count = base + (1 if slice_idx < remainder else 0)
    offset = slice_idx * base + min(slice_idx, remainder)
    return offset, slice_count


def _effective_red_slices(
    count: int,
    nranks: int,
    num_slices: int,
    num_red_slices: int,
    red_slice_size: int,
) -> int:
    if num_red_slices != 0:
        return num_red_slices
    if count == 0 or red_slice_size == 0 or num_slices == 0 or nranks <= 0:
        return 1
    divisor = nranks * num_slices * red_slice_size
    return max(1, (count + divisor - 1) // divisor)


def _rank_slice(count: int, nranks: int, num_slices: int, chunk: int, slice_idx: int) -> Tuple[int, int]:
    chunk_offset, chunk_count = _rank_chunk(count, nranks, chunk)
    slice_offset, slice_count = _slice_chunk(chunk_count, num_slices, slice_idx)
    return chunk_offset + slice_offset, slice_count
