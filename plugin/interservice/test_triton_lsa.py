# SPDX-License-Identifier: Apache-2.0
"""
Test: Intra-node LSA (Local Shared Access) semantics via FlagCX Device API + Triton.

This test demonstrates the full workflow:
  1. Init host communicator (flagcxCommInitRank)
  2. Allocate device buffer (flagcxMemAlloc)
  3. Register buffer with symmetric window (flagcxCommWindowRegister)
  4. Create DevComm + DevMem
  5. Retrieve device pointers for Triton
  6. Launch a Triton kernel that uses FlagCX Device API IR bitcode
     to perform intra-node peer pointer access (LSA read)

Usage (single-node, 2 GPUs):
  torchrun --nproc_per_node=2 test_triton_lsa.py
"""

import os
import tempfile

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.cuda.memory import CUDAPluggableAllocator
from torch.utils.cpp_extension import load_inline

from flagcx_wrapper import (
    FLAGCX_UNIQUE_ID_BYTES,
    FLAGCXLibrary,
    flagcxDevCommRequirements,
    flagcxUniqueId,
    FLAGCX_WIN_COLL_SYMMETRIC,
)

# Path to FlagCX Device API LLVM bitcode (built by bindings/ir/nvidia)
FLAGCX_BITCODE_PATH = os.environ.get(
    "FLAGCX_BITCODE_PATH",
    os.path.join(os.path.dirname(__file__), "../../build/lib/libflagcx_device.bc"),
)

# FlagCX include path for compiling the allocator wrapper
FLAGCX_INCLUDE_PATH = os.environ.get(
    "FLAGCX_INCLUDE_PATH",
    os.path.join(os.path.dirname(__file__), "../../flagcx/include"),
)

# FlagCX library path for linking
FLAGCX_LIB_PATH = os.environ.get(
    "FLAGCX_LIB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../build/lib"),
)

# ============================================================
# FlagCX pluggable allocator (wraps flagcxMemAlloc/flagcxMemFree)
# ============================================================

flagcx_allocator_source = """
#include <flagcx.h>
extern "C" {

void* flagcx_alloc_plug(size_t size, int device, void* stream) {
  void* ptr = nullptr;
  flagcxResult_t err = flagcxMemAlloc(&ptr, size);
  if (err != flagcxSuccess) {
    return nullptr;
  }
  return ptr;
}

void flagcx_free_plug(void* ptr, size_t size, int device, void* stream) {
  if (ptr != nullptr) {
    flagcxMemFree(ptr);
  }
}

}
"""

_allocator = None
_allocator_wrapper = None
_mem_pool = None
_flagcx_allocator_failed_to_compile = False


def compile_flagcx_allocator():
    global _allocator, _allocator_wrapper, _flagcx_allocator_failed_to_compile

    try:
        out_dir = tempfile.gettempdir()
        lib_name = "flagcx_allocator"
        lib_path = os.path.join(out_dir, f"{lib_name}.so")

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        if rank == 0:
            if not os.path.isfile(lib_path):
                print(
                    f"[INFO] FlagCX allocator not found, compiling: {lib_path}",
                    flush=True,
                )

                load_inline(
                    name=lib_name,
                    cpp_sources=flagcx_allocator_source,
                    with_cuda=True,
                    extra_ldflags=[
                        f"-L{FLAGCX_LIB_PATH}",
                        "-lflagcx",
                        f"-Wl,-rpath,{FLAGCX_LIB_PATH}",
                    ],
                    verbose=True,
                    is_python_module=False,
                    build_directory=out_dir,
                    extra_include_paths=[FLAGCX_INCLUDE_PATH],
                )
            else:
                print(
                    f"[INFO] Using cached FlagCX allocator: {lib_path}",
                    flush=True,
                )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not os.path.isfile(lib_path):
            raise FileNotFoundError(
                f"FlagCX allocator library not found after compilation: {lib_path}"
            )

        _allocator_wrapper = CUDAPluggableAllocator(
            lib_path,
            "flagcx_alloc_plug",
            "flagcx_free_plug",
        )

        _allocator = _allocator_wrapper.allocator()

    except Exception as e:
        _flagcx_allocator_failed_to_compile = True
        print(
            f"[WARNING] Failed to load FlagCX memory allocator: {e}",
            flush=True,
        )


def get_flagcx_mem_pool():
    """Return a cached PyTorch MemPool backed by flagcxMemAlloc."""
    global _mem_pool, _flagcx_allocator_failed_to_compile
    if _mem_pool is None and not _flagcx_allocator_failed_to_compile:
        compile_flagcx_allocator()
        if _allocator is not None:
            _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


def _cleanup_flagcx_mem_pool():
    global _mem_pool
    _mem_pool = None


def _cleanup_flagcx_allocator_wrapper():
    global _allocator_wrapper
    _allocator_wrapper = None


import atexit
atexit.register(_cleanup_flagcx_allocator_wrapper)
atexit.register(_cleanup_flagcx_mem_pool)


# ============================================================
# Triton kernel: read peer's data via FlagCX intra-node pointer
# ============================================================

@triton.jit
def lsa_read_kernel(
    dev_comm_ptr,  # flagcxDevComm device pointer
    dev_mem_ptr,   # flagcxDevMem device pointer (symmetric window)
    output_ptr,    # output buffer (local device memory)
    N: tl.constexpr,  # number of elements
):
    """
    Each rank reads element [rank_id] from peer (rank+1) % nRanks
    via FlagCX intra-node pointer access, and writes it to output_ptr[rank_id].

    This tests LSA semantics: symmetric window gives direct peer memory access.
    """
    pid = tl.program_id(0)
    # Query comm info
    my_rank = tl.extern_fn(
        "flagcxDevCommGetIntraRank", dev_comm_ptr, return_type=tl.int32
    )
    n_ranks = tl.extern_fn(
        "flagcxDevCommGetIntraSize", dev_comm_ptr, return_type=tl.int32
    )
    # Determine peer
    peer = (my_rank + 1) % n_ranks

    # Get peer pointer via LSA (intra-node symmetric window)
    offset = tl.cast(pid, tl.uint64) * 4  # float32 = 4 bytes
    peer_ptr = tl.extern_fn(
        "flagcxGetIntraPointerC",
        dev_mem_ptr,
        offset,
        peer,
        return_type=tl.pointer_type(tl.float32),
    )

    # Load from peer
    val = tl.load(peer_ptr)

    # Store to local output
    tl.store(output_ptr + pid, val)


# ============================================================
# Main test logic
# ============================================================

def main():
    # Initialize torch distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}] Starting LSA test (world_size={world_size})")

    # Initialize FlagCX
    flagcx = FLAGCXLibrary()

    # Create unique ID on rank 0 and broadcast
    if rank == 0:
        unique_id = flagcx.flagcxGetUniqueId()
        id_bytes = bytes(unique_id.internal)
    else:
        id_bytes = b"\x00" * FLAGCX_UNIQUE_ID_BYTES

    # Broadcast unique_id bytes via torch distributed
    id_tensor = torch.frombuffer(bytearray(id_bytes), dtype=torch.uint8).cuda()
    dist.broadcast(id_tensor, src=0)
    id_bytes = id_tensor.cpu().numpy().tobytes()

    if rank != 0:
        unique_id = flagcx.unique_id_from_bytes(id_bytes)

    # Init FlagCX communicator
    comm = flagcx.flagcxCommInitRank(world_size, unique_id, rank)
    print(f"[Rank {rank}] FlagCX comm initialized")

    # Allocate buffer via flagcxMemAlloc (wrapped in torch pluggable allocator)
    N = 64
    buf_size = N * 4  # float32

    flagcx_pool = get_flagcx_mem_pool()
    if flagcx_pool is None:
        raise RuntimeError(
            "Failed to initialize FlagCX memory pool. "
            "Check compilation warnings above."
        )
    with torch.cuda.use_mem_pool(flagcx_pool):
        buf_tensor = torch.full((N,), float(rank), dtype=torch.float32, device="cuda")

    buf_ptr = buf_tensor.data_ptr()

    # Register buffer with symmetric window for LSA
    win = flagcx.flagcxCommWindowRegister(comm, buf_ptr, buf_size,
                                           flags=FLAGCX_WIN_COLL_SYMMETRIC)
    print(f"[Rank {rank}] Window registered (symmetric)")

    # Create DevComm with 1 intra barrier
    reqs = flagcxDevCommRequirements()
    reqs.intraMulticast = False
    reqs.barrierCount = 0
    reqs.intraBarrierCount = 1
    reqs.interBarrierCount = 0
    reqs.intraLLA2ABlockCount = 0
    reqs.intraLLA2ASlotCount = 0
    reqs.interForceEnable = False
    reqs.interContextCount = 4
    reqs.interSignalCount = 0
    reqs.interCounterCount = 0

    dev_comm = flagcx.flagcxDevCommCreate(comm, reqs)
    print(f"[Rank {rank}] DevComm created")

    # Create DevMem (with window)
    dev_mem = flagcx.flagcxDevMemCreate(comm, buf_ptr, buf_size, win)
    print(f"[Rank {rank}] DevMem created")

    # Get device pointers for Triton
    dev_comm_dptr = flagcx.flagcxDevCommGetDevicePtr(dev_comm)
    dev_mem_dptr = flagcx.flagcxDevMemGetDevicePtr(dev_mem)
    print(f"[Rank {rank}] Device pointers: comm={dev_comm_dptr.value:#x}, "
          f"mem={dev_mem_dptr.value:#x}")

    # Allocate output buffer
    output = torch.zeros(N, dtype=torch.float32, device="cuda")

    # Synchronize all ranks before kernel launch
    dist.barrier()

    # Launch Triton kernel
    grid = (N,)
    lsa_read_kernel[grid](
        dev_comm_dptr.value,
        dev_mem_dptr.value,
        output.data_ptr(),
        N=N,
        extern_libs={"libflagcx_device": FLAGCX_BITCODE_PATH},
    )
    torch.cuda.synchronize()

    # Verify: kernel reads from peer = (intra_rank + 1) % intra_size.
    # Buffer is filled with float(rank). In single-node: intra_rank == rank.
    peer_rank = (rank + 1) % world_size
    expected = torch.full((N,), float(peer_rank), dtype=torch.float32, device="cuda")
    if torch.allclose(output, expected):
        print(f"[Rank {rank}] PASSED: read peer rank {peer_rank} data correctly")
    else:
        print(f"[Rank {rank}] FAILED: expected {peer_rank}, got {output[:4].tolist()}")

    # Cleanup
    dist.barrier()
    flagcx.flagcxDevMemFreeDevicePtr(dev_mem)
    flagcx.flagcxDevCommFreeDevicePtr(dev_comm)
    flagcx.flagcxDevMemDestroy(comm, dev_mem)
    flagcx.flagcxDevCommDestroy(comm, dev_comm)
    flagcx.flagcxCommWindowDeregister(comm, win)
    # buf_tensor memory is managed by torch, no explicit free needed
    flagcx.flagcxCommDestroy(comm)

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    main()
