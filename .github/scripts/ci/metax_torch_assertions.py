#!/usr/bin/env python3
"""Four-rank FlagCX Torch correctness check used only by MetaX CI."""

import argparse
import os

import torch
import flagcx  # noqa: F401  Registers the FlagCX distributed backend.
import torch.distributed as dist


def require_equal(label, actual, expected, rank):
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"{label} rank={rank}: expected={expected.tolist()}, actual={actual.tolist()}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("homogeneous", "heterogeneous"))
    mode = parser.parse_args().mode
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"MetaX Torch correctness check requires four ranks, got {world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError(f"MetaX CUDA-compatible devices unavailable on rank {rank}")
    if mode == "heterogeneous" and os.environ.get("FLAGCX_CLUSTER_SPLIT_LIST") != "2":
        raise RuntimeError("heterogeneous mode requires FLAGCX_CLUSTER_SPLIT_LIST=2")
    if mode == "homogeneous" and os.environ.get("FLAGCX_CLUSTER_SPLIT_LIST"):
        raise RuntimeError("homogeneous mode must not set FLAGCX_CLUSTER_SPLIT_LIST")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("cpu:gloo,cuda:flagcx", rank=rank, world_size=world_size)
    try:
        send = torch.tensor(
            [rank * 10 + target + 1 for target in range(world_size)],
            device="cuda", dtype=torch.float32,
        )
        recv = torch.zeros_like(send)
        dist.all_to_all_single(recv, send)
        expected = torch.tensor(
            [source * 10 + rank + 1 for source in range(world_size)],
            device="cuda", dtype=torch.float32,
        )
        require_equal("all_to_all_single", recv, expected, rank)

        send_list = [torch.full((2,), rank * 10 + target + 1,
                                device="cuda", dtype=torch.float32)
                     for target in range(world_size)]
        recv_list = [torch.zeros(2, device="cuda", dtype=torch.float32)
                     for _ in range(world_size)]
        dist.all_to_all(recv_list, send_list)
        for source, actual in enumerate(recv_list):
            require_equal("all_to_all", actual,
                          torch.full_like(actual, source * 10 + rank + 1), rank)

        value = torch.tensor([rank + 1], device="cuda", dtype=torch.float32)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        require_equal("all_reduce", value,
                      torch.full_like(value, world_size * (world_size + 1) // 2), rank)
        print(f"[ASSERT] MetaX Torch {mode} rank={rank}: AllToAll and AllReduce data passed",
              flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
