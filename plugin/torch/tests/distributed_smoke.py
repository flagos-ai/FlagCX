"""Two-rank hardware smoke test for vendor and FlagOS Torch backends.

Run this file with torchrun after building the plugin for the selected adaptor
and backend. It intentionally uses the same collective workflow in both modes.
"""

import argparse
import importlib
import os

import torch
import torch.distributed as dist


VENDOR_PACKAGES = {
    "ascend": ("torch_npu", "npu"),
    "enflame": ("torch_gcu", "gcu"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptor", choices=VENDOR_PACKAGES, required=True)
    return parser.parse_args()


def resolve_device_type(adaptor):
    backend = os.environ.get("FLAGCX_TORCH_BACKEND", "vendor").strip().lower()
    if backend == "flagos":
        importlib.import_module("torch_fl")
        return "flagos"
    if backend != "vendor":
        raise RuntimeError(f"Unsupported FLAGCX_TORCH_BACKEND={backend!r}")

    package, device_type = VENDOR_PACKAGES[adaptor]
    importlib.import_module(package)
    return device_type


def assert_tensor_equal(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


def churn_allocator(device, numel):
    """Try to reuse a just-released flattened collective buffer."""
    for value in range(32):
        torch.full((numel,), -value, dtype=torch.float32, device=device)


def main():
    args = parse_args()
    device_type = resolve_device_type(args.adaptor)

    import flagcx  # noqa: F401 - registers the process-group backend

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError("This smoke test requires exactly two ranks")

    device_module = getattr(torch, device_type)
    device_module.set_device(local_rank)
    device = torch.device(device_type, local_rank)
    dist.init_process_group(
        f"cpu:gloo,{device_type}:flagcx",
        rank=rank,
        world_size=world_size,
    )

    try:
        value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        work = dist.all_reduce(value, async_op=True)
        work.wait()
        assert_tensor_equal(value, torch.tensor([3.0]))

        future_value = torch.tensor(
            [rank + 1], dtype=torch.float32, device=device
        )
        future = dist.all_reduce(future_value, async_op=True).get_future()
        future.wait()
        assert_tensor_equal(future_value, torch.tensor([3.0]))

        source = torch.tensor([rank], dtype=torch.int64, device=device)
        gathered = [
            torch.empty_like(source) for _ in range(world_size)
        ]
        work = dist.all_gather(gathered, source, async_op=True)
        churn_allocator(device, world_size)
        work.wait()
        assert_tensor_equal(
            torch.cat(gathered), torch.tensor([0, 1], dtype=torch.int64)
        )

        list_all_to_all_input = [
            torch.tensor(
                [rank * 10 + peer], dtype=torch.float32, device=device
            )
            for peer in range(world_size)
        ]
        list_all_to_all_output = [
            torch.empty(1, dtype=torch.float32, device=device)
            for _ in range(world_size)
        ]
        work = dist.all_to_all(
            list_all_to_all_output,
            list_all_to_all_input,
            async_op=True,
        )
        churn_allocator(device, world_size)
        work.wait()
        assert_tensor_equal(
            torch.cat(list_all_to_all_output),
            torch.tensor([rank, rank + 10], dtype=torch.float32),
        )

        gather_input = torch.tensor(
            [rank + 5], dtype=torch.float32, device=device
        )
        gather_output = (
            [torch.empty_like(gather_input) for _ in range(world_size)]
            if rank == 0
            else None
        )
        work = dist.gather(
            gather_input,
            gather_list=gather_output,
            dst=0,
            async_op=True,
        )
        churn_allocator(device, world_size)
        work.wait()
        if rank == 0:
            assert_tensor_equal(
                torch.cat(gather_output),
                torch.tensor([5, 6], dtype=torch.float32),
            )

        list_reduce_scatter_input = [
            torch.tensor(
                [rank * 10 + peer + 1],
                dtype=torch.float32,
                device=device,
            )
            for peer in range(world_size)
        ]
        list_reduce_scatter_output = torch.empty(
            1, dtype=torch.float32, device=device
        )
        work = dist.reduce_scatter(
            list_reduce_scatter_output,
            list_reduce_scatter_input,
            async_op=True,
        )
        churn_allocator(device, world_size)
        work.wait()
        expected = torch.tensor([12.0 if rank == 0 else 14.0])
        assert_tensor_equal(list_reduce_scatter_output, expected)

        scatter_output = torch.empty(1, dtype=torch.float32, device=device)
        scatter_input = (
            [
                torch.tensor([21 + peer], dtype=torch.float32, device=device)
                for peer in range(world_size)
            ]
            if rank == 0
            else None
        )
        work = dist.scatter(
            scatter_output,
            scatter_list=scatter_input,
            src=0,
            async_op=True,
        )
        churn_allocator(device, world_size)
        work.wait()
        assert_tensor_equal(scatter_output, torch.tensor([21.0 + rank]))

        scatter_input = torch.tensor(
            [rank * 2 + 1, rank * 2 + 2],
            dtype=torch.float32,
            device=device,
        )
        scatter_output = torch.empty(1, dtype=torch.float32, device=device)
        dist.reduce_scatter_tensor(scatter_output, scatter_input)
        expected = torch.tensor([4.0 if rank == 0 else 6.0])
        assert_tensor_equal(scatter_output, expected)

        all_to_all_input = torch.tensor(
            [rank * 10, rank * 10 + 1], dtype=torch.int64, device=device
        )
        all_to_all_output = torch.empty_like(all_to_all_input)
        dist.all_to_all_single(all_to_all_output, all_to_all_input)
        expected = torch.tensor(
            [rank, rank + 10], dtype=torch.int64
        )
        assert_tensor_equal(all_to_all_output, expected)

        barrier = dist.barrier(async_op=True)
        barrier.wait()
        if rank == 0:
            print(
                f"PASS: adaptor={args.adaptor} "
                f"backend={os.environ.get('FLAGCX_TORCH_BACKEND', 'vendor')}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
