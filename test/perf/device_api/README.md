# Device API benchmarks

These benchmarks use Window registration (`-R 2`), which is also the default.
Explicit `-R 0` and `-R 1` are unsupported.

Select the Window mode before launching the ranks:

| Communication path / backend | `FLAGCX_TEST_WIN_MODE` | Allocation setting |
|---|---|---|
| Hetero / Default | `asymmetric` (default) | Ordinary IPC-exportable allocation; the launcher defaults `FLAGCX_VMM_ENABLE` to `0` if unset |
| Hetero / Default | `symmetric` | Supported symmetric/VMM allocation, normally `FLAGCX_VMM_ENABLE=1` |
| Homo / NCCL or NVSHMEM | `symmetric` (explicit) | Vendor-compatible allocation or the SHMEM symmetric heap |

Default backend runs require the hetero communication path
(`FLAGCX_USE_HETERO_COMM=1`), including on machines with identical devices.
Both Default modes prepare NET registration for cross-node peers. A requested
mode is never silently replaced by the other mode.

For example, with a library built for the Default backend:

```sh
FLAGCX_USE_HETERO_COMM=1 FLAGCX_TEST_WIN_MODE=asymmetric FLAGCX_VMM_ENABLE=0 \
  mpirun -np 2 ./build/bin/perf_allreduce_intranode -R 2
```

The benchmarks synchronize local work and converge import teardown results on
their MPI communicator before releasing backing buffers. Cleanup failure keeps
the backing allocations alive. This application-level coordination does not
change the local semantics of `flagcxCommWindowDeregister`.
