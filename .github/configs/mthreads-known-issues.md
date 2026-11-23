# Mthreads CI known capability gaps

See [the 2026-11-23 Mthreads container retest](mthreads-rdma-retest-2026-11-23.md)
for the enabled RDMA phases, results, and original failures.

| Coverage | Current gap |
| --- | --- |
| `device_api`, `device_api_unified_ir` | The current MUSA 4.3.6 CI SDK does not provide `musa::atomic_ref`, required by the upstream Device API kernels. Re-enable these suites with an SDK image that provides it. |
| RMA PUT with signal | The runtime does not support a stream acquire of remote writes. The GDR visibility phase verifies that the public wait rejects the operation; it does not claim immediate visibility of a WRITE payload. |
| SymMem VMM | VMM mapping and the VA/DMA-BUF MR routes are unavailable. Only the VMM data phases are excluded. CI exercises ordinary IPC/NET data and verifies rejection of an unsupported VMM route. |
| Shared P2P engine, IBUC, PXN | Disabled in the current Mthreads platform matrix. The ordinary P2P engine, IBRC, and forced NET runner phases remain enabled. |
