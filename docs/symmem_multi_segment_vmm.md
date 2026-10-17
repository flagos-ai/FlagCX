# Multi-segment VMM symmetric windows

CUDA, Hygon DU, MetaX, and PPU adaptors support symmetric windows backed by
multiple VMM physical allocations created by FlagCX. Set `FLAGCX_VMM_ENABLE=1`
and `FLAGCX_VMM_SEGMENT_BYTES=<bytes>` before `flagcxMemAlloc`. The segment
size must be a positive multiple of the provider's VMM allocation granularity.
FlagCX uses the existing single-allocation path when the setting is absent or
the rounded allocation fits in one segment. CUDA requires a toolkit with
`CUDART_VERSION >= 12010` for this path.

The common symmetric heap describes only the physical segments touched by the
registered user range. It normalizes their offsets to the first touched
segment, then exchanges segment sizes and offsets and transfers one shareable
FD per segment in a fixed rank and segment order. It closes each
exported or received FD before advancing to the next segment, so the number
of live FDs per process does not grow with the number of segments. Each adaptor
retains and exports its own physical handles, maps each peer segment, and owns
imported handles until every alias has been unmapped. Collective registration
publishes a window only after every rank succeeds. Latest VMM adaptors map
every successful window into a communicator-level LSA reservation. A failed
reservation, offset allocation, or mapping fails registration collectively;
it cannot publish a window at an unrelated per-window VA. Released windows
return their offsets to the communicator allocator for reuse. Network
registration of a tracked multi-segment VMM allocation prefers exporting the
registered VA range as one DMA-BUF FD and registering it with `regMrDmaBuf`.
When DMA-BUF is disabled or explicitly unsupported for that range, FlagCX
attempts one plain VA `regMr` over the complete registered range if both the
device and NET adaptor declare VMM VA support. A provider may still reject
that particular multi-segment VA. Registration then fails collectively without
publishing a partial window. Set `FLAGCX_DMABUF_ENABLE=1` to permit DMA-BUF MR
registration; the default value `0` skips DMA-BUF and uses an eligible VA
route. Hygon SHCA does not advertise a usable VMM VA NET route.

`FLAGCX_VMM_ENABLE` controls how the native adaptor allocates memory.
`FLAGCX_DMABUF_ENABLE` controls whether NET registration attempts DMA-BUF.
With the latter enabled, an ordinary device allocation such as `cudaMalloc`
also attempts DMA-BUF when the device and NET expose the required operations.
An allocation-specific export rejection falls back to ordinary VA `regMr`.
VMM ranges, including those spanning multiple physical segments, may use the
same fallback only when both adaptors declare VMM VA MR support. Other export
or registration errors remain failures.

An externally created VMM mapping whose physical offset is unknown returns
`flagcxNotSupported` during symmetric-window registration. FlagCX does not
remap it from physical offset zero. An untracked allocation may use the
ordinary-allocation IPC fallback only when its provider can export a legacy
IPC handle from the allocation base. This also supports windows beginning at
an offset inside an ordinary allocation. The adaptors do not retain unknown
VMM handles to decide between these paths.

CUDA and MetaX can also create one communicator-level multicast object and VA
reservation. An eligible window binds its local VMM mapping at the same LSA
offset and unbinds it before releasing the flat mapping or offset. Multicast
requires uniform mapped size and user offset across local ranks and compatible
provider granularity. If multicast setup or binding is unavailable and cleanup
succeeds, the window uses the unicast LSA mapping. DU and PPU do not provide
communicator LSA multicast callbacks. MetaX uses the multicast address-binding
symbols only when the installed SDK exports them.

The local VMM MPI suite runs the LSA reuse, subrange, mapping rollback, and
multicast tests on all four adaptors. Multicast tests skip on hardware without
that capability. On CUDA or MetaX hardware expected to support multicast,
set `FLAGCX_CI_REQUIRE_MULTICAST=1` when running `test/script/symmem_test.sh`
to require successful binding and exercise offset reuse, bind failure fallback,
and unbind retry.
Build each adaptor with `HOST_CXX_STANDARD=-std=c++11` in its hardware CI
environment to check the library's C++11 contract. The test harness keeps its
existing C++17 compiler setting.

The opt-in MPI test `FLAGCX_CI_REQUIRE_LOW_FD_MULTI_SEGMENT=1` uses a
64-segment allocation and a temporarily reduced `RLIMIT_NOFILE` on Linux.
Run it with at least two ranks on one node and `FLAGCX_VMM_ENABLE=1`.
