# Iluvatar CI capability limits and restoration checks

The runner now executes `FlagCXCollTest.SendRecv` in HYBRID P2P, forced NET
and PXN modes, and `FlagCXCollTest.AlltoAll` in PXN mode. Historical T-3/T-4
quarantine has been removed. The earlier failures were recorded in
[run 42185973360](https://github.com/flagos-ai/FlagCX/actions/runs/42185973360):
SendRecv timed out and neither logical node reported PXN relay traffic.
Restoration is complete only when a new CoreX hardware CI run confirms correct
data, completion without timeout, and relay traffic from both logical nodes.

The Iluvatar adaptor suite uses the shared required-by-default GPU MR policy.
`NetAdaptorLoopback.GpuIput`, `GpuIget`, `GpuBatchPutAndGet` and `GpuBounds`
must all execute and pass; an unsupported-capability skip fails the suite.

Other capability limits remain separate:

- T-1: VMM-only SymMem scenarios are gated by
  `FLAGCX_CI_SYMMEM_VMM_SUPPORTED=0`; IPC, NET fallback and route checks remain.
- T-5/T-6: tests retain their existing capability probes and individual skips
  for unsupported interfaces. No whole Adaptor, RMA or SymMem suite is excluded.
