# Iluvatar CI known issues

Temporary exclusions for [PR #670](https://github.com/flagos-ai/FlagCX/pull/670),
based on run [42185973360](https://github.com/flagos-ai/FlagCX/actions/runs/42185973360)
at `ee88e46`. These exclusions record unverified functionality, not passing tests.

| TODO | Case | Excluded modes | Reason | Restore when |
| --- | --- | --- | --- | --- |
| T-3 | `FlagCXCollTest.SendRecv` | HYBRID P2P, forced NET, PXN | Repeated timeout; a local reproduction remains in stream synchronization | Communication and synchronization complete with correct data in all three modes |
| T-4 | `FlagCXCollTest.AlltoAll` | PXN only | Both logical-node relay counts are zero | Topology is verified and actual relay plus data correctness are demonstrated |

The Iluvatar runner environment enables `FLAGCX_CI_ILUVATAR_QUARANTINE=1`.
Set it to `0` before invoking `run_unit_test.sh` inside the CI container to run
these cases for diagnosis and restoration. Invalid values fail the invocation.
Remove each exclusion after its restoration condition has been verified.

The wrapper matches exact test names and explicit MPI mode arguments, not log
labels. It keeps default HOMO SendRecv, ordinary AlltoAll, NET multi-QP striping,
PXN AllReduce and all unrelated cases enabled. New failures still fail CI.
An invocation containing only quarantined cases fails rather than reporting an
empty run as successful.

Every excluded case emits a warning with its TODO and reason. The same record
is saved in `iluvatar-runner-quarantine.tsv` under `FLAGCX_CI_LOG_DIR` (uploaded
with attempt logs), and in the step summary when that path is available.

Existing capability handling remains separate:

- T-1: VMM-only SymMem scenarios are gated by
  `FLAGCX_CI_SYMMEM_VMM_SUPPORTED=0`; IPC, NET fallback and route checks remain.
- T-2: `gpu-mr-optional` accepts only the recognized runtime GPU MR capability
  skip. Missing required cases and actual failures remain errors.
- T-5/T-6: tests retain their existing capability probes and individual skips
  for unsupported interfaces. No whole Adaptor, RMA or SymMem suite is excluded.
