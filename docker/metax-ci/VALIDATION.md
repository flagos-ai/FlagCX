# MACA 3.8.1.3 image validation

Validated on 2026-11-09 with the MetaX C550 node's driver 3.9.25 and eight
devices. The checked-out source was `17c5adc` plus the CI changes in this
directory's commit, with the repository-pinned googletest and JSON submodules.
The hardware environment used host network/IPC, unlimited memlock and the
host's read-only `/opt/mxdriver` mount, matching the existing CI requirements.

Image: `harbor.baai.ac.cn/flagos-dev/flagcx-metax-maca3.8.1.3:2.3.0`

Image manifest index:
`sha256:12a0fed42dd5e6657e157ded9263228077dce19dbf024c9172dff5c812e7ac9f`

| Check | Result |
| --- | --- |
| Python/Torch | `/flagos/bin/python3`, `torch==2.10.0+metax3.8.1.0` |
| Native MACA and Torch enumeration | Both report 8 devices before masking |
| Tensor operations on four devices | Initialization, allocation, sum/readback and synchronization pass |
| Open MPI 4.1.6 | Four ranks start; compiled MPI AllReduce returns 10 as expected |
| Unit suites | adaptor, core, p2p, rma, runner, service and symmem pass |
| Runner coverage | Ordinary, heterogeneous, forced NET, multi-QP striping and PXN pass |
| Torch API | Both homogeneous and heterogeneous four-rank runs complete; no `[SKIP]` |
| Python Torch tests | 17 tests pass; JSON coverage report generated |
| CI routing/preflight regression checks | 73 tests pass |
| No-device negative check | An empty `CUDA_VISIBLE_DEVICES` causes a nonzero preflight exit |

Other vLLM jobs started during the first runner and Torch attempts. Those
attempts were interrupted after queue-creation timeouts and are not counted
as successful validation. After devices became free, Torch and symmem used
physical GPUs 4–7, mapped to logical devices 0–3; runner non-PXN modes used the
same four devices. PXN then ran separately with the unchanged CI topology of
physical GPUs 0,2,4,7 and `bnxt_roce1,bnxt_roce2,bnxt_roce3`.

Suite success does not imply every optional GoogleTest case executed: existing
capability-dependent skips remain in C++ tests. The no-skip requirement above
applies to the two Torch API runs. This validates the image in a local hardware
container; the ARC job Pod still needs the device/driver/RDMA resources described
in README.md, and Kubernetes lifecycle/workspace behavior needs a CI run.

Build and test logs from this run are retained locally under
`/tmp/flagcx-metax-image-20261109/`, including `final-preflight.log`,
`mpi-allreduce.log`, `torch-validation-gpus4567.log` and `workspace/.ci-logs/`.
