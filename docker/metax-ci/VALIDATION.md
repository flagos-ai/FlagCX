# MACA 3.8.1.3 image validation

## RDMA provider fix, 2026-11-09

The repaired `2.3.0` image (also published as `2.3.0-rdma235.2.96.0`) has digest
`sha256:e9b511953d3e8675b1e8d215b7d2f4777a5d7898bd2cee9e7e58a6663c56e575`.
The CI image reference pins this digest so a node cannot reuse the previous tag's
cached image.

The Dockerfile copies Broadcom `libbnxt_re` 235.2.96.0 from the digest-pinned
previous Ubuntu 24.04 CI image, after installing Ubuntu's RDMA dependencies.
Its SHA-256 is
`2032c6861b1c4ab38fc2c00597e8c4d927e9b42a8c5beff85516eb14dc3c7be4`.
This provider supports the nodes' ABI 8; Ubuntu's original provider accepted
only ABI 1 and enumerated no devices.

Validation used source commit `cacc2a3` in a local hardware container with a
read-only root filesystem. The node has automation that replaces RDMA libraries
inside writable Docker containers, so initial writable-container results were
excluded. The provider checksum in the unstarted image and after every test
remained identical to the Dockerfile's checksum.

| Check | Result |
| --- | --- |
| RDMA enumeration/open | `ibv_devices` lists the Broadcom devices; `ibv_devinfo` opens all four `bnxt_roce` devices with ACTIVE ports and no ABI warnings |
| MACA, Torch, MCCL, MPI | Device enumeration, four-device tensor operations, MCCL AllToAll symbol and four-rank MPI launch pass |
| Unit suites | adaptor, core, p2p, rma, runner, service and symmem exit 0 |
| RMA network path | Uses IB/RoCE; the 54 host tests pass; mode/capability-dependent MPI skips remain |
| Torch API | Homogeneous and heterogeneous four-rank runs complete without `[SKIP]`; all ranks reach the last collective |
| Python Torch tests | 17 tests pass and JSON coverage is generated |
| Performance | Eight homogeneous and seven heterogeneous operations pass with verification enabled; existing configured homogeneous AllReduce/ReduceScatter skips remain |
| CI regression checks | 28 tests pass |

After the first six suites passed, other GPU workloads appeared on the node.
An interrupted symmem attempt encountered MACA queue timeouts and was excluded.
The final symmem, Torch and perf runs used `MACA_VISIBLE_DEVICES=4,5,6,7` and
logical CUDA devices 0-3. Local host network/IPC, driver/device access and
unlimited memlock were provided by the validation environment; repository
workflows retain the Kubernetes job-container execution path with no Docker
launch settings. Kubernetes CI still requires a separate remote run.

Logs and per-check provider verification are retained under
`/tmp/flagcx-metax-rdma-20261109/`, including `results.json`, `*-readonly.log`,
`*-gpus4567.log`, and `preflight-readonly.log`.

## Original image validation

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
