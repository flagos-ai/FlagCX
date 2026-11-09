# MetaX FlagCX CI image

Image: `harbor.baai.ac.cn/flagos-dev/flagcx-metax-maca3.8.1.3:2.3.0`

The Dockerfile pins the MACA 3.8.1.3 runtime base by digest and keeps its
`/flagos` Python environment with `torch==2.10.0+metax3.8.1.0`. Do not install
upstream Torch over this vendor build. The previous CI image loaded
`2.10.0+cu128` from `/usr/local/lib/python3.12/dist-packages`, which reported
zero devices and requested an NVIDIA driver although MACA could see eight GPUs.

The image exposes MACA's Open MPI 4.1.6, headers and libraries at
`/usr/local/mpi`, adds the SSH client required by its launcher, and installs
RDMA development dependencies, coverage 7.10.7, and gcovr 8.2. The preinstalled
FlagCX package is removed so CI builds and tests the checked-out source.

Build from the repository root (registry login is configured separately):

```bash
docker build --network=host \
  -t harbor.baai.ac.cn/flagos-dev/flagcx-metax-maca3.8.1.3:2.3.0 \
  docker/metax-ci
```

Hardware validation requires a MetaX node with driver/device access, active
RDMA ports and sufficient shared memory. For Kubernetes, the job Pod template
must supply those resources, the driver mount at `/opt/mxdriver`, host network
and IPC as required by the current CI configuration, and memlock permissions.
The image does not allocate accelerators itself.

Run the following inside the hardware test environment, in the repository
workspace with its pinned submodules initialized:

```bash
export GITHUB_WORKSPACE="$PWD"
export FLAGCX_CI_LOG_DIR="$PWD/.ci-logs"
export FLAGCX_CI_PLATFORM=metax
export FLAGCX_CI_SET_ENV=.github/scripts/set_env/metax.sh
python3 .github/scripts/ci/check_metax_devices.py
mpirun --allow-run-as-root -np 4 hostname
for suite in adaptor core p2p rma runner service symmem; do
  bash .github/scripts/ci/run_unit_test.sh "$FLAGCX_CI_SET_ENV" "$suite" || exit
done
FLAGCX_CI_TORCH_COVERAGE=1 \
  bash .github/scripts/ci/run_metax_workload.sh torch-api
```

The Torch workload checks native MACA device count and the actual Torch
interpreter/package, initializes Torch, and verifies tensor read/write on
`cuda:0` through `cuda:3` before building. Both four-rank collective modes must
complete without `[SKIP]`; each rank must reach the last collective in both
modes. Ordinary local use of the shared Torch example keeps its skip behavior.
