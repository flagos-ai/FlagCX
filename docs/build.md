## Build Guide

1. Clone the repository:

   ```shell
   git clone https://github.com/FlagOpen/FlagCX.git
   ```

1. Initialize third-party dependencies:

   ```shell
   git submodule update --init --recursive
   ```

1. Build the library with different flags targeting to different platforms:

   ```shell
   cd FlagCX
   make [backend]
   ```

   where `backend` is used to enable different platforms. The valid options are
   one of:

   - `USE_NVIDIA=1`: Enable the NVIDIA support;
   - `USE_ILUVATAR_COREX=1`: Enable the Iluvatar Corex support;
   - `USE_CAMBRICON=1`: Enable the support to Cambricon chips;
   - `USE_GLOO=1`: Enable the support to GLOO host-side CCL;
   - `USE_MPI=1`: Enable the support to MPI library;
   - `USE_METAX=1`: Enable the support to MetaX chips;
   - `USE_MUSA=1`: Enable the support to Moore Threads chips;
   - `USE_KUNLUNXIN=1`: Enable the support to Kunlunxin chips;
   - `USE_DU=1`, `USE_ASCEND=1`: Enable support to Huawei Ascend hardware;
   - `USE_AMD=1`: Enable support to AMD hardware;
   - `USE_TOPS=1`: Enable support to Enflame TOPS hardware.

   The default installation path is set to `build/`, you can manually set `BUILDDIR` environment variable to customize the build path.
   You may also specify `DEVICE_HOME` and/or `CCL_HOME` to indicate the installation paths of the device runtime and installation path
   of the communication libraries respectively.

