# FlagCX CCL Adaptor Plugin Documentation

This page describes the FlagCX CCL Adaptor plugin API and how to implement a device-side CCL plugin for FlagCX.

## Overview

FlagCX supports external CCL (Collective Communication Library) plugins to allow custom device-side collective implementations without modifying the FlagCX source tree. Plugins implement the FlagCX CCL adaptor API as a shared library (`.so`), which FlagCX loads at runtime via `dlopen`.

When a plugin is loaded, it replaces the built-in device-side CCL adaptor (index 1 in the `cclAdaptors` array). The host-side adaptor (bootstrap/gloo/MPI at index 0) is never affected by the plugin.

## Plugin Architecture

### Loading

FlagCX looks for a plugin when the `FLAGCX_CCL_ADAPTOR_PLUGIN` environment variable is set. The value can be:

- An absolute or relative path to a `.so` file (e.g. `./libflagcx-ccl-myplugin.so`)
- `none` to explicitly disable plugin loading

If the variable is unset, no plugin is loaded and the built-in adaptor is used.

### Symbol Versioning

Once the library is loaded, FlagCX looks for a symbol named `flagcxCCLAdaptorPlugin_v1`. This versioned naming allows future API changes while maintaining backwards compatibility.

The symbol must be a `struct flagcxCCLAdaptor_v1` instance with `visibility("default")` so that `dlsym` can find it.

### Lifecycle

The CCL adaptor plugin is initialized during `flagcxCommInitRank()` (before any device-side CCL calls) and finalized during `flagcxCommDestroy()` (after all device-side communicators are destroyed). A reference count ensures the plugin stays loaded when multiple communicators exist.

## Building a Plugin

### Headers

Plugins should copy the required FlagCX headers into their own source tree to avoid build-time dependency on the full FlagCX source. The example plugin demonstrates this pattern with a local `flagcx/` directory containing:

- `flagcx.h` — Core types and error codes
- `flagcx_ccl_adaptor.h` — The `flagcxCCLAdaptor_v1` struct and plugin symbol macro
- **Platform adaptor header** — Copy the vendor adaptor header corresponding to your target platform from `flagcx/adaptor/include/`. For example, `nvidia_adaptor.h` for NVIDIA/NCCL. This header provides struct definitions for `flagcxInnerComm`, `flagcxStream`, `flagcxEvent`, `flagcxIpcMemHandle`, `flagcxWindow`, etc.

When copying the vendor adaptor header, **remove the `#ifdef USE_XXX_ADAPTOR` / `#endif` guard**. Since your plugin targets a specific platform, the platform choice is implicit — adding the guard would require an unnecessary `-DUSE_XXX_ADAPTOR` flag in your Makefile. See `example/flagcx/nvidia_adaptor.h` and `nccl/flagcx/nvidia_adaptor.h` for reference.

### Compilation

Plugins must be compiled as shared libraries with `-fPIC`. Using `-fvisibility=hidden` is recommended to avoid exporting internal symbols, with only the plugin symbol marked visible:

```c
__attribute__((visibility("default")))
struct flagcxCCLAdaptor_v1 FLAGCX_CCL_ADAPTOR_PLUGIN_SYMBOL_V1 = {
    "MyPlugin",
    myGetVersion, myGetUniqueId, myGetErrorString,
    ...
};
```

A minimal Makefile:

```makefile
build: libflagcx-ccl-myplugin.so

libflagcx-ccl-myplugin.so: plugin.cc
	g++ -Iflagcx -fPIC -shared -o $@ $^

clean:
	rm -f libflagcx-ccl-myplugin.so
```

## API (v1)

Below is the `flagcxCCLAdaptor_v1` struct with all 35 members (1 name + 34 function pointers).

```c
struct flagcxCCLAdaptor_v1 {
  const char *name;

  // Basic functions
  flagcxResult_t (*getVersion)(int *version);
  flagcxResult_t (*getUniqueId)(flagcxUniqueId_t *uniqueId);
  const char *(*getErrorString)(flagcxResult_t result);
  const char *(*getLastError)(flagcxInnerComm_t comm);
  flagcxResult_t (*getStagedBuffer)(const flagcxInnerComm_t comm, void **buff,
                                    size_t size, int isRecv);

  // Communicator functions
  flagcxResult_t (*commInitRank)(flagcxInnerComm_t *comm, int nranks,
                                 flagcxUniqueId *commId, int rank,
                                 bootstrapState *bootstrap);
  flagcxResult_t (*commFinalize)(flagcxInnerComm_t comm);
  flagcxResult_t (*commDestroy)(flagcxInnerComm_t comm);
  flagcxResult_t (*commAbort)(flagcxInnerComm_t comm);
  flagcxResult_t (*commResume)(flagcxInnerComm_t comm);
  flagcxResult_t (*commSuspend)(flagcxInnerComm_t comm);
  flagcxResult_t (*commCount)(const flagcxInnerComm_t comm, int *count);
  flagcxResult_t (*commGetDeviceNumber)(const flagcxInnerComm_t comm,
                                        int *device);
  flagcxResult_t (*commUserRank)(const flagcxInnerComm_t comm, int *rank);
  flagcxResult_t (*commGetAsyncError)(flagcxInnerComm_t comm,
                                      flagcxResult_t *asyncError);
  flagcxResult_t (*memAlloc)(void **ptr, size_t size);
  flagcxResult_t (*memFree)(void *ptr);
  flagcxResult_t (*commRegister)(const flagcxInnerComm_t comm, void *buff,
                                 size_t size, void **handle);
  flagcxResult_t (*commDeregister)(const flagcxInnerComm_t comm, void *handle);

  // Symmetric functions
  flagcxResult_t (*commWindowRegister)(flagcxInnerComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags);
  flagcxResult_t (*commWindowDeregister)(flagcxInnerComm_t comm,
                                         flagcxWindow_t win);

  // Communication functions
  flagcxResult_t (*reduce)(const void *sendbuff, void *recvbuff, size_t count,
                           flagcxDataType_t datatype, flagcxRedOp_t op,
                           int root, flagcxInnerComm_t comm,
                           flagcxStream_t stream);
  flagcxResult_t (*gather)(const void *sendbuff, void *recvbuff, size_t count,
                           flagcxDataType_t datatype, int root,
                           flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*scatter)(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, int root,
                            flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*broadcast)(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype, int root,
                              flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*allReduce)(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype,
                              flagcxRedOp_t op, flagcxInnerComm_t comm,
                              flagcxStream_t stream);
  flagcxResult_t (*reduceScatter)(const void *sendbuff, void *recvbuff,
                                  size_t recvcount, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, flagcxInnerComm_t comm,
                                  flagcxStream_t stream);
  flagcxResult_t (*allGather)(const void *sendbuff, void *recvbuff,
                              size_t sendcount, flagcxDataType_t datatype,
                              flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*alltoAll)(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, flagcxInnerComm_t comm,
                             flagcxStream_t stream);
  flagcxResult_t (*alltoAllv)(const void *sendbuff, size_t *sendcounts,
                              size_t *sdispls, void *recvbuff,
                              size_t *recvcounts, size_t *rdispls,
                              flagcxDataType_t datatype, flagcxInnerComm_t comm,
                              flagcxStream_t stream);
  flagcxResult_t (*send)(const void *sendbuff, size_t count,
                         flagcxDataType_t datatype, int peer,
                         flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*recv)(void *recvbuff, size_t count,
                         flagcxDataType_t datatype, int peer,
                         flagcxInnerComm_t comm, flagcxStream_t stream);

  // Group semantics
  flagcxResult_t (*groupStart)();
  flagcxResult_t (*groupEnd)();
};
```

### Validation

When loading a plugin, FlagCX validates that all 34 function pointers (and `name`) are non-NULL:
- `name`
- `getVersion`, `getUniqueId`, `getErrorString`, `getLastError`, `getStagedBuffer`
- `commInitRank`, `commFinalize`, `commDestroy`, `commAbort`, `commResume`, `commSuspend`
- `commCount`, `commGetDeviceNumber`, `commUserRank`, `commGetAsyncError`
- `memAlloc`, `memFree`, `commRegister`, `commDeregister`
- `commWindowRegister`, `commWindowDeregister`
- `reduce`, `gather`, `scatter`, `broadcast`, `allReduce`, `reduceScatter`, `allGather`
- `alltoAll`, `alltoAllv`, `send`, `recv`
- `groupStart`, `groupEnd`

If any field is NULL, the plugin is not loaded and FlagCX falls back to the built-in adaptor. Functions that your platform does not support should be implemented as stubs returning `flagcxInternalError` or `flagcxNotSupported`.

### Error Codes

All plugin functions return `flagcxResult_t`. Return `flagcxSuccess` on success.

- `flagcxSuccess` — Operation completed successfully.
- `flagcxSystemError` — A system or hardware call failed.
- `flagcxInternalError` — An internal logic error or unsupported operation.

## Example

The `example/` directory contains a minimal skeleton plugin where all operations return `flagcxInternalError`. It demonstrates the required file structure, headers, and export symbol.

### Build and Test

```bash
# Build the example plugin
cd adaptor_plugin/ccl/example
make

# Run with the plugin (plugin loads but operations will fail)
FLAGCX_CCL_ADAPTOR_PLUGIN=./adaptor_plugin/ccl/example/libflagcx-ccl-example.so \
  FLAGCX_DEBUG=INFO <your_app>

# Expect log output:
#   ADAPTOR/Plugin: Loaded CCL adaptor plugin 'Example'

# Disable plugin
FLAGCX_CCL_ADAPTOR_PLUGIN=none <your_app>
```
