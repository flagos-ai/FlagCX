# FlagCX Device Adaptor Plugin Documentation

This page describes the FlagCX Device Adaptor plugin API and how to implement a device adaptor plugin for FlagCX.

## Overview

FlagCX supports external device adaptor plugins to allow custom device runtime implementations without modifying the FlagCX source tree. Plugins implement the FlagCX device adaptor API as a shared library (`.so`), which FlagCX loads at runtime via `dlopen`.

When a plugin is loaded, it replaces the built-in device adaptor (`deviceAdaptor`) and rebuilds the cached `globalDeviceHandle` so that all subsequent device operations use the plugin's implementations.

## Plugin Architecture

### Loading

FlagCX looks for a plugin when the `FLAGCX_DEVICE_ADAPTOR_PLUGIN` environment variable is set. The value can be:

- An absolute or relative path to a `.so` file (e.g. `./libflagcx-device-myplugin.so`)
- `none` to explicitly disable plugin loading

If the variable is unset, no plugin is loaded and the built-in adaptor is used.

### Symbol Versioning

Once the library is loaded, FlagCX looks for a symbol named `flagcxDeviceAdaptorPlugin_v1`. This versioned naming allows future API changes while maintaining backwards compatibility.

The symbol must be a `struct flagcxDeviceAdaptor` instance with `visibility("default")` so that `dlsym` can find it.

### Lifecycle

The device adaptor plugin is initialized during `flagcxHandleInit()` (before the device handle is copied) and finalized during `flagcxHandleFree()` (after freeing the device handle). A reference count ensures the plugin stays loaded when multiple handles exist.

## Building a Plugin

### Headers

Plugins should copy the required FlagCX headers into their own source tree to avoid build-time dependency on the full FlagCX source. The example plugin demonstrates this pattern with a local `flagcx/` directory containing:

- `flagcx.h` — Core types and error codes
- `flagcx_device_adaptor.h` — The `flagcxDeviceAdaptor` struct and plugin symbol macro
- **Platform adaptor header** — Copy the vendor adaptor header corresponding to your target platform from `flagcx/adaptor/include/`. For example, `nvidia_adaptor.h` for NVIDIA/CUDA. This header provides struct definitions for `flagcxStream`, `flagcxEvent`, `flagcxIpcMemHandle`, `flagcxWindow`, etc. Note: `flagcxDevProps` is defined in `flagcx_device_adaptor.h`, not in the vendor platform header — do not rely on the vendor header for `flagcxDevProps`.

When copying the vendor adaptor header, **remove the `#ifdef USE_XXX_ADAPTOR` / `#endif` guard**. Since your plugin targets a specific platform, the platform choice is implicit — adding the guard would require an unnecessary `-DUSE_XXX_ADAPTOR` flag in your Makefile. See `example/flagcx/nvidia_adaptor.h` and `cuda/flagcx/nvidia_adaptor.h` for reference.

### Compilation

Plugins must be compiled as shared libraries with `-fPIC`. Using `-fvisibility=hidden` is recommended to avoid exporting internal symbols, with only the plugin symbol marked visible:

```c
__attribute__((visibility("default")))
struct flagcxDeviceAdaptor FLAGCX_DEVICE_ADAPTOR_PLUGIN_SYMBOL_V1 = {
    "MyPlugin",
    myDeviceSynchronize, myDeviceMemcpy, myDeviceMemset,
    ...
};
```

A minimal Makefile:

```makefile
build: libflagcx-device-myplugin.so

libflagcx-device-myplugin.so: plugin.cc
	g++ -Iflagcx -fPIC -shared -o $@ $^

clean:
	rm -f libflagcx-device-myplugin.so
```

## API (v1)

Below is the `flagcxDeviceAdaptor` struct with all members (1 name + function pointers).

```c
struct flagcxDeviceAdaptor {
  char name[32];

  // Basic functions
  flagcxResult_t (*deviceSynchronize)();
  flagcxResult_t (*deviceMemcpy)(void *dst, void *src, size_t size,
                                 flagcxMemcpyType_t type, flagcxStream_t stream,
                                 void *args);
  flagcxResult_t (*deviceMemset)(void *ptr, int value, size_t size,
                                 flagcxMemType_t type, flagcxStream_t stream);
  flagcxResult_t (*deviceMalloc)(void **ptr, size_t size, flagcxMemType_t type,
                                 flagcxStream_t stream);
  flagcxResult_t (*deviceFree)(void *ptr, flagcxMemType_t type,
                               flagcxStream_t stream);
  flagcxResult_t (*setDevice)(int dev);
  flagcxResult_t (*getDevice)(int *dev);
  flagcxResult_t (*getDeviceCount)(int *count);
  flagcxResult_t (*getVendor)(char *vendor);
  flagcxResult_t (*hostGetDevicePointer)(void **pDevice, void *pHost);

  // GDR functions
  flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
  flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
  flagcxResult_t (*gdrMemAlloc)(void **ptr, size_t size, void *memHandle);
  flagcxResult_t (*gdrMemFree)(void *ptr, void *memHandle);
  flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void *memHandle);
  flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
  flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t sz);
  flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);

  // Stream functions
  flagcxResult_t (*streamCreate)(flagcxStream_t *stream);
  flagcxResult_t (*streamDestroy)(flagcxStream_t stream);
  flagcxResult_t (*streamCopy)(flagcxStream_t *newStream, void *oldStream);
  flagcxResult_t (*streamFree)(flagcxStream_t stream);
  flagcxResult_t (*streamSynchronize)(flagcxStream_t stream);
  flagcxResult_t (*streamQuery)(flagcxStream_t stream);
  flagcxResult_t (*streamWaitEvent)(flagcxStream_t stream, flagcxEvent_t event);
  flagcxResult_t (*streamWaitValue64)(flagcxStream_t stream, void *addr,
                                      uint64_t value, int flags);
  flagcxResult_t (*streamWriteValue64)(flagcxStream_t stream, void *addr,
                                       uint64_t value, int flags);

  // Event functions
  flagcxResult_t (*eventCreate)(flagcxEvent_t *event,
                                flagcxEventType_t eventType);
  flagcxResult_t (*eventDestroy)(flagcxEvent_t event);
  flagcxResult_t (*eventRecord)(flagcxEvent_t event, flagcxStream_t stream);
  flagcxResult_t (*eventSynchronize)(flagcxEvent_t event);
  flagcxResult_t (*eventQuery)(flagcxEvent_t event);
  flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t start,
                                     flagcxEvent_t end);

  // IpcMemHandle functions
  flagcxResult_t (*ipcMemHandleCreate)(flagcxIpcMemHandle_t *handle,
                                       size_t *size);
  flagcxResult_t (*ipcMemHandleGet)(flagcxIpcMemHandle_t handle, void *devPtr);
  flagcxResult_t (*ipcMemHandleOpen)(flagcxIpcMemHandle_t handle,
                                     void **devPtr);
  flagcxResult_t (*ipcMemHandleClose)(void *devPtr);
  flagcxResult_t (*ipcMemHandleFree)(flagcxIpcMemHandle_t handle);

  // Kernel launch
  flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
                                 unsigned int block_y, unsigned int block_z,
                                 unsigned int grid_x, unsigned int grid_y,
                                 unsigned int grid_z, void **args,
                                 size_t share_mem, void *stream,
                                 void *memHandle);
  flagcxResult_t (*copyArgsInit)(void **args);
  flagcxResult_t (*copyArgsFree)(void *args);
  flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream,
                                     flagcxLaunchFunc_t fn, void *args);

  // Others
  flagcxResult_t (*getDeviceProperties)(struct flagcxDevProps *props, int dev);
  flagcxResult_t (*getDevicePciBusId)(char *pciBusId, int len, int dev);
  flagcxResult_t (*getDeviceByPciBusId)(int *dev, const char *pciBusId);

  // HostFunc launch
  flagcxResult_t (*launchHostFunc)(flagcxStream_t stream, void (*fn)(void *),
                                   void *args);

  // DMA buffer
  flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
  flagcxResult_t (*getHandleForAddressRange)(void *handleOut, void *buffer,
                                             size_t size,
                                             unsigned long long flags);
};
```

### Validation

When loading a plugin, FlagCX validates that `name` is non-empty and the function pointers that all built-in adaptors implement are non-NULL:
- `name[0] != '\0'`
- Basic: `deviceSynchronize`, `deviceMemcpy`, `deviceMemset`, `deviceMalloc`, `deviceFree`, `setDevice`, `getDevice`, `getDeviceCount`, `getVendor`
- GDR: `gdrMemAlloc`, `gdrMemFree`
- Stream: `streamCreate`, `streamDestroy`, `streamCopy`, `streamFree`, `streamSynchronize`, `streamQuery`, `streamWaitEvent`, `streamWaitValue64`, `streamWriteValue64`
- Event: `eventCreate`, `eventDestroy`, `eventRecord`, `eventSynchronize`, `eventQuery`, `eventElapsedTime`
- IPC: `ipcMemHandleCreate`, `ipcMemHandleGet`, `ipcMemHandleOpen`, `ipcMemHandleClose`, `ipcMemHandleFree`
- Other: `launchHostFunc`

The following fields are **not** validated because some built-in adaptors leave them NULL: `hostGetDevicePointer`, `memHandleInit`, `memHandleDestroy`, `hostShareMemAlloc`, `hostShareMemFree`, `gdrPtrMmap`, `gdrPtrMunmap`, `launchKernel`, `copyArgsInit`, `copyArgsFree`, `launchDeviceFunc`, `getDeviceProperties`, `getDevicePciBusId`, `getDeviceByPciBusId`, `dmaSupport`, `getHandleForAddressRange`.

If any required field is missing, the plugin is not loaded and FlagCX falls back to the built-in adaptor.

### Error Codes

All plugin functions return `flagcxResult_t`. Return `flagcxSuccess` on success.

- `flagcxSuccess` — Operation completed successfully.
- `flagcxUnhandledDeviceError` — A device runtime call failed.
- `flagcxSystemError` — A system call failed.
- `flagcxInternalError` — An internal logic error or unsupported operation.

## Examples

### Example Plugin (Skeleton)

The `example/` directory contains a minimal skeleton plugin where all operations return `flagcxInternalError`. It demonstrates the required file structure, headers, and export symbol.

### CUDA Plugin

The `cuda/` directory contains a real plugin wrapping CUDA runtime APIs. It can be used as a reference for implementing device plugins for other platforms.

### Build and Test

```bash
# Build the example plugin (no dependencies)
cd adaptor_plugin/device/example
make

# Run with the plugin (plugin loads but operations will fail)
FLAGCX_DEVICE_ADAPTOR_PLUGIN=./adaptor_plugin/device/example/libflagcx-device-example.so \
  FLAGCX_DEBUG=INFO <your_app>

# Expect log output:
#   ADAPTOR/Plugin: Loaded device adaptor plugin 'Example'

# Build the CUDA plugin
cd adaptor_plugin/device/cuda
CUDA_HOME=/usr/local/cuda make

# Run with CUDA plugin
FLAGCX_DEVICE_ADAPTOR_PLUGIN=./adaptor_plugin/device/cuda/libflagcx-device-cuda.so \
  FLAGCX_DEBUG=INFO <your_app>

# Disable plugin
FLAGCX_DEVICE_ADAPTOR_PLUGIN=none <your_app>

# Test with bad path (warning logged, fallback to default)
FLAGCX_DEVICE_ADAPTOR_PLUGIN=/nonexistent.so FLAGCX_DEBUG=INFO <your_app>
```
