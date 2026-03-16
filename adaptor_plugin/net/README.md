# FlagCX Net Adaptor Plugin Documentation

This page describes the FlagCX Net Adaptor plugin API and how to implement a network plugin for FlagCX.

## Overview

FlagCX supports external network plugins to allow custom network implementations without modifying the FlagCX source tree. Plugins implement the FlagCX net adaptor API as a shared library (`.so`), which FlagCX loads at runtime via `dlopen`.

When a plugin is loaded, it takes priority over the built-in network adaptors (IBRC and Socket). If the plugin reports zero devices or its `init` fails, FlagCX falls back to the built-in adaptors.

## Plugin Architecture

### Loading

FlagCX looks for a plugin when the `FLAGCX_NET_ADAPTOR_PLUGIN` environment variable is set. The value can be:

- An absolute or relative path to a `.so` file (e.g. `./libflagcx-net-myplugin.so`)
- `none` to explicitly disable plugin loading

If the variable is unset, no plugin is loaded and the built-in adaptors are used.

### Symbol Versioning

Once the library is loaded, FlagCX looks for a symbol named `flagcxNetAdaptorPlugin_v1`. This versioned naming allows future API changes while maintaining backwards compatibility.

The symbol must be a `struct flagcxNetAdaptor` instance with `visibility("default")` so that `dlsym` can find it.

### Adaptor Slot Priority

FlagCX maintains an array of net adaptors. A loaded plugin is placed in slot 0, giving it highest priority during device selection. Built-in adaptors (IBRC, Socket) occupy subsequent slots.

## Building a Plugin

### Headers

Plugins should copy the required FlagCX headers into their own source tree to avoid build-time dependency on the full FlagCX source. The example plugin demonstrates this pattern with a local `flagcx/` directory containing:

- `flagcx.h` — Core types and error codes
- `flagcx_net_adaptor.h` — The `flagcxNetAdaptor` struct and plugin symbol macro

### Compilation

Plugins must be compiled as shared libraries with `-fPIC`. Using `-fvisibility=hidden` is recommended to avoid exporting internal symbols, with only the plugin symbol marked visible:

```c
__attribute__((visibility("default")))
struct flagcxNetAdaptor FLAGCX_NET_ADAPTOR_PLUGIN_SYMBOL = {
    "MyPlugin",
    myInit, myDevices, myGetProperties,
    ...
};
```

A minimal Makefile:

```makefile
build: libflagcx-net-myplugin.so

libflagcx-net-myplugin.so: plugin.cc
	g++ -Iflagcx -fPIC -shared -o $@ $^

clean:
	rm -f libflagcx-net-myplugin.so
```

## API (v1)

Below is the `flagcxNetAdaptor` struct. Each function pointer is explained in later sections.

```c
struct flagcxNetAdaptor {
  const char *name;
  flagcxResult_t (*init)();
  flagcxResult_t (*devices)(int *ndev);
  flagcxResult_t (*getProperties)(int dev, void *props);
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  flagcxResult_t (*getDeviceMr)(void *comm, void *mhandle, void **dptr_mhandle);
  flagcxResult_t (*irecvConsumed)(void *recvComm, int n, void *request);

  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  flagcxResult_t (*connect)(int dev, void *handle, void **sendComm);
  flagcxResult_t (*accept)(void *listenComm, void **recvComm);
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  flagcxResult_t (*regMr)(void *comm, void *data, size_t size, int type,
                          void **mhandle);
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);

  flagcxResult_t (*isend)(void *sendComm, void *data, size_t size, int tag,
                          void *mhandle, void *phandle, void **request);
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, size_t *sizes,
                          int *tags, void **mhandles, void **phandles,
                          void **request);
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  flagcxResult_t (*test)(void *request, int *done, int *sizes);

  flagcxResult_t (*put)(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                        size_t size, int srcRank, int dstRank, void **gHandles,
                        void **request);
  flagcxResult_t (*putSignal)(void *sendComm, uint64_t dstOff, int dstRank,
                              void **gHandles, void **request);
  flagcxResult_t (*waitValue)(void **gHandles, int rank, uint64_t offset,
                              uint64_t expected);

  flagcxResult_t (*getDevFromName)(char *name, int *dev);
};
```

### Error Codes

All plugin functions return `flagcxResult_t`. Return `flagcxSuccess` on success.

- `flagcxSuccess` — Operation completed successfully.
- `flagcxSystemError` — A system or hardware call failed (e.g. network errors).
- `flagcxInternalError` — An internal logic error or unsupported operation.

### Initialization

`name`

A string identifying the plugin, used in log messages (e.g. when `FLAGCX_DEBUG=INFO`).

`init`

Called once after the plugin is loaded. The plugin should discover and initialize network devices. If `init` does not return `flagcxSuccess`, FlagCX will not use the plugin and falls back to built-in adaptors.

`devices`

Returns the number of available network devices in `*ndev`. If zero, FlagCX skips the plugin and uses built-in adaptors.

`getProperties`

Returns properties for device `dev` as a `flagcxNetProperties_t`. Key fields include:
- `name` — Device name for logging.
- `pciPath` — PCI device path in `/sys` for topology detection.
- `guid` — Unique identifier; devices sharing a GUID are considered the same physical port.
- `speed` — Port speed in Mbps.
- `ptrSupport` — Bitmask of `FLAGCX_PTR_HOST`, `FLAGCX_PTR_CUDA`, `FLAGCX_PTR_DMABUF`.
- `maxComms` — Maximum number of connections.
- `maxRecvs` — Maximum grouped receive count.

### Connection Establishment

Connections are unidirectional with a sender side and a receiver side.

`listen`

Called on the receiver side. Takes a device index `dev`, returns a `listenComm` object and fills `handle` (an opaque buffer passed to the sender via bootstrap).

`connect`

Called on the sender side with the `handle` from `listen`. Should not block — if the connection is not yet ready, set `*sendComm = NULL` and return `flagcxSuccess`. FlagCX will call `connect` again.

`accept`

Called on the receiver side to finalize the connection. Like `connect`, should not block — set `*recvComm = NULL` if not ready yet.

`closeSend` / `closeRecv` / `closeListen`

Free resources associated with send, receive, or listen comm objects.

### Memory Registration

`regMr`

Register a buffer for communication. The `comm` argument can be either a sendComm or recvComm. `type` indicates `FLAGCX_PTR_HOST` or `FLAGCX_PTR_CUDA`. The returned `mhandle` is passed to subsequent send/recv calls.

`regMrDmaBuf`

Like `regMr` but for DMA-BUF backed memory. Only needed if `ptrSupport` includes `FLAGCX_PTR_DMABUF`.

`deregMr`

Deregister a previously registered buffer.

### Two-Sided Communication

`isend`

Initiate an asynchronous send. Returns a `request` handle for use with `test`. If the operation cannot be initiated, set `*request = NULL` — FlagCX will retry later.

`irecv`

Initiate an asynchronous receive. Supports grouped receives (`n > 1`) for aggregation. Tags distinguish individual sends within a grouped receive. Returns a single `request` handle covering all `n` receives.

`iflush`

After a receive targeting GPU memory completes, FlagCX calls `iflush` to ensure data visibility to the GPU. Returns a `request` to be polled with `test`.

`test`

Poll a request for completion. Set `*done = 1` when complete, with `*sizes` indicating actual bytes transferred.

### One-Sided Communication

`put`

Initiate an RDMA write from `srcOff` to `dstOff` using global handles.

`putSignal`

Send a signal (atomic increment) to a remote address.

`waitValue`

Busy-wait until the value at a given offset matches `expected`.

### Device Name Lookup

`getDevFromName`

Resolve a device name string to a device index.

### Optional Functions

The following function pointers may be set to `NULL` if not supported:
- `reduceSupport`
- `getDeviceMr`
- `irecvConsumed`

## Example

The `example/` directory contains a minimal skeleton plugin that reports 0 devices, causing FlagCX to fall back to built-in adaptors. It demonstrates the required file structure, headers, and export symbol.

### Build and Test

```bash
# Build the example plugin
cd adaptor_plugin/net/example
make

# Run with the plugin
FLAGCX_NET_ADAPTOR_PLUGIN=./adaptor_plugin/net/example/libflagcx-net-example.so <your_app>

# Expect log output:
#   ADAPTOR/Plugin: Loaded net adaptor plugin 'Example'
# The plugin reports 0 devices, so FlagCX falls back to built-in IBRC/Socket.

# Disable plugin
FLAGCX_NET_ADAPTOR_PLUGIN=none <your_app>
```
