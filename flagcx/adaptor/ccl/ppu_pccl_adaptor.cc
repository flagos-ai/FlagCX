#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "comm.h"
#include "dlsymbols.h"
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define PPU_PCCL_SYMBOLS(M)                                                    \
  M(pcclGetVersion, ppupcclGetVersionFn)                                       \
  M(pcclGetUniqueId, ppupcclGetUniqueIdFn)                                     \
  M(pcclGetErrorString, ppupcclGetErrorStringFn)                               \
  M(pcclGetLastError, ppupcclGetLastErrorFn)                                   \
  M(pcclCommInitRank, ppupcclCommInitRankFn)                                   \
  M(pcclCommFinalize, ppupcclCommFinalizeFn)                                   \
  M(pcclCommDestroy, ppupcclCommDestroyFn)                                     \
  M(pcclCommAbort, ppupcclCommAbortFn)                                         \
  M(pcclCommCount, ppupcclCommCountFn)                                         \
  M(pcclCommCuDevice, ppupcclCommCuDeviceFn)                                   \
  M(pcclCommUserRank, ppupcclCommUserRankFn)                                   \
  M(pcclCommGetAsyncError, ppupcclCommGetAsyncErrorFn)                         \
  M(pcclReduce, ppupcclReduceFn)                                               \
  M(pcclBroadcast, ppupcclBroadcastFn)                                         \
  M(pcclAllReduce, ppupcclAllReduceFn)                                         \
  M(pcclReduceScatter, ppupcclReduceScatterFn)                                 \
  M(pcclAllGather, ppupcclAllGatherFn)                                         \
  M(pcclSend, ppupcclSendFn)                                                   \
  M(pcclRecv, ppupcclRecvFn)                                                   \
  M(pcclGroupStart, ppupcclGroupStartFn)                                       \
  M(pcclGroupEnd, ppupcclGroupEndFn)

typedef pcclResult_t (*ppupcclGetVersionFn)(int *);
typedef pcclResult_t (*ppupcclGetUniqueIdFn)(pcclUniqueId *);
typedef pcclResult_t (*ppupcclCommInitRankFn)(pcclComm_t *, int, pcclUniqueId,
                                              int);
typedef pcclResult_t (*ppupcclCommFinalizeFn)(pcclComm_t);
typedef pcclResult_t (*ppupcclCommDestroyFn)(pcclComm_t);
typedef pcclResult_t (*ppupcclCommAbortFn)(pcclComm_t);
typedef pcclResult_t (*ppupcclCommCountFn)(pcclComm_t, int *);
typedef pcclResult_t (*ppupcclCommCuDeviceFn)(pcclComm_t, int *);
typedef pcclResult_t (*ppupcclCommUserRankFn)(pcclComm_t, int *);
typedef pcclResult_t (*ppupcclReduceFn)(const void *, void *, size_t,
                                        pcclDataType_t, pcclRedOp_t, int,
                                        pcclComm_t, hggcStream_t);
typedef pcclResult_t (*ppupcclBroadcastFn)(const void *, void *, size_t,
                                           pcclDataType_t, int, pcclComm_t,
                                           hggcStream_t);
typedef pcclResult_t (*ppupcclReduceScatterFn)(const void *, void *, size_t,
                                               pcclDataType_t, pcclRedOp_t,
                                               pcclComm_t, hggcStream_t);
typedef pcclResult_t (*ppupcclAllGatherFn)(const void *, void *, size_t,
                                           pcclDataType_t, pcclComm_t,
                                           hggcStream_t);
typedef pcclResult_t (*ppupcclSendFn)(const void *, size_t, pcclDataType_t, int,
                                      pcclComm_t, hggcStream_t);
typedef pcclResult_t (*ppupcclRecvFn)(void *, size_t, pcclDataType_t, int,
                                      pcclComm_t, hggcStream_t);
typedef pcclResult_t (*ppupcclGroupStartFn)(void);
typedef pcclResult_t (*ppupcclGroupEndFn)(void);
typedef const char *(*ppupcclGetErrorStringFn)(pcclResult_t);
typedef const char *(*ppupcclGetLastErrorFn)(pcclComm_t);
typedef pcclResult_t (*ppupcclCommGetAsyncErrorFn)(pcclComm_t, pcclResult_t *);
typedef pcclResult_t (*ppupcclAllReduceFn)(const void *, void *, size_t,
                                           pcclDataType_t, pcclRedOp_t,
                                           pcclComm_t, hggcStream_t);

struct ppupcclSymbols {
#define PPU_PCCL_DECLARE(name, type) type name;
  PPU_PCCL_SYMBOLS(PPU_PCCL_DECLARE)
#undef PPU_PCCL_DECLARE
};

static pthread_once_t ppupcclApiOnce = PTHREAD_ONCE_INIT;
static struct ppupcclSymbols ppupcclApi;
static flagcxResult_t ppupcclApiResult = flagcxSystemError;
static void *ppupcclLibraryHandle;
static pthread_once_t ppupcclPluginOnce = PTHREAD_ONCE_INIT;
static void *ppupcclPluginHandle;

static void ppupcclLoadError(const char *path, int flags, const char *message) {
  INFO(FLAGCX_INIT, "PPU/PCCL: cannot load %s: %s", path, message);
}

static void ppupcclIgnoreLoadError(const char *path, int flags,
                                   const char *message) {}

static int ppupcclLoadFlags() {
  int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
  flags |= RTLD_DEEPBIND;
#endif
  return flags;
}

static void ppupcclInitApi(void) {
  const char *enabled = getenv("PCCL_ENABLE_EXT_KERNEL");
  const char *paths[] = {"libpccl.so.2", "libpccl.so"};
  struct ppupcclSymbols loaded;
  memset(&loaded, 0, sizeof(loaded));
  int flags = ppupcclLoadFlags();
  // PCCL can cache this setting in its library initializer.
  if (enabled == NULL)
    enabled = getenv("NCCL_ENABLE_EXT_KERNEL");
  if (enabled == NULL || *enabled == '\0') {
    if (setenv("PCCL_ENABLE_EXT_KERNEL", "0", 1) != 0) {
      WARN("PPU/PCCL: could not disable optional extension kernels");
      return;
    }
  }
  // Use CCL_HOME's rpath and keep PCCL alive for asynchronous work/threads.
  for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
    ppupcclLibraryHandle = flagcxOpenLib(paths[i], flags, ppupcclLoadError);
    if (ppupcclLibraryHandle != NULL)
      break;
  }
  if (ppupcclLibraryHandle == NULL) {
    WARN("PPU/PCCL: failed to load the configured PCCL library");
    return;
  }
#define PPU_PCCL_LOAD(name, type)                                              \
  do {                                                                         \
    dlerror();                                                                 \
    (void)sizeof(1 ? loaded.name : &name);                                     \
    loaded.name = (type)dlsym(ppupcclLibraryHandle, #name);                    \
    const char *error = dlerror();                                             \
    if (error != NULL || loaded.name == NULL) {                                \
      WARN("PPU/PCCL: cannot resolve %s: %s", #name,                           \
           error != NULL ? error : "null symbol");                             \
      return;                                                                  \
    }                                                                          \
  } while (0);
  PPU_PCCL_SYMBOLS(PPU_PCCL_LOAD)
#undef PPU_PCCL_LOAD
  ppupcclApi = loaded;
  ppupcclApiResult = flagcxSuccess;
}

static flagcxResult_t ppupcclLoadApi(const struct ppupcclSymbols **api) {
  if (pthread_once(&ppupcclApiOnce, ppupcclInitApi) != 0)
    return flagcxSystemError;
  *api = ppupcclApiResult == flagcxSuccess ? &ppupcclApi : NULL;
  return ppupcclApiResult;
}

static const char *ppupcclPluginSetting(const char *name,
                                        const char *compatName) {
  const char *value = getenv(name);
  return value != NULL ? value : getenv(compatName);
}

static void *ppupcclOpenKernelPlugin(const char *path) {
#ifdef RTLD_NOLOAD
  void *resident = flagcxOpenLib(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD,
                                 ppupcclIgnoreLoadError);
  if (resident != NULL) {
    INFO(FLAGCX_INIT,
         "PPU/PCCL: kernel plugin %s was already loaded; retaining its "
         "existing symbol bindings",
         path);
    return resident;
  }
#endif
  INFO(FLAGCX_INIT, "PPU/PCCL: preloading kernel plugin %s with flags %#x",
       path, ppupcclLoadFlags());
  return flagcxOpenLib(path, ppupcclLoadFlags(), ppupcclLoadError);
}

static void ppupcclInitKernelPlugin(void) {
  const char *enabled =
      ppupcclPluginSetting("PCCL_ENABLE_EXT_KERNEL", "NCCL_ENABLE_EXT_KERNEL");
  if (enabled != NULL && *enabled != '\0' && strtoll(enabled, NULL, 0) == 0)
    return;
  if (ppupcclPluginSetting("PCCL_ALGO", "NCCL_ALGO") != NULL ||
      ppupcclPluginSetting("PCCL_PROTO", "NCCL_PROTO") != NULL)
    return;
  const char *configured =
      ppupcclPluginSetting("PCCL_EXT_KERNEL_PLUGIN", "NCCL_EXT_KERNEL_PLUGIN");
  if (configured != NULL && strncasecmp(configured, "STATIC_PLUGIN",
                                        sizeof("STATIC_PLUGIN") - 1) == 0)
    return;
  const char *name = configured != NULL && *configured != '\0'
                         ? configured
                         : "libpccl-ext-kernel.so";
  // Load before PCCL's ordinary dlopen so registration and queries bind to
  // the same HGGC runtime. Prefer the selected PCCL installation.
  if (configured == NULL || *configured == '\0') {
    Dl_info info = {};
    if (dladdr((const void *)ppupcclApi.pcclGetVersion, &info) &&
        info.dli_fname != NULL) {
      const char *slash = strrchr(info.dli_fname, '/');
      size_t nameBytes = strlen(name) + 1;
      size_t directoryBytes =
          slash == NULL ? 0 : (size_t)(slash - info.dli_fname) + 1;
      if (directoryBytes != 0 && directoryBytes <= SIZE_MAX - nameBytes) {
        char *path = (char *)malloc(directoryBytes + nameBytes);
        if (path != NULL) {
          memcpy(path, info.dli_fname, directoryBytes);
          memcpy(path + directoryBytes, name, nameBytes);
          ppupcclPluginHandle = ppupcclOpenKernelPlugin(path);
          free(path);
        }
      }
    }
  }
  if (ppupcclPluginHandle == NULL)
    ppupcclPluginHandle = ppupcclOpenKernelPlugin(name);
  if (ppupcclPluginHandle != NULL)
    INFO(FLAGCX_INIT, "PPU/PCCL: retained kernel plugin %s", name);
  // Missing optional plugins let PCCL select built-in kernels. Retain the
  // reference for process lifetime: HGGC keeps registered host entry points.
}

static void ppupcclPreloadKernelPlugin(const struct ppupcclSymbols *api) {
  pthread_once(&ppupcclPluginOnce, ppupcclInitKernelPlugin);
}

#undef PPU_PCCL_SYMBOLS

// Native PCCL enums are not a FlagCX ABI: use explicit conversions, including
// errors introduced by newer SDKs (e.g. timeout) and unknown future values.
static flagcxResult_t ppupcclResultToFlagcx(pcclResult_t result) {
  switch (result) {
    case pcclSuccess:
      return flagcxSuccess;
    case pcclUnhandledHggcError:
      return flagcxUnhandledDeviceError;
    case pcclSystemError:
      return flagcxSystemError;
    case pcclInternalError:
      return flagcxInternalError;
    case pcclInvalidArgument:
      return flagcxInvalidArgument;
    case pcclInvalidUsage:
      return flagcxInvalidUsage;
    case pcclRemoteError:
      return flagcxRemoteError;
    case pcclInProgress:
      return flagcxInProgress;
    default:
      return flagcxUnhandledCCLError;
  }
}

static const char *ppupcclErrorString(pcclResult_t result) {
  const ppupcclSymbols *api = NULL;
  if (ppupcclLoadApi(&api) != flagcxSuccess)
    return "PPU PCCL library could not be loaded; consult the initialization "
           "log";
  return api->pcclGetErrorString(result);
}

static flagcxResult_t ppupcclDataTypeToNative(flagcxDataType_t type,
                                              pcclDataType_t *native) {
  switch (type) {
    case flagcxInt8:
      *native = pcclInt8;
      break;
    case flagcxUint8:
      *native = pcclUint8;
      break;
    case flagcxInt32:
      *native = pcclInt32;
      break;
    case flagcxUint32:
      *native = pcclUint32;
      break;
    case flagcxInt64:
      *native = pcclInt64;
      break;
    case flagcxUint64:
      *native = pcclUint64;
      break;
    case flagcxFloat16:
      *native = pcclFloat16;
      break;
    case flagcxFloat32:
      *native = pcclFloat32;
      break;
    case flagcxFloat64:
      *native = pcclFloat64;
      break;
    case flagcxBfloat16:
      *native = pcclBfloat16;
      break;
    default:
      return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

static flagcxResult_t ppupcclRedOpToNative(flagcxRedOp_t op,
                                           pcclRedOp_t *native) {
  switch (op) {
    case flagcxSum:
      *native = pcclSum;
      break;
    case flagcxProd:
      *native = pcclProd;
      break;
    case flagcxMax:
      *native = pcclMax;
      break;
    case flagcxMin:
      *native = pcclMin;
      break;
    case flagcxAvg:
      *native = pcclAvg;
      break;
    default:
      return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

static_assert(sizeof(pcclUniqueId) <= sizeof(flagcxInnerUniqueId),
              "pcclUniqueId does not fit inside flagcxInnerUniqueId");

flagcxResult_t ppupcclAdaptorGetVersion(int *version) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclGetVersion(version));
}

flagcxResult_t ppupcclAdaptorGetUniqueId(flagcxInnerUniqueId_t *uniqueId) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  if (*uniqueId == NULL) {
    *uniqueId = (flagcxInnerUniqueId_t)calloc(1, sizeof(**uniqueId));
    if (*uniqueId == NULL)
      return flagcxSystemError;
  }
  return ppupcclResultToFlagcx(
      api->pcclGetUniqueId((pcclUniqueId *)(*uniqueId)));
}

flagcxResult_t ppupcclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                             void **buff, size_t size,
                                             int isRecv) {
  return flagcxNotSupported;
}

const char *ppupcclAdaptorGetErrorString(flagcxResult_t result) {
  switch (result) {
    case flagcxSuccess:
      return ppupcclErrorString(pcclSuccess);
    case flagcxUnhandledDeviceError:
      return ppupcclErrorString(pcclUnhandledHggcError);
    case flagcxSystemError:
      return ppupcclErrorString(pcclSystemError);
    case flagcxInternalError:
      return ppupcclErrorString(pcclInternalError);
    case flagcxInvalidArgument:
      return ppupcclErrorString(pcclInvalidArgument);
    case flagcxInvalidUsage:
      return ppupcclErrorString(pcclInvalidUsage);
    case flagcxRemoteError:
      return ppupcclErrorString(pcclRemoteError);
    case flagcxInProgress:
      return ppupcclErrorString(pcclInProgress);
    case flagcxNotSupported:
      return "PPU PCCL adaptor operation is not supported";
    default:
      return "PPU PCCL operation failed; consult pcclGetLastError";
  }
}

const char *ppupcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  const ppupcclSymbols *api = NULL;
  if (ppupcclLoadApi(&api) != flagcxSuccess)
    return "PPU PCCL library could not be loaded; consult the initialization "
           "log";
  return api->pcclGetLastError(comm->base);
}

flagcxResult_t
ppupcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                           flagcxInnerUniqueId_t commId, int rank,
                           struct bootstrapState * /*bootstrap*/) {
  const ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  ppupcclPreloadKernelPlugin(api);
  if (*comm == NULL) {
    *comm = (flagcxInnerComm_t)calloc(1, sizeof(**comm));
    if (*comm == NULL)
      return flagcxSystemError;
  }
  return ppupcclResultToFlagcx(api->pcclCommInitRank(
      &(*comm)->base, nranks, *(pcclUniqueId *)commId, rank));
}

flagcxResult_t ppupcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclCommFinalize(comm->base));
}

flagcxResult_t ppupcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  flagcxResult_t result =
      ppupcclResultToFlagcx(api->pcclCommDestroy(comm->base));
  if (result != flagcxSuccess)
    return result;
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t ppupcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclCommAbort(comm->base));
}

flagcxResult_t ppupcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommCount(const flagcxInnerComm_t comm,
                                       int *count) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclCommCount(comm->base, count));
}

flagcxResult_t ppupcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                          int *device) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclCommCuDevice(comm->base, device));
}

flagcxResult_t ppupcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                          int *rank) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclCommUserRank(comm->base, rank));
}

flagcxResult_t ppupcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                               flagcxResult_t *asyncError) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  if (comm == NULL || asyncError == NULL)
    return flagcxInvalidArgument;
  pcclResult_t nativeError = pcclSuccess;
  const flagcxResult_t result = ppupcclResultToFlagcx(
      api->pcclCommGetAsyncError(comm->base, &nativeError));
  if (result == flagcxSuccess)
    *asyncError = ppupcclResultToFlagcx(nativeError);
  return result;
}

flagcxResult_t ppupcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

flagcxResult_t ppupcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                          size_t size, void **handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommDeregister(flagcxInnerComm_t comm,
                                            void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                                void *buff, size_t size,
                                                flagcxInnerWindow_t *win,
                                                int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                  flagcxInnerWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, int root,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclResultToFlagcx(api->pcclReduce(sendbuff, recvbuff, count,
                                               nativeDatatype, nativeOp, root,
                                               comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int rank, nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclCommUserRank(comm->base, &rank)));
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclCommCount(comm->base, &nranks)));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = (char *)(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupStart()));
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(ppupcclResultToFlagcx(
                          api->pcclRecv((void *)(buffer + r * size), size,
                                        pcclChar, r, comm->base, stream->base)),
                      res, group_exit);
    }
  }
  FLAGCXCHECKGOTO(
      ppupcclResultToFlagcx(api->pcclSend(sendbuff, size, pcclChar, root,
                                          comm->base, stream->base)),
      res, group_exit);
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     int root, flagcxInnerComm_t comm,
                                     flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int rank, nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclCommUserRank(comm->base, &rank)));
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclCommCount(comm->base, &nranks)));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = (const char *)(sendbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupStart()));
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(ppupcclResultToFlagcx(
                          api->pcclSend((const void *)(buffer + r * size), size,
                                        pcclChar, r, comm->base, stream->base)),
                      res, group_exit);
    }
  }
  FLAGCXCHECKGOTO(
      ppupcclResultToFlagcx(api->pcclRecv(recvbuff, size, pcclChar, root,
                                          comm->base, stream->base)),
      res, group_exit);
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       int root, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(api->pcclBroadcast(sendbuff, recvbuff, count,
                                                  nativeDatatype, root,
                                                  comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclResultToFlagcx(api->pcclAllReduce(sendbuff, recvbuff, count,
                                                  nativeDatatype, nativeOp,
                                                  comm->base, stream->base));
}

flagcxResult_t
ppupcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxInnerComm_t comm, flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclResultToFlagcx(
      api->pcclReduceScatter(sendbuff, recvbuff, count, nativeDatatype,
                             nativeOp, comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(api->pcclAllGather(
      sendbuff, recvbuff, count, nativeDatatype, comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclCommCount(comm->base, &nranks)));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = (const char *)(sendbuff);
  char *recvBuffer = (char *)(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupStart()));
  for (int r = 0; r < nranks; r++) {
    FLAGCXCHECKGOTO(ppupcclResultToFlagcx(api->pcclSend(
                        (const void *)(sendBuffer + r * size), size, pcclChar,
                        r, comm->base, stream->base)),
                    res, group_exit);
    FLAGCXCHECKGOTO(ppupcclResultToFlagcx(
                        api->pcclRecv((void *)(recvBuffer + r * size), size,
                                      pcclChar, r, comm->base, stream->base)),
                    res, group_exit);
  }
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                       size_t *sdispls, void *recvbuff,
                                       size_t *recvcounts, size_t *rdispls,
                                       flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclCommCount(comm->base, &nranks)));

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = (const char *)(sendbuff);
  char *recvBuffer = (char *)(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupStart()));
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      FLAGCXCHECKGOTO(
          ppupcclResultToFlagcx(api->pcclSend(
              (const void *)(sendBuffer + sdispls[r] * typeSize),
              sendcounts[r] * typeSize, pcclChar, r, comm->base, stream->base)),
          res, group_exit);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      FLAGCXCHECKGOTO(
          ppupcclResultToFlagcx(api->pcclRecv(
              (void *)(recvBuffer + rdispls[r] * typeSize),
              recvcounts[r] * typeSize, pcclChar, r, comm->base, stream->base)),
          res, group_exit);
    }
  }
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(api->pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorSend(const void *sendbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(api->pcclSend(sendbuff, count, nativeDatatype,
                                             peer, comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorRecv(void *recvbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(api->pcclRecv(recvbuff, count, nativeDatatype,
                                             peer, comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorGroupStart() {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclGroupStart());
}

flagcxResult_t ppupcclAdaptorGroupEnd() {
  const struct ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclGroupEnd());
}

flagcxResult_t
ppupcclAdaptorDevCommReqsInit(flagcxInnerComm_t /*comm*/,
                              flagcxDevCommRequirements * /*reqs*/) {
  return flagcxNotSupported;
}

flagcxResult_t
ppupcclAdaptorDevCommCreate(flagcxInnerComm_t /*comm*/,
                            const flagcxDevCommRequirements * /*reqs*/,
                            flagcxInnerDevComm_t * /*devComm*/) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorDevCommDestroy(flagcxInnerComm_t /*comm*/,
                                            flagcxInnerDevComm_t /*devComm*/) {
  return flagcxNotSupported;
}

struct flagcxCCLAdaptor ppupcclAdaptor = {
    "PPU_PCCL",
    // Basic functions
    ppupcclAdaptorGetVersion, ppupcclAdaptorGetUniqueId,
    ppupcclAdaptorGetErrorString, ppupcclAdaptorGetLastError,
    ppupcclAdaptorGetStagedBuffer,
    // Communicator functions
    ppupcclAdaptorCommInitRank, ppupcclAdaptorCommFinalize,
    ppupcclAdaptorCommDestroy, ppupcclAdaptorCommAbort,
    ppupcclAdaptorCommResume, ppupcclAdaptorCommSuspend,
    ppupcclAdaptorCommCount, ppupcclAdaptorCommCuDevice,
    ppupcclAdaptorCommUserRank, ppupcclAdaptorCommGetAsyncError,
    ppupcclAdaptorMemAlloc, ppupcclAdaptorMemFree, ppupcclAdaptorCommRegister,
    ppupcclAdaptorCommDeregister,
    // Symmetric functions
    ppupcclAdaptorCommWindowRegister, ppupcclAdaptorCommWindowDeregister,
    // Communication functions
    ppupcclAdaptorReduce, ppupcclAdaptorGather, ppupcclAdaptorScatter,
    ppupcclAdaptorBroadcast, ppupcclAdaptorAllReduce,
    ppupcclAdaptorReduceScatter, ppupcclAdaptorAllGather,
    ppupcclAdaptorAlltoAll, ppupcclAdaptorAlltoAllv, ppupcclAdaptorSend,
    ppupcclAdaptorRecv,
    // Group semantics
    ppupcclAdaptorGroupStart, ppupcclAdaptorGroupEnd,
    // Device API
    ppupcclAdaptorDevCommReqsInit, ppupcclAdaptorDevCommCreate,
    ppupcclAdaptorDevCommDestroy};

#endif // USE_PPU_ADAPTOR
