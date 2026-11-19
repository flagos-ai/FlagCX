#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "dlsymbols.h"
#include <cstdlib>
#include <mutex>
#include <string>
#include <strings.h>
#include <utility>

namespace {

#define PPU_PCCL_SYMBOLS(M)                                                    \
  M(pcclGetVersion)                                                            \
  M(pcclGetUniqueId)                                                           \
  M(pcclGetErrorString)                                                        \
  M(pcclGetLastError)                                                          \
  M(pcclCommInitRank)                                                          \
  M(pcclCommFinalize)                                                          \
  M(pcclCommDestroy)                                                           \
  M(pcclCommAbort)                                                             \
  M(pcclCommCount)                                                             \
  M(pcclCommCuDevice)                                                          \
  M(pcclCommUserRank)                                                          \
  M(pcclCommGetAsyncError)                                                     \
  M(pcclReduce)                                                                \
  M(pcclBroadcast)                                                             \
  M(pcclAllReduce)                                                             \
  M(pcclReduceScatter)                                                         \
  M(pcclAllGather)                                                             \
  M(pcclSend)                                                                  \
  M(pcclRecv)                                                                  \
  M(pcclGroupStart)                                                            \
  M(pcclGroupEnd)

struct ppupcclSymbols {
#define PPU_PCCL_DECLARE(name) decltype(&::name) name;
  PPU_PCCL_SYMBOLS(PPU_PCCL_DECLARE)
#undef PPU_PCCL_DECLARE
};

static int ppupcclLoadFlags() {
  int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
  flags |= RTLD_DEEPBIND;
#endif
  return flags;
}

static flagcxResult_t ppupcclLoadApi(const ppupcclSymbols **api) {
  static std::once_flag once;
  static ppupcclSymbols symbols = {};
  static flagcxResult_t result = flagcxSystemError;
  std::call_once(once, [] {
    const int flags = ppupcclLoadFlags();
    // The existing CCL_HOME/lib rpath selects the configured SDK. Keep the
    // handle for process lifetime: PCCL owns asynchronous work and threads.
    void *handle = nullptr;
    const char *paths[] = {"libpccl.so.2", "libpccl.so"};
    for (const char *path : paths) {
      handle = flagcxOpenLib(
          path, flags, [](const char *p, int, const char *message) {
            INFO(FLAGCX_INIT, "PPU/PCCL: cannot load %s: %s", p, message);
          });
      if (handle != nullptr)
        break;
    }
    if (handle == nullptr) {
      WARN("PPU/PCCL: failed to load the configured PCCL library");
      return;
    }
    ppupcclSymbols loaded = {};
#define PPU_PCCL_LOAD(name)                                                    \
  do {                                                                         \
    dlerror();                                                                 \
    loaded.name =                                                              \
        reinterpret_cast<decltype(loaded.name)>(dlsym(handle, #name));         \
    const char *error = dlerror();                                             \
    if (error != nullptr || loaded.name == nullptr) {                          \
      WARN("PPU/PCCL: cannot resolve %s: %s", #name,                           \
           error != nullptr ? error : "null symbol");                          \
      return;                                                                  \
    }                                                                          \
  } while (0);
    PPU_PCCL_SYMBOLS(PPU_PCCL_LOAD)
#undef PPU_PCCL_LOAD
    symbols = loaded;
    result = flagcxSuccess;
  });
  *api = result == flagcxSuccess ? &symbols : nullptr;
  return result;
}

static const char *ppupcclPluginSetting(const char *name,
                                        const char *compatName) {
  const char *value = std::getenv(name);
  return value != nullptr ? value : std::getenv(compatName);
}

static void ppupcclPreloadKernelPlugin(const ppupcclSymbols *api) {
  static std::once_flag once;
  // Retain our reference after PCCL closes a communicator's plugin handle.
  // The runtime's registered host kernel entry points belong to this library.
  static void *pluginHandle = nullptr;
  std::call_once(once, [api] {
    const char *enabled = ppupcclPluginSetting("PCCL_ENABLE_EXT_KERNEL",
                                               "NCCL_ENABLE_EXT_KERNEL");
    if (enabled != nullptr && *enabled != '\0' &&
        std::strtoll(enabled, nullptr, 0) == 0)
      return;
    if (ppupcclPluginSetting("PCCL_ALGO", "NCCL_ALGO") != nullptr ||
        ppupcclPluginSetting("PCCL_PROTO", "NCCL_PROTO") != nullptr)
      return;

    const char *configured = ppupcclPluginSetting("PCCL_EXT_KERNEL_PLUGIN",
                                                  "NCCL_EXT_KERNEL_PLUGIN");
    if (configured != nullptr && strncasecmp(configured, "STATIC_PLUGIN",
                                             sizeof("STATIC_PLUGIN") - 1) == 0)
      return;

    const char *name = configured != nullptr && *configured != '\0'
                           ? configured
                           : "libpccl-ext-kernel.so";
    auto open = [](const char *path) {
#ifdef RTLD_NOLOAD
      void *resident = flagcxOpenLib(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD,
                                     [](const char *, int, const char *) {});
      if (resident != nullptr) {
        INFO(FLAGCX_INIT,
             "PPU/PCCL: kernel plugin %s was already loaded; retaining its "
             "existing symbol bindings",
             path);
        return resident;
      }
#endif
      INFO(FLAGCX_INIT, "PPU/PCCL: preloading kernel plugin %s with flags %#x",
           path, ppupcclLoadFlags());
      return flagcxOpenLib(path, ppupcclLoadFlags(),
                           [](const char *p, int, const char *message) {
                             INFO(FLAGCX_INIT,
                                  "PPU/PCCL: optional kernel plugin %s: %s", p,
                                  message);
                           });
    };

    // PCCL later uses an ordinary dlopen for its optional plugin. Load it
    // first with the same scope as PCCL so kernel registration and attribute
    // queries do not bind to different HGGC runtimes through global wrappers.
    // Prefer the selected PCCL installation for the default plugin.
    if (configured == nullptr || *configured == '\0') {
      Dl_info info = {};
      if (dladdr(reinterpret_cast<const void *>(api->pcclGetVersion), &info) &&
          info.dli_fname != nullptr) {
        std::string path(info.dli_fname);
        const auto slash = path.find_last_of('/');
        if (slash != std::string::npos) {
          path.resize(slash + 1);
          path += name;
          pluginHandle = open(path.c_str());
        }
      }
    }
    if (pluginHandle == nullptr)
      pluginHandle = open(name);
    if (pluginHandle != nullptr) {
      INFO(FLAGCX_INIT, "PPU/PCCL: retained kernel plugin %s", name);
    }
    // Missing plugins remain optional; PCCL selects its built-in kernels.
  });
}

#undef PPU_PCCL_SYMBOLS

} // namespace

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

template <typename... Params, typename... Args>
static flagcxResult_t
ppupcclCall(pcclResult_t (*ppupcclSymbols::*function)(Params...),
            Args &&...args) {
  const ppupcclSymbols *api = nullptr;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx((api->*function)(std::forward<Args>(args)...));
}

static const char *ppupcclErrorString(pcclResult_t result) {
  const ppupcclSymbols *api = nullptr;
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
  return ppupcclCall(&ppupcclSymbols::pcclGetVersion, version);
}

flagcxResult_t ppupcclAdaptorGetUniqueId(flagcxInnerUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    FLAGCXCHECK(flagcxCalloc(uniqueId, 1));
  }
  return ppupcclCall(&ppupcclSymbols::pcclGetUniqueId,
                     (pcclUniqueId *)(*uniqueId));
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
  const ppupcclSymbols *api = nullptr;
  if (ppupcclLoadApi(&api) != flagcxSuccess)
    return "PPU PCCL library could not be loaded; consult the initialization "
           "log";
  return api->pcclGetLastError(comm->base);
}

flagcxResult_t
ppupcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                           flagcxInnerUniqueId_t commId, int rank,
                           struct bootstrapState * /*bootstrap*/) {
  const ppupcclSymbols *api = nullptr;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  ppupcclPreloadKernelPlugin(api);
  if (*comm == NULL) {
    FLAGCXCHECK(flagcxCalloc(comm, 1));
  }
  return ppupcclCall(&ppupcclSymbols::pcclCommInitRank, &(*comm)->base, nranks,
                     *(pcclUniqueId *)commId, rank);
}

flagcxResult_t ppupcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return ppupcclCall(&ppupcclSymbols::pcclCommFinalize, comm->base);
}

flagcxResult_t ppupcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return ppupcclCall(&ppupcclSymbols::pcclCommDestroy, comm->base);
}

flagcxResult_t ppupcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return ppupcclCall(&ppupcclSymbols::pcclCommAbort, comm->base);
}

flagcxResult_t ppupcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommCount(const flagcxInnerComm_t comm,
                                       int *count) {
  return ppupcclCall(&ppupcclSymbols::pcclCommCount, comm->base, count);
}

flagcxResult_t ppupcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                          int *device) {
  return ppupcclCall(&ppupcclSymbols::pcclCommCuDevice, comm->base, device);
}

flagcxResult_t ppupcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                          int *rank) {
  return ppupcclCall(&ppupcclSymbols::pcclCommUserRank, comm->base, rank);
}

flagcxResult_t ppupcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                               flagcxResult_t *asyncError) {
  if (comm == nullptr || asyncError == nullptr)
    return flagcxInvalidArgument;
  pcclResult_t nativeError = pcclSuccess;
  const flagcxResult_t result = ppupcclCall(
      &ppupcclSymbols::pcclCommGetAsyncError, comm->base, &nativeError);
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
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclCall(&ppupcclSymbols::pcclReduce, sendbuff, recvbuff, count,
                     nativeDatatype, nativeOp, root, comm->base, stream->base);
}

flagcxResult_t ppupcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int rank, nranks;
  FLAGCXCHECK(
      ppupcclCall(&ppupcclSymbols::pcclCommUserRank, comm->base, &rank));
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclCommCount, comm->base, &nranks));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupStart));
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(ppupcclCall(&ppupcclSymbols::pcclRecv,
                                  static_cast<void *>(buffer + r * size), size,
                                  pcclChar, r, comm->base, stream->base),
                      res, group_exit);
    }
  }
  FLAGCXCHECKGOTO(ppupcclCall(&ppupcclSymbols::pcclSend, sendbuff, size,
                              pcclChar, root, comm->base, stream->base),
                  res, group_exit);
group_exit:
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupEnd));
  return res;
}

flagcxResult_t ppupcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     int root, flagcxInnerComm_t comm,
                                     flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int rank, nranks;
  FLAGCXCHECK(
      ppupcclCall(&ppupcclSymbols::pcclCommUserRank, comm->base, &rank));
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclCommCount, comm->base, &nranks));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupStart));
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(ppupcclCall(&ppupcclSymbols::pcclSend,
                                  static_cast<const void *>(buffer + r * size),
                                  size, pcclChar, r, comm->base, stream->base),
                      res, group_exit);
    }
  }
  FLAGCXCHECKGOTO(ppupcclCall(&ppupcclSymbols::pcclRecv, recvbuff, size,
                              pcclChar, root, comm->base, stream->base),
                  res, group_exit);
group_exit:
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupEnd));
  return res;
}

flagcxResult_t ppupcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       int root, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclCall(&ppupcclSymbols::pcclBroadcast, sendbuff, recvbuff, count,
                     nativeDatatype, root, comm->base, stream->base);
}

flagcxResult_t ppupcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclCall(&ppupcclSymbols::pcclAllReduce, sendbuff, recvbuff, count,
                     nativeDatatype, nativeOp, comm->base, stream->base);
}

flagcxResult_t
ppupcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxInnerComm_t comm, flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclCall(&ppupcclSymbols::pcclReduceScatter, sendbuff, recvbuff,
                     count, nativeDatatype, nativeOp, comm->base, stream->base);
}

flagcxResult_t ppupcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclCall(&ppupcclSymbols::pcclAllGather, sendbuff, recvbuff, count,
                     nativeDatatype, comm->base, stream->base);
}

flagcxResult_t ppupcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int nranks;
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclCommCount, comm->base, &nranks));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupStart));
  for (int r = 0; r < nranks; r++) {
    FLAGCXCHECKGOTO(
        ppupcclCall(&ppupcclSymbols::pcclSend,
                    static_cast<const void *>(sendBuffer + r * size), size,
                    pcclChar, r, comm->base, stream->base),
        res, group_exit);
    FLAGCXCHECKGOTO(ppupcclCall(&ppupcclSymbols::pcclRecv,
                                static_cast<void *>(recvBuffer + r * size),
                                size, pcclChar, r, comm->base, stream->base),
                    res, group_exit);
  }
group_exit:
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupEnd));
  return res;
}

flagcxResult_t ppupcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                       size_t *sdispls, void *recvbuff,
                                       size_t *recvcounts, size_t *rdispls,
                                       flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int nranks;
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclCommCount, comm->base, &nranks));

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupStart));
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      FLAGCXCHECKGOTO(
          ppupcclCall(
              &ppupcclSymbols::pcclSend,
              static_cast<const void *>(sendBuffer + sdispls[r] * typeSize),
              sendcounts[r] * typeSize, pcclChar, r, comm->base, stream->base),
          res, group_exit);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      FLAGCXCHECKGOTO(
          ppupcclCall(&ppupcclSymbols::pcclRecv,
                      static_cast<void *>(recvBuffer + rdispls[r] * typeSize),
                      recvcounts[r] * typeSize, pcclChar, r, comm->base,
                      stream->base),
          res, group_exit);
    }
  }
group_exit:
  FLAGCXCHECK(ppupcclCall(&ppupcclSymbols::pcclGroupEnd));
  return res;
}

flagcxResult_t ppupcclAdaptorSend(const void *sendbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclCall(&ppupcclSymbols::pcclSend, sendbuff, count, nativeDatatype,
                     peer, comm->base, stream->base);
}

flagcxResult_t ppupcclAdaptorRecv(void *recvbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclCall(&ppupcclSymbols::pcclRecv, recvbuff, count, nativeDatatype,
                     peer, comm->base, stream->base);
}

flagcxResult_t ppupcclAdaptorGroupStart() {
  return ppupcclCall(&ppupcclSymbols::pcclGroupStart);
}

flagcxResult_t ppupcclAdaptorGroupEnd() {
  return ppupcclCall(&ppupcclSymbols::pcclGroupEnd);
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
