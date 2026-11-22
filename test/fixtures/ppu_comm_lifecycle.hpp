#ifndef FLAGCX_TEST_PPU_COMM_LIFECYCLE_HPP_
#define FLAGCX_TEST_PPU_COMM_LIFECYCLE_HPP_
#include "check.h"
#include "flagcx.h"
#include <gtest/gtest.h>

// Exercise the production PCCL close functions without requiring a PPU SDK.
namespace PpuLifecycleShim {
struct NativeContext {
  int finalizes = 0, queries = 0, destroys = 0, aborts = 0;
  int pendingDestroys = 0;
  flagcxResult_t finalizeResult = flagcxSuccess;
  flagcxResult_t queryResult = flagcxSuccess;
  flagcxResult_t asyncResult = flagcxSuccess;
  flagcxResult_t destroyResult = flagcxSuccess;
};
struct Wrapper {
  NativeContext *base;
  int closePhase = 0;
  int frees = 0;
};
typedef Wrapper *flagcxInnerComm_t;
typedef flagcxResult_t pcclResult_t;
static const pcclResult_t pcclSuccess = flagcxSuccess;
struct ppupcclSymbols {
  pcclResult_t (*pcclCommFinalize)(NativeContext *);
  pcclResult_t (*pcclCommGetAsyncError)(NativeContext *, pcclResult_t *);
  pcclResult_t (*pcclCommDestroy)(NativeContext *);
  pcclResult_t (*pcclCommAbort)(NativeContext *);
};
static const ppupcclSymbols api = {
    [](NativeContext *ctx) {
      ctx->finalizes++;
      return ctx->finalizeResult;
    },
    [](NativeContext *ctx, pcclResult_t *status) {
      ctx->queries++;
      *status = ctx->asyncResult;
      return ctx->queryResult;
    },
    [](NativeContext *ctx) {
      ctx->destroys++;
      if (ctx->pendingDestroys > 0) {
        ctx->pendingDestroys--;
        return flagcxInProgress;
      }
      return ctx->destroyResult;
    },
    [](NativeContext *ctx) {
      ctx->aborts++;
      return flagcxSuccess;
    }};
static flagcxResult_t ppupcclLoadApi(const ppupcclSymbols **out) {
  *out = &api;
  return flagcxSuccess;
}
static flagcxResult_t ppupcclResultToFlagcx(pcclResult_t result) {
  return result;
}
static void free(void *ptr) { ((Wrapper *)ptr)->frees++; }
#include "../../flagcx/adaptor/ccl/ppu_comm_lifecycle.h"
} // namespace PpuLifecycleShim
#endif
