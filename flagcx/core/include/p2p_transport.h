/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * P2P Engine internal transport submission interface.
 *
 * The common Engine plans slices and schedules batches through this ABI.
 * Transport adaptors own lane selection, credit accounting, native request
 * construction and completion progress. No transport-native type is exposed
 * here.
 ************************************************************************/

#ifndef FLAGCX_P2P_TRANSPORT_H_
#define FLAGCX_P2P_TRANSPORT_H_

#include "flagcx.h"
#include "flagcx_net_adaptor.h"

#include <stddef.h>
#include <stdint.h>

enum flagcxP2pTransportOpcode : uint8_t {
  FLAGCX_P2P_TRANSPORT_WRITE = 0,
  FLAGCX_P2P_TRANSPORT_READ = 1,
};

/* One fully planned network operation. Both address ranges must fit in the
 * supplied local and remote physical MR chunks. */
struct flagcxP2pTransportSlice {
  uint64_t localVa;
  uint64_t remoteVa;
  uint32_t length;
  uint8_t opcode;
  uint8_t reserved[3];

  /* The local handle remains opaque to the Engine. The adaptor uses it to
   * recover transport-native registration state (ibv_mr, memp_t, ...). */
  void *localMrHandle;
  struct flagcxNetMrInfo localMrInfo;
  struct flagcxNetMrInfo remoteMrInfo;
};

struct flagcxP2pTransportCaps {
  uint32_t maxBatchSize;
  uint32_t maxInflightBatches;
};

/* Apply the Engine's common cut policy to one position in a logical transfer.
 * The result never crosses either physical MR boundary or the transport's
 * uint32 length field. A short tail may be merged into the preceding slice. */
static inline uint32_t flagcxP2pPlanSliceLength(size_t remaining,
                                                size_t localMrRemaining,
                                                size_t remoteMrRemaining,
                                                size_t configuredSliceSize,
                                                size_t fragmentLimit) {
  size_t hardLimit = remaining;
  if (localMrRemaining < hardLimit)
    hardLimit = localMrRemaining;
  if (remoteMrRemaining < hardLimit)
    hardLimit = remoteMrRemaining;
  if (hardLimit > UINT32_MAX)
    hardLimit = UINT32_MAX;
  if (hardLimit == 0)
    return 0;

  if (configuredSliceSize != 0 && hardLimit > configuredSliceSize &&
      hardLimit - configuredSliceSize > fragmentLimit)
    hardLimit = configuredSliceSize;
  return (uint32_t)hardLimit;
}

/*
 * Request ownership:
 * - submitBatch() returns one opaque request for the whole batch.
 * - test() is non-blocking. When it sets done=1, it also consumes/frees the
 *   request; the caller must not test it again.
 * - flagcxInProgress from submitBatch() is transient backpressure. The Engine
 *   retains the slices and retries later.
 * - progress() may be a no-op for callback-driven transports.
 */
struct flagcxP2pTransportOps {
  const char *name;

  /* Translate the adaptor's public device index to the small device context
   * integer expected by regMr/deregMr. */
  flagcxResult_t (*getRegistrationDevice)(int netDev, int *registrationDev);

  flagcxResult_t (*getCaps)(void *sendComm,
                            struct flagcxP2pTransportCaps *caps);

  flagcxResult_t (*submitBatch)(void *sendComm,
                                const struct flagcxP2pTransportSlice *slices,
                                int count, void **request);

  flagcxResult_t (*progress)(void *sendComm);

  flagcxResult_t (*test)(void *request, int *done, int *failed);
};

extern const struct flagcxP2pTransportOps flagcxP2pIbrcTransportOps;

#ifdef USE_ACCL_BAREX
extern const struct flagcxP2pTransportOps flagcxP2pBarexTransportOps;
#endif

#endif // FLAGCX_P2P_TRANSPORT_H_
