/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_P2P_ENGINE_BACKEND_H_
#define FLAGCX_P2P_ENGINE_BACKEND_H_

#include "p2p_engine_core.h"

#include <mutex>

struct flagcxNetAdaptor;

struct flagcxP2pNetBackendContext {
  struct flagcxNetAdaptor *adaptor = NULL;
  void *sendComm = NULL;
  std::mutex *progressMutex = NULL;
  int write = 0;
};

flagcxResult_t
flagcxP2pNetBackendInit(struct flagcxP2pNetBackendContext *context,
                        struct flagcxNetAdaptor *adaptor, void *sendComm,
                        std::mutex *progressMutex, int write,
                        struct flagcxP2pTransferBackend *backend);

#endif // FLAGCX_P2P_ENGINE_BACKEND_H_
