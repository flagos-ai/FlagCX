/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UCX_ADAPTOR_H
#define UCX_ADAPTOR_H

#ifdef USE_UCX

#include "check.h"
#include "ib_common.h"
#include "socket.h"
#include <pthread.h>
#include <ucp/api/ucp.h>

// UCX Constants

// UCX Communication State Enum
enum flagcxUcxCommState {
  flagcxUcxCommStateStart = 0,
  flagcxUcxCommStateConnect = 1,
  flagcxUcxCommStateAccept = 3,
};

// UCX Communication Stage Structure
struct flagcxUcxCommStage {
  enum flagcxUcxCommState state;
  uint8_t iteration;
  void *sock;
  void *comm;
};

// UCX Memory Handle
typedef struct flagcxUcxMhandle {
  ucp_mem_h ucpMemh;
  ucp_rkey_h rkey;
  int memType;
} flagcxUcxMhandle_t;

// UCX Endpoint List
struct flagcxUcxEpList {
  struct flagcxSocket *sock;
  struct flagcxUcxEpList *next;
};

// UCX Worker Structure
typedef struct flagcxUcxWorker {
  ucp_worker_h worker; /* ucp worker associated with ctx */
  ucp_context_h ctx;   /* ucp_context bounded to specific device */
  struct flagcxUcxEpList
      *eps; /* oob conection to all endpoints that were opened on this worker */

  int count;        /* number of connections that uses this worker */
  int dev;          /* Managed device */
  pthread_t thread; /* Owner thread */

  struct flagcxUcxWorker *next;
} flagcxUcxWorker_t;

// UCX Listen Handle
typedef struct flagcxUcxListenHandle {
  union flagcxSocketAddress connectAddr; /* reciever socket address */
  uint64_t magic;                        /* random number to help debugging */
  ucp_tag_t tag; /* tag that is used to distiguish data that was sent to
                    this reciever. Required when shared worker is used. */
  struct flagcxUcxCommStage stage;
} flagcxUcxListenHandle_t;

// UCX Listen Communicator
typedef struct flagcxUcxListenComm {
  int dev;                  /* device number in flagcxIbDevs which will
                             * be used to recieve data */
  struct flagcxSocket sock; /* socket for OOB connection */
  ucp_context_h ctx; /* ucp_context associated with specific device dev */
  flagcxUcxWorker_t *ucxWorker; /* flagcxUcxWorker created on ctx, worker can
                           be shared between multiple connections */
  ucp_tag_t tag; /* tag that is used to distiguish data that was sent to
                    this reciever. Required when shared worker is used.*/
  struct flagcxUcxCommStage stage;
} flagcxUcxListenComm_t;

// UCX Connect Message
typedef struct flagcxUcxConnectMsg {
  size_t addrLen;
} flagcxUcxConnectMsg_t;

// Forward declaration
struct flagcxUcxComm;

// UCX Request Structure
typedef struct flagcxUcxRequest {
  struct flagcxUcxRequest *next; /* Next request in the free list */
  struct flagcxUcxComm *comm;    /* Owning communicator */
  ucp_worker_h worker;           /* Worker for all requests */
  int pending;                   /* How many requests are still pending */
  int count;                     /* How many requests are contained */
  int size[FLAGCX_NET_IB_MAX_RECVS];
} flagcxUcxRequest_t;

// UCX GPU Flush Structure
typedef struct flagcxUcxGpuFlush {
  int enabled;
  int hostMem;
  ucp_ep_h flushEp;
} flagcxUcxGpuFlush_t;

// UCX Context Structure
typedef struct flagcxUcxCtx {
  ucp_context_h flagcxUcxCtx;
  flagcxUcxGpuFlush_t gpuFlush;
} flagcxUcxCtx_t;

// UCX Communicator Structure
typedef struct flagcxUcxComm {
  ucp_context_h ctx;            /* ucp_context bounded to specific device */
  flagcxUcxGpuFlush_t gpuFlush; /* flushing handle */
  flagcxUcxWorker_t *ucxWorker; /* ucp worker associated with ctx */
  ucp_ep_h ep;                  /* ucp endpoint created on worker */
  ucp_tag_t tag;  /* datapath tag to filter out message that are not
                     belong to this connnection */
  ucp_tag_t ctag; /* controlpath tag to filter out message that are not
                     belong to this connnection */
  struct flagcxSocket sock; /* socket for OOB connection */
  int ready; /* indicates that receive communicator is fully initialized */
  flagcxUcxRequest_t reqs[MAX_REQUESTS]; /* max inflight requests */
  flagcxUcxRequest_t *freeReq;           /* first request available */
  flagcxUcxConnectMsg_t *msg; /* message to establish reverse connection */
  void *connectReq;           /* msg request */
} flagcxUcxComm_t;

// UCX Macros
#define UCXCHECK(cmd)                                                          \
  do {                                                                         \
    ucs_status_t e = cmd;                                                      \
    if (UCS_OK != e) {                                                         \
      WARN("Failed: UCX error %s:%d '%s'\n", __FILE__, __LINE__,               \
           ucs_status_string(e));                                              \
      return flagcxInternalError;                                              \
    }                                                                          \
  } while (0)

#define UCXCHECK_VOID(cmd)                                                     \
  do {                                                                         \
    ucs_status_t e = cmd;                                                      \
    if (UCS_OK != e) {                                                         \
      WARN("Failed: UCX error %s:%d '%s'\n", __FILE__, __LINE__,               \
           ucs_status_string(e));                                              \
    }                                                                          \
  } while (0)

#endif // USE_UCX

#endif // UCX_ADAPTOR_H
