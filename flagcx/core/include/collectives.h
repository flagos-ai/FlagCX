#ifndef FLAGCX_COLLECTIVES_H_
#define FLAGCX_COLLECTIVES_H_

#include "flagcx.h"
#include "info.h"

flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream);

flagcxResult_t flagcxHeteroRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream);

#endif
