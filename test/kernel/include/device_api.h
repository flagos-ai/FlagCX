/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Test-only Device API kernel declarations.
 * These kernels are compiled from device_api.cu in test/kernel/[platform]/
 * NOT part of libflagcx.so.
 ************************************************************************/

#ifndef TEST_KERNEL_DEVICE_API_H_
#define TEST_KERNEL_DEVICE_API_H_

#include "flagcx_kernel.h"

// Intra-node AllReduce using FlagCX Device API.
flagcxResult_t flagcxIntraAllReduce(flagcxDevMem_t devMem, size_t count,
                                    flagcxDataType_t datatype,
                                    flagcxDevComm_t devComm,
                                    flagcxStream_t stream);

// Inter-node one-sided AlltoAll (put + waitSignal + flush).
flagcxResult_t flagcxInterOneSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node two-sided AlltoAll (send/recv + term/wait via FIFO).
flagcxResult_t flagcxInterTwoSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node Device API test kernels.
flagcxResult_t flagcxInterTestPutSignalInc(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

flagcxResult_t flagcxInterTestPutSignalAddDecoupled(
    flagcxDevMem_t sendMem, flagcxDevMem_t recvMem, size_t count,
    flagcxDataType_t datatype, flagcxDevComm_t devComm, flagcxStream_t stream);

flagcxResult_t
flagcxInterTestCounterPipeline(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                               size_t count, flagcxDataType_t datatype,
                               flagcxDevComm_t devComm, flagcxStream_t stream,
                               uint64_t *resultBuf);

flagcxResult_t flagcxInterTestPutValue(flagcxDevMem_t recvMem,
                                       flagcxDevComm_t devComm,
                                       flagcxStream_t stream,
                                       size_t putValBase);

flagcxResult_t flagcxInterTestSignal(flagcxDevComm_t devComm,
                                     flagcxStream_t stream);

flagcxResult_t
flagcxInterTestFlushDecouple(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                             size_t count, flagcxDataType_t datatype,
                             flagcxDevComm_t devComm, flagcxStream_t stream);

flagcxResult_t flagcxInterTestFollowShadow(flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

flagcxResult_t flagcxInterTestMeetShadow(flagcxDevComm_t devComm,
                                         flagcxStream_t stream);

flagcxResult_t flagcxInterTestReset(flagcxDevComm_t devComm,
                                    flagcxStream_t stream, uint64_t *resultBuf);

flagcxResult_t flagcxInterTestGet(flagcxDevMem_t sendMem,
                                  flagcxDevMem_t recvMem, size_t count,
                                  flagcxDataType_t datatype,
                                  flagcxDevComm_t devComm,
                                  flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_API_H_
