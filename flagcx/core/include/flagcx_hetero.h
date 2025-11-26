#include "flagcx.h"
#include "type.h"

typedef struct flagcxHeteroComm *flagcxHeteroComm_t;

flagcxResult_t flagcxHeteroGetVersion(int *version);

flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream);

flagcxResult_t flagcxHeteroRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream);

flagcxResult_t flagcxHeteroGroupStart();

flagcxResult_t flagcxHeteroGroupEnd();

flagcxResult_t flagcxHeteroGetUniqueId(flagcxUniqueId *out);

flagcxResult_t flagcxHeteroCommInitRank(flagcxHeteroComm_t *newcomm, int nranks,
                                        flagcxUniqueId commId, int myrank);

flagcxResult_t flagcxHeteroCommCount(const flagcxHeteroComm_t comm, int *count);

flagcxResult_t flagcxHeteroCommUserRank(const flagcxHeteroComm_t comm,
                                        int *rank);

flagcxResult_t flagcxHeteroCommDestroy(flagcxHeteroComm_t comm);
