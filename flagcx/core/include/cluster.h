#ifndef FLAGCX_CLUSTER_H_
#define FLAGCX_CLUSTER_H_

#include "adaptor.h"
#include "flagcx.h"
#include "param.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

flagcxResult_t parseClusterSplitList(const char *input,
                                     std::vector<int> &output);

flagcxResult_t flagcxCollectClusterInfos(const flagcxVendor *allData,
                                         flagcxCommunicatorType_t *type,
                                         int *homo_rank, int *homo_root_rank,
                                         int *homo_ranks, int *cluster_id,
                                         int *cluster_inter_rank,
                                         int *nclusters, int rank, int nranks);

flagcxResult_t flagcxFillClusterVendorInfo(const flagcxVendor *allData,
                                           flagcxComm *comm, int *clusterIdData,
                                           int nranks, int ncluster);

#endif // end include guard