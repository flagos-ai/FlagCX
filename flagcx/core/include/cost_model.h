#ifndef FLAGCX_COST_MODEL_H
#define FLAGCX_COST_MODEL_H

#include "c2c_algo.h"
#include "flagcx.h"
#include <vector>

// typedef enum {
//     FullyConnected,
//     Ring,
// } flagcxHomoSimTopo;

// struct flagcxHomoSimInfo {
//     flagcxHomoSimTopo topo[2];
//     int npusCount[2];
//     float bandwidth[2];
//     float latency[2];
// };

constexpr int FLAGCX_INTRA_LAT_IDX = 0;
constexpr int FLAGCX_INTER_LAT_IDX = 1;

#define FLAGCX_VENDOR_NUM 4

class flagcxAlgoTimeEstimator {
public:
  flagcxAlgoTimeEstimator(flagcxC2cPlanner &planner, flagcxDataType_t dtype)
      : planner_(planner), datatype(dtype) {}

  flagcxResult_t getAlgoTime(float *time);

private:
  flagcxResult_t getPreHomoAlgoTime(float *time);

  flagcxResult_t getPostHomoAlgoTime(float *time);

  flagcxResult_t getHomoAlgoTime(flagcxC2cHomoFunc &homoFunc, int rankSize,
                                 int vendor, float *time);

  flagcxResult_t getHeteroAlgoTime(float *time);

  flagcxResult_t getHomoInterAlgoTime(int loop, float *time);

  void generateHeteroFuncForMultiNic(int rank, int loop,
                                     flagcxC2cHeteroFunc &heteroFunc);

  void generateHeteroFuncForSingleNic(int rank,
                                      flagcxC2cHeteroFunc &heteroFunc);

  float getP2pTimePerNic(
      uint64_t netGuid,
      std::unordered_map<uint64_t, std::vector<int>> &nicRankMap,
      std::unordered_map<int, std::vector<flagcxC2cHeteroFunc>> &heteroFuncMap);

  float getRefreshTime();

  float getSendRecvTime(float curClusterLat, float remoteClusterLat, float bw,
                        int totalCount, size_t chunkSize);

  flagcxC2cPlanner &planner_;
  flagcxDataType_t datatype;
};

#endif