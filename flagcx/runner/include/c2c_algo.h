#ifndef FLAGCX_C2C_ALGO_H_
#define FLAGCX_C2C_ALGO_H_

#include "adaptor.h"
#include "collectives.h"
#include "flagcx.h"
#include "group.h"
#include "param.h"
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <unordered_map>

typedef enum {
  flagcxAlgoSequential = 0,
  flagcxAlgoPipeline = 1,
  flagcxAlgoInput = 2
} flagcxAlgorithm_t;

size_t getC2cCommPatternHash(size_t count, size_t rootClusterId,
                             flagcxCommOp_t commOp, flagcxRedOp_t redOp,
                             flagcxComm_t comm);

template <typename Key, typename Value>
class flagcxLRUCache {
public:
  flagcxLRUCache(size_t capacity) : capacity_(capacity) {}

  bool get(const Key &key, Value &value) {
    auto it = cacheMap_.find(key);
    if (it == cacheMap_.end())
      return false;

    // Move the accessed item to the front of the list
    cacheItems_.splice(cacheItems_.begin(), cacheItems_, it->second);
    value = it->second->second;
    return true;
  }

  void put(const Key &key, const Value &value) {
    auto it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
      // Update and move to front
      it->second->second = value;
      cacheItems_.splice(cacheItems_.begin(), cacheItems_, it->second);
    } else {
      // Insert new element
      if (cacheItems_.size() == capacity_) {
        // Remove least recently used item
        auto lru = cacheItems_.back();
        cacheMap_.erase(lru.first);
        cacheItems_.pop_back();
      }
      cacheItems_.emplace_front(key, value);
      cacheMap_[key] = cacheItems_.begin();
    }
  }

private:
  size_t capacity_;
  std::list<std::pair<Key, Value>> cacheItems_;
  std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator>
      cacheMap_;
};

struct flagcxBufferInfo {
public:
  flagcxBufferInfo(size_t offset, size_t count, int clusterIdToSend, int isRecv,
                   int isScheduled, int peerRank, int loopId)
      : offset_(offset), count_(count), clusterIdToSend_(clusterIdToSend),
        isRecv_(isRecv), isScheduled_(isScheduled), peerRank_(peerRank),
        loopId_(loopId) {}
  ~flagcxBufferInfo() {}

  size_t offset_;
  size_t count_;
  int clusterIdToSend_; // only required for send
  int isRecv_;          // 0: send, 1: recv
  int isScheduled_;     // 0: un-scheduled, 1: scheduled
  int peerRank_;
  int loopId_;
};

class flagcxInterRankBufferInfoManager {
public:
  flagcxInterRankBufferInfoManager(size_t totalCount);
  ~flagcxInterRankBufferInfoManager();
  flagcxInterRankBufferInfoManager() = default;
  flagcxInterRankBufferInfoManager(const flagcxInterRankBufferInfoManager &) =
      default;

  bool checkIfPossibleToPush(int clusterId, int rank, size_t offset,
                             size_t count);
  bool checkIfPossibleToSplitAndPush(int clusterId, int rank, size_t offset,
                                     size_t count, size_t *splitCount,
                                     int *pushMode);
  bool checkIsFull(int clusterId, int rank);
  bool checkIsScheduled(int clusterId, int rank);
  std::list<flagcxBufferInfo> &getBufferInfoList(int clusterId, int rank);
  void pushBackBufferInfo(int clusterId, int rank, size_t offset, size_t count,
                          int clusterIdToSend, int isRecv, int isScheduled,
                          int peerRank, int loopId);
  void popFrontBufferInfo(int clusterId, int rank);
  void resetBufferInfo();
  void printBufferInfo(int step); // 0: intial, 1: internal, 2: final

  size_t totalCount_; // total communication count
  std::map<int, std::map<int, std::list<flagcxBufferInfo>>>
      bufferInfos_; // map<clusterId, map<rank, list[struct{offset, count,
                    // isRecv, isScheduled}]>>
};

class flagcxC2cP2pOp {
public:
  flagcxC2cP2pOp(int rank, int peerRank, size_t offset, size_t count,
                 int isRecv);
  ~flagcxC2cP2pOp();

  flagcxResult_t run(void *buff, flagcxDataType_t datatype, flagcxComm_t comm,
                     flagcxStream_t stream);

  int rank_;
  int peerRank_;
  size_t offset_;
  size_t count_;
  int isRecv_; // 0: send, 1: recv
};

class flagcxC2cHomoFunc {
public:
  flagcxC2cHomoFunc(int rootRank, int sendType, int recvType, size_t sendOffset,
                    size_t recvOffset, size_t count, int homoType,
                    flagcxCommOp_t commOp);
  flagcxC2cHomoFunc(
      int rootRank, int sendType, int recvType, size_t sendOffset,
      size_t recvOffset, size_t count, int homoType, flagcxCommOp_t commOp,
      flagcxInterRankBufferInfoManager interRankBufferInfoManager);
  ~flagcxC2cHomoFunc();

  flagcxResult_t run(const void *sendbuff, void *recvbuff, void *scratchbuff,
                     flagcxDataType_t datatype, flagcxRedOp_t redOp, int root,
                     flagcxComm_t comm, flagcxStream_t stream,
                     size_t *sendCounts = nullptr, size_t *sDispls = nullptr,
                     size_t *recvCounts = nullptr, size_t *rDispls = nullptr);

  flagcxC2cHomoFunc(FILE *file, size_t chunksize);

  int rootRank_;
  int sendType_;
  int recvType_;
  size_t sendOffset_;
  size_t recvOffset_;
  size_t count_;
  int homoType_;
  flagcxCommOp_t commOp_;
  flagcxInterRankBufferInfoManager interRankBufferInfoManager_;
};

class flagcxC2cHeteroFunc {
public:
  friend class flagcxAlgoTimeEstimator;
  friend void serializeHeteroFunc(FILE *file, size_t chunksize,
                                  const flagcxC2cHeteroFunc &func, int indent);
  flagcxC2cHeteroFunc();
  ~flagcxC2cHeteroFunc();

  void addP2pOp(int rank, int peerRank, size_t offset, size_t count,
                int isRecv);
  flagcxResult_t run(void *sendbuff, void *recvbuff, flagcxDataType_t datatype,
                     flagcxComm_t comm, flagcxStream_t stream);
  flagcxC2cHeteroFunc(FILE *file, size_t chunksize);

private:
  std::vector<flagcxC2cP2pOp> p2pOps_;
};

class flagcxC2cRefreshFunc {
public:
  flagcxC2cRefreshFunc();
  flagcxC2cRefreshFunc(size_t offset, size_t count, size_t totalCount,
                       flagcxRedOp_t redOp);
  flagcxC2cRefreshFunc(int bufftype, size_t start, size_t offset, size_t count,
                       size_t totalCount, flagcxRedOp_t redOp);
  ~flagcxC2cRefreshFunc();

  flagcxResult_t run(void *recvbuff, void *scratchbuff,
                     flagcxDataType_t datatype, flagcxStream_t stream);

  int bufftype_;
  size_t start_;
  size_t offset_;
  size_t count_;
  size_t totalCount_;
  flagcxRedOp_t redOp_;
};

class flagcxC2cPlanner {
public:
  friend class flagcxAlgoTimeEstimator;
  flagcxC2cPlanner(size_t sendCount, size_t recvCount, int rootRank,
                   flagcxComm_t comm, flagcxCommOp_t commOp,
                   flagcxRedOp_t redOp);
  ~flagcxC2cPlanner();
  flagcxC2cPlanner() = default;
  flagcxC2cPlanner(const flagcxC2cPlanner &) = default;
  // constructor for reading a c2c algorithm from an xml input file
  flagcxC2cPlanner(const char *path);
  flagcxC2cPlanner &operator=(const flagcxC2cPlanner &) = default;

  flagcxCommOp_t getC2cHomoCommOp(int homoType, int mode);
  // import a planner from an xml file
  flagcxResult_t importXml(const char *prefix);
  // export a planner to an xml file
  flagcxResult_t exportXml(const char *prefix);
  flagcxResult_t refresh(
      int isSendRecv); // 0: refresh recv info only; 1: refresh send+recv info
  flagcxResult_t searchHeteroSendRecvOps(int searchMethod,
                                         int loopId); // 0: DFS; 1: BFS
  flagcxResult_t findStrategy();
  flagcxResult_t execute(const void *sendbuff, void *recvbuff,
                         flagcxDataType_t datatype, int root,
                         flagcxStream_t stream, size_t *sendCounts = nullptr,
                         size_t *sDispls = nullptr,
                         size_t *recvCounts = nullptr,
                         size_t *rDispls = nullptr);

private:
  int nSeqPreSteps_;
  int nPipePreSteps_;
  int nSeqInterSteps_;
  int nPipePostSteps_;
  int nSeqPostSteps_;
  size_t nchunks_;
  size_t sendCount_;
  size_t recvCount_;
  int rootRank_; // used for gather, scatter
  flagcxComm_t comm_;
  flagcxCommOp_t commOp_;
  flagcxRedOp_t redOp_;
  size_t *sendCounts_; // used for alltoallv, etc.
  size_t *sDispls_;
  size_t *recvCounts_;
  size_t *rDispls_;
  std::vector<std::vector<int>> clusterInterRankList_;
  int clusterId_;
  int rank_; // global rank
  int homoMyRank_;
  int homoRootRank_;
  int homoRanks_;
  int homoInterMyRank_;
  int homoInterRootRank_;
  int homoInterRanks_;
  size_t totalCount_; // equal to either sendCount_ or recvCount_
  int rootClusterId_;
  int isRootCluster_;
  int clusterCount_;
  int clusterOffset_;
  int multiNic_;
  int eachNicPerRank_;
  int strategyFound_;
  flagcxInterRankBufferInfoManager interRankBufferInfoManager_;
  flagcxC2cRefreshFunc refreshFunc_;
  flagcxAlgorithm_t algorithm_;
  std::vector<std::vector<flagcxC2cHomoFunc>> preHomoFuncSteps_;
  std::vector<std::vector<flagcxC2cHeteroFunc>> heteroFuncSteps_;
  std::vector<std::vector<flagcxC2cHomoFunc>> homoInterFuncSteps_;
  std::vector<std::vector<flagcxC2cHomoFunc>> postHomoFuncSteps_;
  void *scratchBuffer_; // used for intermediate processing
};

#endif // end include guard
