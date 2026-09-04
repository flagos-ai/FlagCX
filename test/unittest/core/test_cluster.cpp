// Unit tests for homogeneous-cluster split validation.

#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

#include "cluster.h"

namespace {

class ScopedClusterSplit {
public:
  explicit ScopedClusterSplit(const char *value) {
    const char *previous = std::getenv("FLAGCX_CLUSTER_SPLIT_LIST");
    if (previous != nullptr) {
      hadPrevious_ = true;
      previous_ = previous;
    }
    setenv("FLAGCX_CLUSTER_SPLIT_LIST", value, 1);
  }

  ~ScopedClusterSplit() {
    if (hadPrevious_) {
      setenv("FLAGCX_CLUSTER_SPLIT_LIST", previous_.c_str(), 1);
    } else {
      unsetenv("FLAGCX_CLUSTER_SPLIT_LIST");
    }
  }

private:
  bool hadPrevious_ = false;
  std::string previous_;
};

flagcxVendor makeVendor(const char *name) {
  flagcxVendor vendor{};
  std::strncpy(vendor.internal, name, MAX_VENDOR_LEN - 1);
  return vendor;
}

flagcxResult_t collectClusterInfo(const flagcxVendor *vendors, int rank,
                                  int nranks, int *homoRanks = nullptr,
                                  int *nclusters = nullptr) {
  flagcxCommunicatorType_t type;
  int homoRank;
  int homoRootRank;
  int localHomoRanks;
  int clusterId;
  int clusterInterRank;
  int localNclusters;
  const flagcxResult_t result = flagcxCollectClusterInfos(
      vendors, &type, &homoRank, &homoRootRank, &localHomoRanks, &clusterId,
      &clusterInterRank, &localNclusters, rank, nranks);
  if (homoRanks != nullptr) {
    *homoRanks = localHomoRanks;
  }
  if (nclusters != nullptr) {
    *nclusters = localNclusters;
  }
  return result;
}

} // namespace

TEST(ClusterSplit, RejectsZeroCount) {
  ScopedClusterSplit split("0");
  const flagcxVendor vendors[] = {makeVendor("NVIDIA"), makeVendor("NVIDIA")};

  EXPECT_EQ(collectClusterInfo(vendors, 0, 2), flagcxSystemError);
}

TEST(ClusterSplit, RejectsCountLargerThanCluster) {
  ScopedClusterSplit split("3");
  const flagcxVendor vendors[] = {makeVendor("NVIDIA"), makeVendor("NVIDIA")};

  EXPECT_EQ(collectClusterInfo(vendors, 0, 2), flagcxSystemError);
}

TEST(ClusterSplit, ValidatesEveryClusterOnEveryRank) {
  ScopedClusterSplit split("1,2");
  const flagcxVendor vendors[] = {makeVendor("NVIDIA"), makeVendor("NVIDIA"),
                                  makeVendor("METAX")};

  // Rank 0 belongs to the valid first cluster, but it must still reject the
  // oversized split count for the second cluster.
  EXPECT_EQ(collectClusterInfo(vendors, 0, 3), flagcxSystemError);
}

TEST(ClusterSplit, AcceptsTwoWaySplitForTwoRanks) {
  ScopedClusterSplit split("2");
  const flagcxVendor vendors[] = {makeVendor("NVIDIA"), makeVendor("NVIDIA")};
  int homoRanks = 0;
  int nclusters = 0;

  EXPECT_EQ(collectClusterInfo(vendors, 0, 2, &homoRanks, &nclusters),
            flagcxSuccess);
  EXPECT_EQ(homoRanks, 1);
  EXPECT_EQ(nclusters, 2);
}
