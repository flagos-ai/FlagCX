#include "tuner_util.h"

#ifdef USE_KUNLUNXIN_ADAPTOR

static EnvVar xdrEnable("BKCL_ENABLE_XDR", {"0", "1"}, "0");

static EnvVar l3Rdma("BKCL_FORCE_L3_RDMA", {"0", "1"}, "0");

static EnvVar rdmaVerbs("BKCL_RDMA_VERBS", {"0", "1"}, "0");

static EnvVar treeEnable("BKCL_ENABLE_TREE", {"0", "1"}, "0");

static EnvVar treeThreshold("BKCL_MULTI_TREE_THRESHOLD", {"1048576", "2097152"},
                            "1048576");

std::vector<EnvVar> xcclTunerVars = {xdrEnable, l3Rdma, rdmaVerbs, treeEnable,
                                     treeThreshold};
#endif // USE_KUNLUNXIN_ADAPTOR
