#ifndef FLAGCX_UNIRUNNER_HELPER_H_
#define FLAGCX_UNIRUNNER_HELPER_H_

#include "uni_runner_impl.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

using Json = nlohmann::json;

inline constexpr int kUniRunnerDagCacheFormatVersion = 5;

inline bool uniRunnerDagDataTypeValueValid(int value) {
  return value >= 0 && value < flagcxNumTypes;
}

inline bool uniRunnerDagRedOpValueValid(int value, bool allowNoOp) {
  if (allowNoOp && value == flagcxRedNoOp)
    return true;
  return value >= flagcxSum && value < flagcxNumRedOps;
}

enum uniRunnerDagBufferType {
  uniRunnerDagBufferTypeNone = 0,
  uniRunnerDagBufferTypeInput = 1,
  uniRunnerDagBufferTypeOutput = 2,
  uniRunnerDagBufferTypeScratch = 3
};

struct uniRunnerDagBufferRef {
  uniRunnerDagBufferType bufferType = uniRunnerDagBufferTypeNone;
  int64_t offsetBytes = 0;
};

inline bool uniRunnerDagBindingRangeContains(const void *ptr, const void *base,
                                             size_t bytes,
                                             size_t accessBytes,
                                             int64_t *offsetBytes) {
  if (ptr == nullptr || base == nullptr || offsetBytes == nullptr) {
    return false;
  }
  const uintptr_t ptrAddr = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t baseAddr = reinterpret_cast<uintptr_t>(base);
  if (ptrAddr < baseAddr) {
    return false;
  }
  const uintptr_t delta = ptrAddr - baseAddr;
  if (delta > bytes || accessBytes > bytes - static_cast<size_t>(delta) ||
      delta > static_cast<uintptr_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  *offsetBytes = static_cast<int64_t>(delta);
  return true;
}

struct uniRunnerDagP2pOpDesc {
  uniRunnerDagBufferRef buffer;
  size_t count = 0;
  int peerRank = -1;
  flagcxDataType_t datatype = flagcxInt8;
  flagcxDevicePrim type = flagcxDevicePrimSend;
};

struct uniRunnerDagRedOpDesc {
  uniRunnerDagBufferRef input1;
  uniRunnerDagBufferRef input2;
  uniRunnerDagBufferRef output;
  size_t count = 0;
  flagcxDataType_t datatype = flagcxInt8;
  flagcxRedOp_t redOp = flagcxRedNoOp;
};

struct uniRunnerDagCpyOpDesc {
  uniRunnerDagBufferRef src;
  uniRunnerDagBufferRef dst;
  size_t count = 0;
  flagcxDataType_t datatype = flagcxInt8;
};

// Cacheable IPC DAG structure. Invocation-local execution state (for example
// DevMem/Window handles, parent flag offsets, trigger indices, epochs, chunk
// counters, and trigger state) is deliberately rebound after materialization.
struct uniRunnerDagIpcOpDesc {
  size_t srcOffsetBytes = 0;
  size_t dstOffsetBytes = 0;
  size_t bytes = 0;
  flagcxIpcBufferType srcBufferType = flagcxIpcBufferInput;
  int peerLocalRank = -1;
  uint32_t readySlot = 0;
};

struct uniRunnerDagNodeDesc {
  uniRunnerDagNodeType nodeType = uniRunnerDagNodeTypeP2p;
  int nodeIdx = 0;
  std::vector<int> parents;
  std::vector<int> children;
  std::vector<uniRunnerDagP2pOpDesc> p2pOps;
  uniRunnerDagRedOpDesc red;
  uniRunnerDagCpyOpDesc cpy;
  uniRunnerDagIpcOpDesc ipc;
};

struct uniRunnerDagTemplate {
  uniRunnerDagCacheKey key = {};
  size_t hashValue = 0;
  std::vector<uniRunnerDagNodeDesc> nodes;
};

// Process-local compiled form of a cacheable DAG. Disk persistence continues
// to contain only dagTemplate; the stable topological order is reconstructed
// once when a template first enters this process and then reused read-only.
struct uniRunnerCompiledDagTemplate {
  uniRunnerDagTemplate dagTemplate;
  std::vector<int> topoOrder;
  size_t numNodes = 0;
  size_t numHostNodes = 0;
  size_t numRedNodes = 0;
  size_t numIpcNodes = 0;
  size_t maxParallelRedNodes = 0;
  size_t maxRedNodeCount = 0;
  size_t maxIpcNodeBytes = 0;
};

flagcxResult_t compileUniRunnerDagTemplate(
    const uniRunnerDagTemplate &dagTemplate,
    uniRunnerCompiledDagTemplate *compiledTemplate);

// Production validation helpers exposed here so CPU-only cache tests exercise
// the exact IPC materialization and topology identity rules.
flagcxResult_t validateUniRunnerIpcDagTemplateBindings(
    const uniRunnerDagTemplate &dagTemplate, size_t inputBytes,
    size_t outputBytes, int localRanks);
flagcxResult_t getUniRunnerIpcTopologyHash(int localRanks, int nranks,
                                           const int *localRankToRank,
                                           uint64_t *topologyHash);

inline const char *uniRunnerDagAlgoTypeToString(uniRunnerDagAlgoType algoType) {
  switch (algoType) {
    case uniRunnerDagAlgoDummy:
      return "dummy";
    case uniRunnerDagAlgoLocRed:
      return "loc_red";
    case uniRunnerDagAlgoGroupedAG:
      return "grouped_ag";
    case uniRunnerDagAlgoRingAG:
      return "ring_ag";
    case uniRunnerDagAlgoRingAR:
      return "ring_ar";
    case uniRunnerDagAlgoSlicedAR:
      return "sliced_ar";
    case uniRunnerDagAlgoRingRS:
      return "ring_rs";
    case uniRunnerDagAlgoTreeRed:
      return "tree_red";
    case uniRunnerDagAlgoIpcAR:
      return "ipc_ar";
    default:
      return "unknown";
  }
}

inline bool uniRunnerDagAlgoTypeFromString(const std::string &text,
                                           uniRunnerDagAlgoType *algoType) {
  if (text == "dummy") {
    *algoType = uniRunnerDagAlgoDummy;
  } else if (text == "loc_red") {
    *algoType = uniRunnerDagAlgoLocRed;
  } else if (text == "grouped_ag") {
    *algoType = uniRunnerDagAlgoGroupedAG;
  } else if (text == "ring_ag") {
    *algoType = uniRunnerDagAlgoRingAG;
  } else if (text == "ring_ar") {
    *algoType = uniRunnerDagAlgoRingAR;
  } else if (text == "sliced_ar") {
    *algoType = uniRunnerDagAlgoSlicedAR;
  } else if (text == "ring_rs") {
    *algoType = uniRunnerDagAlgoRingRS;
  } else if (text == "tree_red") {
    *algoType = uniRunnerDagAlgoTreeRed;
  } else if (text == "ipc_ar") {
    *algoType = uniRunnerDagAlgoIpcAR;
  } else {
    return false;
  }
  return true;
}

inline const char *uniRunnerCommOpToString(flagcxCommOp_t commOp) {
  switch (commOp) {
    case flagcxCommOpSend:
      return "send";
    case flagcxCommOpRecv:
      return "recv";
    case flagcxCommOpBroadcast:
      return "broadcast";
    case flagcxCommOpGather:
      return "gather";
    case flagcxCommOpScatter:
      return "scatter";
    case flagcxCommOpReduce:
      return "reduce";
    case flagcxCommOpAllReduce:
      return "all_reduce";
    case flagcxCommOpAllGather:
      return "all_gather";
    case flagcxCommOpReduceScatter:
      return "reduce_scatter";
    case flagcxCommOpAlltoAll:
      return "all_to_all";
    case flagcxCommOpAlltoAllv:
      return "all_to_allv";
    case flagcxCommNoOp:
      return "noop";
    default:
      return "unknown";
  }
}

inline bool uniRunnerCommOpFromString(const std::string &text,
                                      flagcxCommOp_t *commOp) {
  if (text == "send") {
    *commOp = flagcxCommOpSend;
  } else if (text == "recv") {
    *commOp = flagcxCommOpRecv;
  } else if (text == "broadcast") {
    *commOp = flagcxCommOpBroadcast;
  } else if (text == "gather") {
    *commOp = flagcxCommOpGather;
  } else if (text == "scatter") {
    *commOp = flagcxCommOpScatter;
  } else if (text == "reduce") {
    *commOp = flagcxCommOpReduce;
  } else if (text == "all_reduce") {
    *commOp = flagcxCommOpAllReduce;
  } else if (text == "all_gather") {
    *commOp = flagcxCommOpAllGather;
  } else if (text == "reduce_scatter") {
    *commOp = flagcxCommOpReduceScatter;
  } else if (text == "all_to_all") {
    *commOp = flagcxCommOpAlltoAll;
  } else if (text == "all_to_allv") {
    *commOp = flagcxCommOpAlltoAllv;
  } else if (text == "noop") {
    *commOp = flagcxCommNoOp;
  } else {
    return false;
  }
  return true;
}

inline const char *uniRunnerDagNodeTypeToString(uniRunnerDagNodeType nodeType) {
  switch (nodeType) {
    case uniRunnerDagNodeTypeP2p:
      return "p2p";
    case uniRunnerDagNodeTypeRed:
      return "red";
    case uniRunnerDagNodeTypeCpy:
      return "cpy";
    case uniRunnerDagNodeTypeIpc:
      return "ipc";
    default:
      return "unknown";
  }
}

inline bool uniRunnerDagNodeTypeFromString(const std::string &text,
                                           uniRunnerDagNodeType *nodeType) {
  if (text == "p2p") {
    *nodeType = uniRunnerDagNodeTypeP2p;
  } else if (text == "red") {
    *nodeType = uniRunnerDagNodeTypeRed;
  } else if (text == "cpy") {
    *nodeType = uniRunnerDagNodeTypeCpy;
  } else if (text == "ipc") {
    *nodeType = uniRunnerDagNodeTypeIpc;
  } else {
    return false;
  }
  return true;
}

inline const char *
uniRunnerDagBufferTypeToString(uniRunnerDagBufferType bufferType) {
  switch (bufferType) {
    case uniRunnerDagBufferTypeNone:
      return "none";
    case uniRunnerDagBufferTypeInput:
      return "input";
    case uniRunnerDagBufferTypeOutput:
      return "output";
    case uniRunnerDagBufferTypeScratch:
      return "scratch";
    default:
      return "unknown";
  }
}

inline bool uniRunnerDagBufferTypeFromString(const std::string &text,
                                             uniRunnerDagBufferType *type) {
  if (text == "none") {
    *type = uniRunnerDagBufferTypeNone;
  } else if (text == "input") {
    *type = uniRunnerDagBufferTypeInput;
  } else if (text == "output") {
    *type = uniRunnerDagBufferTypeOutput;
  } else if (text == "scratch") {
    *type = uniRunnerDagBufferTypeScratch;
  } else {
    return false;
  }
  return true;
}

inline const char *
uniRunnerDagIpcBufferTypeToString(flagcxIpcBufferType bufferType) {
  switch (bufferType) {
    case flagcxIpcBufferInput:
      return "input";
    case flagcxIpcBufferOutput:
      return "output";
    default:
      return "unknown";
  }
}

inline bool
uniRunnerDagIpcBufferTypeFromString(const std::string &text,
                                    flagcxIpcBufferType *bufferType) {
  if (bufferType == nullptr) {
    return false;
  }
  if (text == "input") {
    *bufferType = flagcxIpcBufferInput;
  } else if (text == "output") {
    *bufferType = flagcxIpcBufferOutput;
  } else {
    return false;
  }
  return true;
}

inline const char *uniRunnerDevicePrimToString(flagcxDevicePrim prim) {
  switch (prim) {
    case flagcxDevicePrimSend:
      return "send";
    case flagcxDevicePrimRecv:
      return "recv";
    case flagcxDevicePrimTerm:
      return "term";
    case flagcxDevicePrimWait:
      return "wait";
    case flagcxDevicePrimPut:
      return "put";
    case flagcxDevicePrimSignal:
      return "signal";
    case flagcxDevicePrimBarrierSignal:
      return "barrier_signal";
    case flagcxDevicePrimWaitSignal:
      return "wait_signal";
    case flagcxDevicePrimPutValue:
      return "put_value";
    case flagcxDevicePrimPutSignal:
      return "put_signal";
    case flagcxDevicePrimGet:
      return "get";
    default:
      return "unknown";
  }
}

inline bool uniRunnerDevicePrimFromString(const std::string &text,
                                          flagcxDevicePrim *prim) {
  if (text == "send") {
    *prim = flagcxDevicePrimSend;
  } else if (text == "recv") {
    *prim = flagcxDevicePrimRecv;
  } else if (text == "term") {
    *prim = flagcxDevicePrimTerm;
  } else if (text == "wait") {
    *prim = flagcxDevicePrimWait;
  } else if (text == "put") {
    *prim = flagcxDevicePrimPut;
  } else if (text == "signal") {
    *prim = flagcxDevicePrimSignal;
  } else if (text == "barrier_signal") {
    *prim = flagcxDevicePrimBarrierSignal;
  } else if (text == "wait_signal") {
    *prim = flagcxDevicePrimWaitSignal;
  } else if (text == "put_value") {
    *prim = flagcxDevicePrimPutValue;
  } else if (text == "put_signal") {
    *prim = flagcxDevicePrimPutSignal;
  } else if (text == "get") {
    *prim = flagcxDevicePrimGet;
  } else {
    return false;
  }
  return true;
}

inline Json uniRunnerDagBufferRefToJson(const uniRunnerDagBufferRef &ref) {
  return Json{{"buffer", uniRunnerDagBufferTypeToString(ref.bufferType)},
              {"offset_bytes", ref.offsetBytes}};
}

inline bool uniRunnerDagBufferRefFromJson(const Json &j,
                                          uniRunnerDagBufferRef *ref) {
  std::string bufferName = j.at("buffer").get<std::string>();
  if (!uniRunnerDagBufferTypeFromString(bufferName, &ref->bufferType)) {
    return false;
  }
  ref->offsetBytes = j.at("offset_bytes").get<int64_t>();
  return true;
}

inline Json uniRunnerDagCacheKeyToJson(const uniRunnerDagCacheKey &key) {
  return Json{
      // Persist uint64_t as decimal text so external JSON tooling cannot lose
      // precision by routing the value through an IEEE-754 number.
      {"algo_hash", std::to_string(key.algoHash)},
      {"comm_op", uniRunnerCommOpToString(key.commOp)},
      {"count", key.count},
      {"datatype", static_cast<int>(key.datatype)},
      {"red_op", static_cast<int>(key.redOp)},
      {"rank", key.rank},
      {"nranks", key.nranks},
      {"root", key.root},
  };
}

inline bool uniRunnerDagUint64FromJsonString(const Json &j, const char *field,
                                             uint64_t *value) {
  if (value == nullptr || !j.contains(field) || !j.at(field).is_string()) {
    return false;
  }
  try {
    const std::string encoded = j.at(field).get<std::string>();
    if (encoded.empty() ||
        encoded.find_first_not_of("0123456789") != std::string::npos) {
      return false;
    }
    size_t consumed = 0;
    const unsigned long long parsed = std::stoull(encoded, &consumed, 10);
    if (consumed != encoded.size()) {
      return false;
    }
    *value = static_cast<uint64_t>(parsed);
  } catch (...) {
    return false;
  }
  return true;
}

inline bool uniRunnerDagSizeFromJson(const Json &j, size_t *value) {
  if (value == nullptr) {
    return false;
  }
  try {
    if (j.is_number_unsigned()) {
      const uint64_t decoded = j.get<uint64_t>();
      if (decoded > std::numeric_limits<size_t>::max()) {
        return false;
      }
      *value = static_cast<size_t>(decoded);
      return true;
    }
    if (j.is_number_integer()) {
      const int64_t decoded = j.get<int64_t>();
      if (decoded < 0 ||
          static_cast<uint64_t>(decoded) >
              std::numeric_limits<size_t>::max()) {
        return false;
      }
      *value = static_cast<size_t>(decoded);
      return true;
    }
  } catch (...) {
    return false;
  }
  return false;
}

inline bool uniRunnerDagNonNegativeIntFromJson(const Json &j, int *value) {
  if (value == nullptr) {
    return false;
  }
  size_t decoded = 0;
  if (!uniRunnerDagSizeFromJson(j, &decoded) ||
      decoded > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *value = static_cast<int>(decoded);
  return true;
}

inline bool uniRunnerDagUint32FromJson(const Json &j, uint32_t *value) {
  if (value == nullptr) {
    return false;
  }
  size_t decoded = 0;
  if (!uniRunnerDagSizeFromJson(j, &decoded) ||
      decoded > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  *value = static_cast<uint32_t>(decoded);
  return true;
}

inline Json
uniRunnerDagIpcOpDescToJson(const uniRunnerDagIpcOpDesc &ipc) {
  return Json{
      {"src_offset_bytes", ipc.srcOffsetBytes},
      {"dst_offset_bytes", ipc.dstOffsetBytes},
      {"bytes", ipc.bytes},
      {"src_buffer_type",
       uniRunnerDagIpcBufferTypeToString(ipc.srcBufferType)},
      {"peer_local_rank", ipc.peerLocalRank},
      {"ready_slot", ipc.readySlot},
  };
}

inline bool uniRunnerDagIpcOpDescFromJson(const Json &j,
                                          uniRunnerDagIpcOpDesc *ipc) {
  if (ipc == nullptr || !j.is_object() || j.size() != 6 ||
      !j.contains("src_offset_bytes") || !j.contains("dst_offset_bytes") ||
      !j.contains("bytes") || !j.contains("src_buffer_type") ||
      !j.contains("peer_local_rank") || !j.contains("ready_slot")) {
    return false;
  }
  try {
    const std::string srcBufferType =
        j.at("src_buffer_type").get<std::string>();
    return uniRunnerDagSizeFromJson(j.at("src_offset_bytes"),
                                    &ipc->srcOffsetBytes) &&
           uniRunnerDagSizeFromJson(j.at("dst_offset_bytes"),
                                    &ipc->dstOffsetBytes) &&
           uniRunnerDagSizeFromJson(j.at("bytes"), &ipc->bytes) &&
           uniRunnerDagIpcBufferTypeFromString(srcBufferType,
                                               &ipc->srcBufferType) &&
           uniRunnerDagNonNegativeIntFromJson(j.at("peer_local_rank"),
                                               &ipc->peerLocalRank) &&
           uniRunnerDagUint32FromJson(j.at("ready_slot"), &ipc->readySlot);
  } catch (...) {
    return false;
  }
}

inline bool uniRunnerDagCacheKeyFromJson(const Json &j,
                                         uniRunnerDagCacheKey *key) {
  if (key == nullptr || !j.is_object()) {
    return false;
  }
  try {
    const std::string commOpName = j.at("comm_op").get<std::string>();
    if (!uniRunnerCommOpFromString(commOpName, &key->commOp) ||
        !uniRunnerDagUint64FromJsonString(j, "algo_hash", &key->algoHash)) {
      return false;
    }
    if (!uniRunnerDagSizeFromJson(j.at("count"), &key->count)) {
      return false;
    }
    const int datatype = j.at("datatype").get<int>();
    const int redOp = j.at("red_op").get<int>();
    if (!uniRunnerDagDataTypeValueValid(datatype) ||
        !uniRunnerDagRedOpValueValid(redOp, true)) {
      return false;
    }
    key->datatype = static_cast<flagcxDataType_t>(datatype);
    key->redOp = static_cast<flagcxRedOp_t>(redOp);
    key->rank = j.at("rank").get<int>();
    key->nranks = j.at("nranks").get<int>();
    key->root = j.at("root").get<int>();
  } catch (...) {
    return false;
  }
  return true;
}

inline Json
uniRunnerDagTemplateToJson(const uniRunnerDagTemplate &dagTemplate) {
  size_t hashValue = getUniRunnerDagPatternHash(dagTemplate.key);
  Json nodes = Json::array();
  for (const uniRunnerDagNodeDesc &node : dagTemplate.nodes) {
    Json nodeJson{
        {"node_idx", node.nodeIdx},
        {"node_type", uniRunnerDagNodeTypeToString(node.nodeType)},
        {"parents", node.parents},
        {"children", node.children},
    };
    if (node.nodeType == uniRunnerDagNodeTypeP2p) {
      Json ops = Json::array();
      for (const uniRunnerDagP2pOpDesc &op : node.p2pOps) {
        ops.push_back(Json{
            {"type", uniRunnerDevicePrimToString(op.type)},
            {"peer_rank", op.peerRank},
            {"count", op.count},
            {"datatype", static_cast<int>(op.datatype)},
            {"buffer", uniRunnerDagBufferRefToJson(op.buffer)},
        });
      }
      nodeJson["p2p_ops"] = ops;
    } else if (node.nodeType == uniRunnerDagNodeTypeRed) {
      nodeJson["red"] = Json{
          {"input1", uniRunnerDagBufferRefToJson(node.red.input1)},
          {"input2", uniRunnerDagBufferRefToJson(node.red.input2)},
          {"output", uniRunnerDagBufferRefToJson(node.red.output)},
          {"count", node.red.count},
          {"datatype", static_cast<int>(node.red.datatype)},
          {"red_op", static_cast<int>(node.red.redOp)},
      };
    } else if (node.nodeType == uniRunnerDagNodeTypeCpy) {
      nodeJson["cpy"] = Json{
          {"src", uniRunnerDagBufferRefToJson(node.cpy.src)},
          {"dst", uniRunnerDagBufferRefToJson(node.cpy.dst)},
          {"count", node.cpy.count},
          {"datatype", static_cast<int>(node.cpy.datatype)},
      };
    } else if (node.nodeType == uniRunnerDagNodeTypeIpc) {
      nodeJson["ipc"] = uniRunnerDagIpcOpDescToJson(node.ipc);
    }
    nodes.push_back(nodeJson);
  }

  return Json{
      {"format_version", kUniRunnerDagCacheFormatVersion},
      {"hash", std::to_string(hashValue)},
      {"key", uniRunnerDagCacheKeyToJson(dagTemplate.key)},
      {"dag", Json{{"num_nodes", dagTemplate.nodes.size()}, {"nodes", nodes}}},
  };
}

inline bool uniRunnerDagTemplateFromJson(const Json &j,
                                         uniRunnerDagTemplate *dagTemplate) {
  if (dagTemplate == nullptr) {
    return false;
  }
  try {
    if (!j.is_object() || !j.contains("format_version") ||
        !j.at("format_version").is_number_integer() ||
        j.at("format_version").get<int>() !=
            kUniRunnerDagCacheFormatVersion) {
      return false;
    }
    if (!uniRunnerDagCacheKeyFromJson(j.at("key"), &dagTemplate->key)) {
      return false;
    }
    size_t computedHash = getUniRunnerDagPatternHash(dagTemplate->key);
    if (j.contains("hash")) {
      uint64_t encodedHashValue = 0;
      if (!uniRunnerDagUint64FromJsonString(j, "hash", &encodedHashValue) ||
          encodedHashValue > std::numeric_limits<size_t>::max()) {
        return false;
      }
      const size_t encodedHash = static_cast<size_t>(encodedHashValue);
      if (encodedHash != computedHash) {
        return false;
      }
    }
    const Json &dagJson = j.at("dag");
    if (!dagJson.is_object() || !dagJson.contains("num_nodes") ||
        !dagJson.contains("nodes") || !dagJson.at("nodes").is_array()) {
      return false;
    }
    size_t declaredNumNodes = 0;
    const Json &numNodesJson = dagJson.at("num_nodes");
    if (numNodesJson.is_number_unsigned()) {
      const uint64_t value = numNodesJson.get<uint64_t>();
      if (value >
          static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return false;
      }
      declaredNumNodes = static_cast<size_t>(value);
    } else if (numNodesJson.is_number_integer()) {
      const int64_t value = numNodesJson.get<int64_t>();
      if (value < 0 || value > std::numeric_limits<int>::max()) {
        return false;
      }
      declaredNumNodes = static_cast<size_t>(value);
    } else {
      return false;
    }

    const Json &nodes = dagJson.at("nodes");
    if (nodes.size() != declaredNumNodes) {
      return false;
    }
    dagTemplate->hashValue = computedHash;
    dagTemplate->nodes.clear();
    dagTemplate->nodes.reserve(declaredNumNodes);

    const size_t maxInt =
        static_cast<size_t>(std::numeric_limits<int>::max());
    for (const Json &nodeJson : nodes) {
      uniRunnerDagNodeDesc node;
      std::string nodeType = nodeJson.at("node_type").get<std::string>();
      if (!uniRunnerDagNodeTypeFromString(nodeType, &node.nodeType)) {
        return false;
      }
      node.nodeIdx = nodeJson.at("node_idx").get<int>();
      if (!nodeJson.at("parents").is_array() ||
          !nodeJson.at("children").is_array() ||
          nodeJson.at("parents").size() > maxInt ||
          nodeJson.at("children").size() > maxInt) {
        return false;
      }
      node.parents = nodeJson.at("parents").get<std::vector<int>>();
      node.children = nodeJson.at("children").get<std::vector<int>>();

      if (node.nodeType == uniRunnerDagNodeTypeP2p) {
        const Json &opsJson = nodeJson.at("p2p_ops");
        if (!opsJson.is_array() || opsJson.size() > maxInt) {
          return false;
        }
        node.p2pOps.reserve(opsJson.size());
        for (const Json &opJson : opsJson) {
          uniRunnerDagP2pOpDesc op;
          std::string primType = opJson.at("type").get<std::string>();
          if (!uniRunnerDevicePrimFromString(primType, &op.type) ||
              !uniRunnerDagBufferRefFromJson(opJson.at("buffer"),
                                             &op.buffer)) {
            return false;
          }
          op.peerRank = opJson.at("peer_rank").get<int>();
          if (!uniRunnerDagSizeFromJson(opJson.at("count"), &op.count)) {
            return false;
          }
          const int datatype = opJson.at("datatype").get<int>();
          if (!uniRunnerDagDataTypeValueValid(datatype)) {
            return false;
          }
          op.datatype = static_cast<flagcxDataType_t>(datatype);
          node.p2pOps.push_back(op);
        }
      } else if (node.nodeType == uniRunnerDagNodeTypeRed) {
        const Json &redJson = nodeJson.at("red");
        if (!uniRunnerDagBufferRefFromJson(redJson.at("input1"),
                                           &node.red.input1) ||
            !uniRunnerDagBufferRefFromJson(redJson.at("input2"),
                                           &node.red.input2) ||
            !uniRunnerDagBufferRefFromJson(redJson.at("output"),
                                           &node.red.output)) {
          return false;
        }
        if (!uniRunnerDagSizeFromJson(redJson.at("count"),
                                      &node.red.count)) {
          return false;
        }
        const int datatype = redJson.at("datatype").get<int>();
        const int redOp = redJson.at("red_op").get<int>();
        if (!uniRunnerDagDataTypeValueValid(datatype) ||
            !uniRunnerDagRedOpValueValid(redOp, false)) {
          return false;
        }
        node.red.datatype = static_cast<flagcxDataType_t>(datatype);
        node.red.redOp = static_cast<flagcxRedOp_t>(redOp);
      } else if (node.nodeType == uniRunnerDagNodeTypeCpy) {
        const Json &cpyJson = nodeJson.at("cpy");
        if (!uniRunnerDagBufferRefFromJson(cpyJson.at("src"), &node.cpy.src) ||
            !uniRunnerDagBufferRefFromJson(cpyJson.at("dst"), &node.cpy.dst)) {
          return false;
        }
        if (!uniRunnerDagSizeFromJson(cpyJson.at("count"),
                                      &node.cpy.count)) {
          return false;
        }
        const int datatype = cpyJson.at("datatype").get<int>();
        if (!uniRunnerDagDataTypeValueValid(datatype)) {
          return false;
        }
        node.cpy.datatype = static_cast<flagcxDataType_t>(datatype);
      } else if (node.nodeType == uniRunnerDagNodeTypeIpc) {
        // IPC nodes have exactly the four common DAG fields plus their six-field
        // structural payload. Rejecting extras prevents runtime trigger state
        // from silently becoming part of the persistent cache schema.
        if (nodeJson.size() != 5 || !nodeJson.contains("node_idx") ||
            !nodeJson.contains("node_type") ||
            !nodeJson.contains("parents") ||
            !nodeJson.contains("children") || !nodeJson.contains("ipc") ||
            !uniRunnerDagIpcOpDescFromJson(nodeJson.at("ipc"), &node.ipc)) {
          return false;
        }
      }
      dagTemplate->nodes.push_back(node);
    }
  } catch (...) {
    return false;
  }
  return true;
}

inline Json
uniRunnerSerializeDagTemplate(const uniRunnerDagTemplate &dagTemplate) {
  return uniRunnerDagTemplateToJson(dagTemplate);
}

inline bool uniRunnerDeserializeDagTemplate(const Json &j,
                                            uniRunnerDagTemplate *dagTemplate) {
  return uniRunnerDagTemplateFromJson(j, dagTemplate);
}

inline Json uniRunnerSerializeDagCacheFile(
    const std::vector<uniRunnerDagTemplate> &dagTemplates) {
  Json entries = Json::array();
  for (const uniRunnerDagTemplate &dagTemplate : dagTemplates) {
    entries.push_back(uniRunnerSerializeDagTemplate(dagTemplate));
  }

  return Json{{"format_version", kUniRunnerDagCacheFormatVersion},
              {"address_model", "buffer_kind+offset_bytes"},
              {"buffer_kinds",
               Json::array({Json("input"), Json("output"), Json("scratch")})},
              {"entries", entries}};
}

inline bool uniRunnerDeserializeDagCacheFile(
    const Json &root, std::vector<uniRunnerDagTemplate> *dagTemplates) {
  if (dagTemplates == nullptr) {
    return false;
  }
  try {
    if (!root.is_object() || !root.contains("format_version") ||
        !root.at("format_version").is_number_integer() ||
        root.at("format_version").get<int>() !=
            kUniRunnerDagCacheFormatVersion) {
      return false;
    }
    if (!root.contains("entries") || !root["entries"].is_array()) {
      return false;
    }

    dagTemplates->clear();
    for (const Json &entryJson : root["entries"]) {
      uniRunnerDagTemplate dagTemplate;
      if (!uniRunnerDeserializeDagTemplate(entryJson, &dagTemplate)) {
        return false;
      }
      dagTemplates->push_back(std::move(dagTemplate));
    }
  } catch (...) {
    return false;
  }
  return true;
}

inline bool uniRunnerDeserializeDagJsonDocument(
    const Json &root, std::vector<uniRunnerDagTemplate> *dagTemplates) {
  if (!root.is_object()) {
    return false;
  }
  if (root.contains("entries")) {
    return uniRunnerDeserializeDagCacheFile(root, dagTemplates);
  }

  uniRunnerDagTemplate dagTemplate;
  if (!uniRunnerDeserializeDagTemplate(root, &dagTemplate)) {
    return false;
  }
  dagTemplates->clear();
  dagTemplates->push_back(std::move(dagTemplate));
  return true;
}

inline flagcxResult_t uniRunnerLoadJsonFile(const std::string &path,
                                            Json *root) {
  std::ifstream input(path.c_str());
  if (!input.good()) {
    return flagcxSystemError;
  }

  try {
    input >> *root;
  } catch (...) {
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

inline flagcxResult_t uniRunnerSaveJsonFile(const std::string &path,
                                            const Json &root) {
  if (path.empty()) {
    return flagcxSuccess;
  }

  std::string tmpPath = path + ".tmp";
  std::ofstream output(tmpPath.c_str(), std::ios::out | std::ios::trunc);
  if (!output.good()) {
    return flagcxSystemError;
  }
  output << root.dump(2);
  output.close();
  if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

inline flagcxResult_t
uniRunnerLoadDagJsonFile(const std::string &path,
                         std::vector<uniRunnerDagTemplate> *dagTemplates) {
  Json root;
  FLAGCXCHECK(uniRunnerLoadJsonFile(path, &root));
  return uniRunnerDeserializeDagJsonDocument(root, dagTemplates)
             ? flagcxSuccess
             : flagcxInternalError;
}

inline flagcxResult_t uniRunnerLoadDagJsonFileIfExists(
    const std::string &path, std::vector<uniRunnerDagTemplate> *dagTemplates) {
  flagcxResult_t loadRes = uniRunnerLoadDagJsonFile(path, dagTemplates);
  if (loadRes == flagcxSystemError) {
    dagTemplates->clear();
    return flagcxSuccess;
  }
  return loadRes;
}

inline flagcxResult_t
uniRunnerSaveDagJsonFile(const std::string &path,
                         const uniRunnerDagTemplate &dagTemplate) {
  return uniRunnerSaveJsonFile(path,
                               uniRunnerSerializeDagTemplate(dagTemplate));
}

inline flagcxResult_t uniRunnerSaveDagJsonCollectionFile(
    const std::string &path,
    const std::vector<uniRunnerDagTemplate> &dagTemplates) {
  return uniRunnerSaveJsonFile(path,
                               uniRunnerSerializeDagCacheFile(dagTemplates));
}

inline flagcxResult_t
uniRunnerLoadDagCacheFile(const std::string &cachePath,
                          std::vector<uniRunnerDagTemplate> *dagTemplates) {
  return uniRunnerLoadDagJsonFileIfExists(cachePath, dagTemplates);
}

inline flagcxResult_t uniRunnerSaveDagCacheFile(
    const std::string &cachePath,
    const std::vector<uniRunnerDagTemplate> &dagTemplates) {
  return uniRunnerSaveDagJsonCollectionFile(cachePath, dagTemplates);
}

#endif // FLAGCX_UNIRUNNER_HELPER_H_
