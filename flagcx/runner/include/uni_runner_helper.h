#ifndef FLAGCX_UNIRUNNER_HELPER_H_
#define FLAGCX_UNIRUNNER_HELPER_H_

#include "uni_runner_impl.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

using Json = nlohmann::json;

inline constexpr int kUniRunnerDagCacheFormatVersion = 1;

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
  size_t nthreads = 0;
  flagcxDataType_t datatype = flagcxInt8;
  flagcxRedOp_t redOp = flagcxRedNoOp;
};

struct uniRunnerDagCpyOpDesc {
  uniRunnerDagBufferRef src;
  uniRunnerDagBufferRef dst;
  size_t count = 0;
  flagcxDataType_t datatype = flagcxInt8;
};

struct uniRunnerDagNodeDesc {
  uniRunnerDagNodeType nodeType = uniRunnerDagNodeTypeP2p;
  int nodeIdx = 0;
  std::vector<int> parents;
  std::vector<int> children;
  std::vector<uniRunnerDagP2pOpDesc> p2pOps;
  uniRunnerDagRedOpDesc red;
  uniRunnerDagCpyOpDesc cpy;
};

struct uniRunnerDagTemplate {
  uniRunnerDagCacheKey key = {};
  size_t hashValue = 0;
  std::vector<uniRunnerDagNodeDesc> nodes;
};

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
      {"format_version", key.formatVersion},
      {"algo", uniRunnerDagAlgoTypeToString(key.algoType)},
      {"comm_op", uniRunnerCommOpToString(key.commOp)},
      {"count", key.count},
      {"datatype", static_cast<int>(key.datatype)},
      {"red_op", static_cast<int>(key.redOp)},
      {"rank", key.rank},
      {"nranks", key.nranks},
      {"root", key.root},
      {"group_size", key.groupSize},
      {"num_slices", key.numSlices},
      {"num_red_slices", key.numRedSlices},
      {"red_slice_size", key.redSliceSize},
      {"nthreads", key.nthreads},
      {"input_output_aliased", key.inputOutputAliased},
      {"input_scratch_aliased", key.inputScratchAliased},
      {"output_scratch_aliased", key.outputScratchAliased},
  };
}

inline bool uniRunnerDagCacheKeyFromJson(const Json &j,
                                         uniRunnerDagCacheKey *key) {
  std::string algoName = j.at("algo").get<std::string>();
  std::string commOpName = j.at("comm_op").get<std::string>();
  if (!uniRunnerDagAlgoTypeFromString(algoName, &key->algoType) ||
      !uniRunnerCommOpFromString(commOpName, &key->commOp)) {
    return false;
  }
  key->formatVersion = j.at("format_version").get<int>();
  key->count = j.at("count").get<size_t>();
  key->datatype = static_cast<flagcxDataType_t>(j.at("datatype").get<int>());
  key->redOp = static_cast<flagcxRedOp_t>(j.at("red_op").get<int>());
  key->rank = j.at("rank").get<int>();
  key->nranks = j.at("nranks").get<int>();
  key->root = j.at("root").get<int>();
  key->groupSize = j.at("group_size").get<int>();
  key->numSlices = j.at("num_slices").get<uint64_t>();
  key->numRedSlices = j.at("num_red_slices").get<uint64_t>();
  key->redSliceSize = j.at("red_slice_size").get<uint64_t>();
  key->nthreads = j.at("nthreads").get<uint64_t>();
  key->inputOutputAliased = j.at("input_output_aliased").get<int>();
  key->inputScratchAliased = j.at("input_scratch_aliased").get<int>();
  key->outputScratchAliased = j.at("output_scratch_aliased").get<int>();
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
          {"nthreads", node.red.nthreads},
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
    }
    nodes.push_back(nodeJson);
  }

  return Json{
      {"hash", std::to_string(hashValue)},
      {"key", uniRunnerDagCacheKeyToJson(dagTemplate.key)},
      {"dag", Json{{"num_nodes", dagTemplate.nodes.size()}, {"nodes", nodes}}},
  };
}

inline bool uniRunnerDagTemplateFromJson(const Json &j,
                                         uniRunnerDagTemplate *dagTemplate) {
  if (!uniRunnerDagCacheKeyFromJson(j.at("key"), &dagTemplate->key)) {
    return false;
  }
  size_t computedHash = getUniRunnerDagPatternHash(dagTemplate->key);
  if (j.contains("hash")) {
    size_t encodedHash =
        static_cast<size_t>(std::stoull(j.at("hash").get<std::string>()));
    if (encodedHash != computedHash) {
      return false;
    }
  }
  dagTemplate->hashValue = computedHash;
  dagTemplate->nodes.clear();

  const Json &nodes = j.at("dag").at("nodes");
  for (const Json &nodeJson : nodes) {
    uniRunnerDagNodeDesc node;
    std::string nodeType = nodeJson.at("node_type").get<std::string>();
    if (!uniRunnerDagNodeTypeFromString(nodeType, &node.nodeType)) {
      return false;
    }
    node.nodeIdx = nodeJson.at("node_idx").get<int>();
    node.parents = nodeJson.at("parents").get<std::vector<int>>();
    node.children = nodeJson.at("children").get<std::vector<int>>();

    if (node.nodeType == uniRunnerDagNodeTypeP2p) {
      for (const Json &opJson : nodeJson.at("p2p_ops")) {
        uniRunnerDagP2pOpDesc op;
        std::string primType = opJson.at("type").get<std::string>();
        if (!uniRunnerDevicePrimFromString(primType, &op.type) ||
            !uniRunnerDagBufferRefFromJson(opJson.at("buffer"), &op.buffer)) {
          return false;
        }
        op.peerRank = opJson.at("peer_rank").get<int>();
        op.count = opJson.at("count").get<size_t>();
        op.datatype =
            static_cast<flagcxDataType_t>(opJson.at("datatype").get<int>());
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
      node.red.count = redJson.at("count").get<size_t>();
      node.red.nthreads = redJson.at("nthreads").get<size_t>();
      node.red.datatype =
          static_cast<flagcxDataType_t>(redJson.at("datatype").get<int>());
      node.red.redOp =
          static_cast<flagcxRedOp_t>(redJson.at("red_op").get<int>());
    } else if (node.nodeType == uniRunnerDagNodeTypeCpy) {
      const Json &cpyJson = nodeJson.at("cpy");
      if (!uniRunnerDagBufferRefFromJson(cpyJson.at("src"), &node.cpy.src) ||
          !uniRunnerDagBufferRefFromJson(cpyJson.at("dst"), &node.cpy.dst)) {
        return false;
      }
      node.cpy.count = cpyJson.at("count").get<size_t>();
      node.cpy.datatype =
          static_cast<flagcxDataType_t>(cpyJson.at("datatype").get<int>());
    }
    dagTemplate->nodes.push_back(node);
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
  if (root.contains("format_version") &&
      root["format_version"].get<int>() != kUniRunnerDagCacheFormatVersion) {
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
  flagcxResult_t loadRes = uniRunnerLoadDagJsonFile(cachePath, dagTemplates);
  if (loadRes == flagcxSystemError) {
    dagTemplates->clear();
    return flagcxSuccess;
  }
  return loadRes;
}

inline flagcxResult_t uniRunnerSaveDagCacheFile(
    const std::string &cachePath,
    const std::vector<uniRunnerDagTemplate> &dagTemplates) {
  return uniRunnerSaveDagJsonCollectionFile(cachePath, dagTemplates);
}

#endif // FLAGCX_UNIRUNNER_HELPER_H_
