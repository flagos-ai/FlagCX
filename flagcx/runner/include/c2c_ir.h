#ifndef FLAGCX_C2C_IR_H_
#define FLAGCX_C2C_IR_H_

#include "adaptor.h"
#include "c2c_algo.h"
#include "collectives.h"
#include "flagcx.h"
#include "group.h"
#include "param.h"
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <unordered_map>

// Max line length for reading xml
#define LINE_LEN 512

size_t genC2cAlgoHash(size_t sendCount, size_t recvCount, size_t rootClusterId,
                      flagcxCommOp_t commOp, flagcxRedOp_t redOp) {
  std::size_t h1 = std::hash<size_t>()(sendCount);
  std::size_t h2 = std::hash<size_t>()(recvCount);
  std::size_t h3 = std::hash<size_t>()(rootClusterId);
  std::size_t h4 = std::hash<size_t>()(commOp);
  std::size_t h5 = std::hash<size_t>()(redOp);
  return static_cast<size_t>(h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^
                             (h5 << 4));
}

inline bool readTagValue(char *line, const char *tag, char *value) {
  size_t len = strlen(tag);
  char openTag[64], closeTag[64];
  sprintf(openTag, "<%s>", tag);
  sprintf(closeTag, "</%s>", tag);

  char *start = strstr(line, openTag);
  char *end = strstr(line, closeTag);
  if (start && end) {
    start += len + 2; // skip over <tag>
    *end = '\0';      // terminate at </tag>
    strcpy(value, start);
    return true;
  }

  return false;
}

inline int readIntTag(char *line, const char *tag) {
  char buf[LINE_LEN];
  if (!readTagValue(line, tag, buf))
    return 0;
  return atoi(buf);
}

inline size_t readSizeTag(char *line, const char *tag) {
  char buf[LINE_LEN];
  if (!readTagValue(line, tag, buf))
    return 0;
  return strtoull(buf, nullptr, 10);
}

inline void serializRefreshFunc(FILE *file, size_t chunksize,
                                const flagcxC2cRefreshFunc &func,
                                int indent = 2) {
  fprintf(file, "%*s<RefreshFunc>\n", indent, "");
  fprintf(file, "%*s<buffType>%d</buffType>\n", indent + 2, "", func.bufftype_);
  fprintf(file, "%*s<start>%zu</start>\n", indent + 2, "",
          func.start_ / chunksize);
  fprintf(file, "%*s<offset>%zu</offset>\n", indent + 2, "",
          func.offset_ / chunksize);
  fprintf(file, "%*s<count>%zu</count>\n", indent + 2, "",
          func.count_ / chunksize);
  fprintf(file, "%*s<totalCount>%zu</totalCount>\n", indent + 2, "",
          func.totalCount_ / chunksize);
  fprintf(file, "%*s<redOp>%d</redOp>\n", indent + 2, "", func.redOp_);
  fprintf(file, "%*s</RefreshFunc>\n", indent, "");
}

inline void serializeHomoFunc(FILE *file, size_t chunksize,
                              const flagcxC2cHomoFunc &func, int indent = 4) {
  fprintf(file, "%*s<HomoFunc>\n", indent, "");
  fprintf(file, "%*s<rootRank>%d</rootRank>\n", indent + 2, "", func.rootRank_);
  fprintf(file, "%*s<sendType>%d</sendType>\n", indent + 2, "", func.sendType_);
  fprintf(file, "%*s<recvType>%d</recvType>\n", indent + 2, "", func.recvType_);
  fprintf(file, "%*s<sendOffset>%zu</sendOffset>\n", indent + 2, "",
          func.sendOffset_ / chunksize);
  fprintf(file, "%*s<recvOffset>%zu</recvOffset>\n", indent + 2, "",
          func.recvOffset_ / chunksize);
  fprintf(file, "%*s<count>%zu</count>\n", indent + 2, "",
          func.count_ / chunksize);
  fprintf(file, "%*s<homoType>%d</homoType>\n", indent + 2, "", func.homoType_);
  fprintf(file, "%*s<commOp>%d</commOp>\n", indent + 2, "", func.commOp_);
  fprintf(file, "%*s</HomoFunc>\n", indent, "");
}

inline void serializeP2pOp(FILE *file, size_t chunksize,
                           const flagcxC2cP2pOp &op, int indent = 6) {
  fprintf(file, "%*s<P2pOp>\n", indent, "");
  fprintf(file, "%*s<rank>%d</rank>\n", indent + 2, "", op.rank_);
  fprintf(file, "%*s<peerRank>%d</peerRank>\n", indent + 2, "", op.peerRank_);
  fprintf(file, "%*s<offset>%zu</offset>\n", indent + 2, "",
          op.offset_ / chunksize);
  fprintf(file, "%*s<count>%zu</count>\n", indent + 2, "",
          op.count_ / chunksize);
  fprintf(file, "%*s<isRecv>%d</isRecv>\n", indent + 2, "", op.isRecv_);
  fprintf(file, "%*s</P2pOp>\n", indent, "");
}

inline void serializeHeteroFunc(FILE *file, size_t chunksize,
                                const flagcxC2cHeteroFunc &func,
                                int indent = 4) {
  fprintf(file, "%*s<HeteroFunc>\n", indent, "");
  for (const auto &op : func.p2pOps_) {
    serializeP2pOp(file, chunksize, op, indent + 2);
  }
  fprintf(file, "%*s</HeteroFunc>\n", indent, "");
}

template <typename T>
void serializeFunc2DVector(FILE *file, size_t chunksize,
                           const std::vector<std::vector<T>> &steps,
                           const char *tagName, int indent = 2) {
  fprintf(file, "%*s<%s>\n", indent, "", tagName);
  for (const auto &stepVec : steps) {
    if (stepVec.size() == 0) {
      fprintf(file, "%*s<Step/>\n", indent + 2, "");
      continue;
    }
    fprintf(file, "%*s<Step>\n", indent + 2, "");
    for (const auto &func : stepVec) {
      if constexpr (std::is_same<T, flagcxC2cHomoFunc>::value) {
        serializeHomoFunc(file, chunksize, func, indent + 4);
      } else if constexpr (std::is_same<T, flagcxC2cHeteroFunc>::value) {
        serializeHeteroFunc(file, chunksize, func, indent + 4);
      }
    }
    fprintf(file, "%*s</Step>\n", indent + 2, "");
  }
  fprintf(file, "%*s</%s>\n", indent, "", tagName);
}

template <typename T>
std::vector<std::vector<T>> readFunc2DVector(FILE *file, size_t chunksize,
                                             const char *tagName) {
  std::vector<std::vector<T>> result;
  char line[LINE_LEN];

  char openTag[64], closeTag[64];
  sprintf(openTag, "<%s>", tagName);
  sprintf(closeTag, "</%s>", tagName);

  while (fgets(line, sizeof(line), file)) {
    if (strstr(line, closeTag))
      break;
    if (strstr(line, "<Step/>")) {
      std::vector<T> step;
      result.push_back(step);
    }
    if (strstr(line, "<Step>")) {
      std::vector<T> step;
      while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "</Step>"))
          break;
        if constexpr (std::is_same<T, flagcxC2cHomoFunc>::value) {
          if (strstr(line, "<HomoFunc>"))
            step.emplace_back(file, chunksize);
        } else if constexpr (std::is_same<T, flagcxC2cHeteroFunc>::value) {
          if (strstr(line, "<HeteroFunc>"))
            step.emplace_back(file, chunksize);
        }
      }
      result.push_back(step);
    }
  }
  return result;
}

#endif
