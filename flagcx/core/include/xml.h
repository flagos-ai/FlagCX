/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_XML_H_
#define FLAGCX_XML_H_

#include "core.h"

#define MAX_STR_LEN 255
#define MAX_ATTR_COUNT 16
#define MAX_SUBS 128

#define NODE_TYPE_NONE 0
#define NODE_TYPE_OPEN 1
#define NODE_TYPE_CLOSE 2
#define NODE_TYPE_SINGLE 3

struct flagcxXmlNode {
  char name[MAX_STR_LEN + 1];
  struct {
    char key[MAX_STR_LEN + 1];
    char value[MAX_STR_LEN + 1];
  } attrs[MAX_ATTR_COUNT + 1]; // Need an extra one to consume extra params
  int nAttrs;
  int type;
  struct flagcxXmlNode *parent;
  struct flagcxXmlNode *subs[MAX_SUBS];
  int nSubs;
};

struct flagcxXml {
  int maxIndex, maxNodes;
  struct flagcxXmlNode nodes[1];
};

#define FLAGCX_TOPO_XML_VERSION 1

// read a xml file and convert it into a flagcxXml structure
// TODO: implement the function
flagcxResult_t flagcxTopoGetXmlFromFile(const char *xmlTopoFile,
                                        struct flagcxXml *xml, int warn);
// convert a flagcxXml structure into a xml file
// TODO: implement the function
flagcxResult_t flagcxTopoDumpXmlToFile(const char *xmlTopoFile,
                                       struct flagcxXml *xml);

// auto-detect functions
// TODO: implement the following 2 functions
flagcxResult_t flagcxTopoFillApu(struct flagcxXml *xml, const char *busId,
                                 struct flagcxXmlNode **gpuNode);
flagcxResult_t flagcxTopoFillNet(struct flagcxXml *xml, const char *pciPath,
                                 const char *netName,
                                 struct flagcxXmlNode **netNode);

flagcxResult_t xmlGetApuByIndex(struct flagcxXml *xml, int apu,
                                struct flagcxXmlNode **apuNode);
flagcxResult_t xmlFindClosestNetUnderCpu(struct flagcxXml *xml,
                                         struct flagcxXmlNode *apuNode,
                                         struct flagcxXmlNode **retNet);
flagcxResult_t xmlFindClosestNetUnderServer(struct flagcxXml *xml,
                                            struct flagcxXmlNode *apuNode,
                                            struct flagcxXmlNode **retNet);

static size_t xmlMemSize(int maxNodes) {
  return offsetof(struct flagcxXml, nodes) +
         sizeof(struct flagcxXmlNode) * maxNodes;
}
static flagcxResult_t xmlAlloc(struct flagcxXml **xml, int maxNodes) {
  char *mem;
  FLAGCXCHECK(flagcxCalloc(&mem, xmlMemSize(maxNodes)));
  *xml = (struct flagcxXml *)mem;
  (*xml)->maxNodes = maxNodes;
  return flagcxSuccess;
}

static flagcxResult_t xmlGetAttrIndex(struct flagcxXmlNode *node,
                                      const char *attrName, int *index) {
  *index = -1;
  const int nAttrs = node->nAttrs;
  for (int a = 0; a < nAttrs; a++) {
    if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN) == 0) {
      *index = a;
      return flagcxSuccess;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t xmlGetAttr(struct flagcxXmlNode *node,
                                 const char *attrName, const char **value) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  *value = index == -1 ? NULL : node->attrs[index].value;
  return flagcxSuccess;
}

static flagcxResult_t xmlGetAttrStr(struct flagcxXmlNode *node,
                                    const char *attrName, const char **value) {
  FLAGCXCHECK(xmlGetAttr(node, attrName, value));
  if (*value == NULL) {
    WARN("Attribute %s of node %s not found", attrName, node->name);
    return flagcxInternalError;
  }
  return flagcxSuccess;
}
static flagcxResult_t xmlGetAttrInt(struct flagcxXmlNode *node,
                                    const char *attrName, int *value) {
  const char *str;
  FLAGCXCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return flagcxSuccess;
}

static flagcxResult_t xmlGetAttrIntDefault(struct flagcxXmlNode *node,
                                           const char *attrName, int *value,
                                           int defaultValue) {
  const char *str;
  FLAGCXCHECK(xmlGetAttr(node, attrName, &str));
  *value = str ? strtol(str, NULL, 0) : defaultValue;
  return flagcxSuccess;
}

static flagcxResult_t xmlGetAttrLong(struct flagcxXmlNode *node,
                                     const char *attrName, int64_t *value) {
  const char *str;
  FLAGCXCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return flagcxSuccess;
}

static flagcxResult_t xmlGetAttrFloat(struct flagcxXmlNode *node,
                                      const char *attrName, float *value) {
  const char *str;
  FLAGCXCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtof(str, NULL);
  return flagcxSuccess;
}

static flagcxResult_t xmlFindTag(struct flagcxXml *xml, const char *tagName,
                                 struct flagcxXmlNode **node) {
  *node = NULL;
  for (int i = 0; i < xml->maxIndex; i++) {
    struct flagcxXmlNode *n = xml->nodes + i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return flagcxSuccess;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t xmlFindNextTag(struct flagcxXml *xml, const char *tagName,
                                     struct flagcxXmlNode *prev,
                                     struct flagcxXmlNode **node) {
  *node = NULL;
  for (int i = prev - xml->nodes + 1; i < xml->maxIndex; i++) {
    struct flagcxXmlNode *n = xml->nodes + i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return flagcxSuccess;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t xmlFindTagKv(struct flagcxXml *xml, const char *tagName,
                                   struct flagcxXmlNode **node,
                                   const char *attrName,
                                   const char *attrValue) {
  *node = NULL;
  for (int i = 0; i < xml->maxIndex; i++) {
    struct flagcxXmlNode *n = xml->nodes + i;
    if (strcmp(n->name, tagName) == 0) {
      const char *value;
      FLAGCXCHECK(xmlGetAttr(n, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *node = n;
        return flagcxSuccess;
      }
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t xmlSetAttr(struct flagcxXmlNode *node,
                                 const char *attrName, const char *value) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return flagcxSuccess;
}

static flagcxResult_t xmlSetAttrIfUnset(struct flagcxXmlNode *node,
                                        const char *attrName,
                                        const char *value) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index != -1)
    return flagcxSuccess;
  index = node->nAttrs++;
  strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  node->attrs[index].key[MAX_STR_LEN] = '\0';
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return flagcxSuccess;
}

static flagcxResult_t xmlSetAttrInt(struct flagcxXmlNode *node,
                                    const char *attrName, const int value) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return flagcxSuccess;
}

static flagcxResult_t xmlSetAttrFloat(struct flagcxXmlNode *node,
                                      const char *attrName, const float value) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%g", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return flagcxSuccess;
}

static flagcxResult_t xmlSetAttrLong(struct flagcxXmlNode *node,
                                     const char *attrName,
                                     const int64_t value) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%#lx", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return flagcxSuccess;
}

static flagcxResult_t xmlInitAttrUint64(struct flagcxXmlNode *node,
                                        const char *attrName,
                                        const uint64_t value) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return flagcxSuccess;
}

static flagcxResult_t xmlUnsetAttr(struct flagcxXmlNode *node,
                                   const char *attrName) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1)
    return flagcxSuccess;
  for (int i = index + 1; i < node->nAttrs; i++) {
    strcpy(node->attrs[i - 1].key, node->attrs[i].key);
    strcpy(node->attrs[i - 1].value, node->attrs[i].value);
  }
  node->nAttrs--;
  return flagcxSuccess;
}

static flagcxResult_t xmlGetSub(struct flagcxXmlNode *node, const char *subName,
                                struct flagcxXmlNode **sub) {
  *sub = NULL;
  for (int s = 0; s < node->nSubs; s++) {
    if (strcmp(node->subs[s]->name, subName) == 0) {
      *sub = node->subs[s];
      return flagcxSuccess;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t xmlGetSubKv(struct flagcxXmlNode *node,
                                  const char *subName,
                                  struct flagcxXmlNode **sub,
                                  const char *attrName, const char *attrValue) {
  *sub = NULL;
  for (int s = 0; s < node->nSubs; s++) {
    struct flagcxXmlNode *subNode = node->subs[s];
    if (strcmp(subNode->name, subName) == 0) {
      const char *value;
      FLAGCXCHECK(xmlGetAttr(subNode, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *sub = node->subs[s];
        return flagcxSuccess;
      }
    }
  }
  return flagcxSuccess;
}
static flagcxResult_t xmlGetSubKvInt(struct flagcxXmlNode *node,
                                     const char *subName,
                                     struct flagcxXmlNode **sub,
                                     const char *attrName,
                                     const int attrValue) {
  char strValue[10];
  snprintf(strValue, 10, "%d", attrValue);
  FLAGCXCHECK(xmlGetSubKv(node, subName, sub, attrName, strValue));
  return flagcxSuccess;
}

static flagcxResult_t xmlAddNode(struct flagcxXml *xml,
                                 struct flagcxXmlNode *parent,
                                 const char *subName,
                                 struct flagcxXmlNode **sub) {
  if (xml->maxIndex == xml->maxNodes) {
    WARN("Error : too many XML nodes (max %d)", xml->maxNodes);
    return flagcxInternalError;
  }
  struct flagcxXmlNode *s = xml->nodes + xml->maxIndex++;
  s->nSubs = 0;
  s->nAttrs = 0;
  *sub = s;
  s->parent = parent;
  if (parent) {
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : too many XML subnodes (max %d)", MAX_SUBS);
      return flagcxInternalError;
    }
    parent->subs[parent->nSubs++] = s;
  }
  strncpy(s->name, subName, MAX_STR_LEN);
  s->name[MAX_STR_LEN] = '\0';
  return flagcxSuccess;
}

static flagcxResult_t xmlRemoveNode(struct flagcxXmlNode *node) {
  node->type = NODE_TYPE_NONE;
  struct flagcxXmlNode *parent = node->parent;
  if (parent == NULL)
    return flagcxSuccess;
  int shift = 0;
  for (int s = 0; s < parent->nSubs; s++) {
    if (parent->subs[s] == node)
      shift = 1;
    else if (shift)
      parent->subs[s - 1] = parent->subs[s];
  }
  parent->nSubs--;
  return flagcxSuccess;
}

// Dictionary for STR -> INT conversions. No dictionary size information,
// there needs to be a last element with str == NULL.
struct kvDict {
  const char *str;
  int value;
};

static flagcxResult_t kvConvertToInt(const char *str, int *value,
                                     struct kvDict *dict) {
  struct kvDict *d = dict;
  while (d->str) {
    if (strncmp(str, d->str, strlen(d->str)) == 0) {
      *value = d->value;
      return flagcxSuccess;
    }
    d++;
  }
  INFO(FLAGCX_GRAPH,
       "KV Convert to int : could not find value of '%s' in dictionary, "
       "falling back to %d",
       str, d->value);
  *value = d->value;
  return flagcxSuccess;
}
static flagcxResult_t kvConvertToStr(int value, const char **str,
                                     struct kvDict *dict) {
  struct kvDict *d = dict;
  while (d->str) {
    if (value == d->value) {
      *str = d->str;
      return flagcxSuccess;
    }
    d++;
  }
  WARN("KV Convert to str : could not find value %d in dictionary", value);
  return flagcxInternalError;
}

#endif // FLAGCX_XML_H_
