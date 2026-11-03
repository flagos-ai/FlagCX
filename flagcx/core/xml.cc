#include "xml.h"
#include "core.h"
#include "flagcx_common.h"
#include <ctype.h>
#include <fcntl.h>
#include <float.h>
#include <map>
#include <queue>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <cpuid.h>
#endif

/*******************/
/* XML File Parser */
/*******************/
flagcxResult_t xmlGetChar(FILE *file, char *c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

static int pciBusIdDomainDigits(const char *busId) {
  if (busId == NULL)
    return 0;
  size_t length = strlen(busId);
  int domainDigits = length == 12 ? 4 : (length == 16 ? 8 : 0);
  if (domainDigits == 0 || busId[domainDigits] != ':' ||
      busId[domainDigits + 3] != ':' || busId[domainDigits + 6] != '.')
    return 0;
  for (size_t i = 0; i < length; i++) {
    if (i == (size_t)domainDigits || i == (size_t)domainDigits + 3 ||
        i == (size_t)domainDigits + 6)
      continue;
    if (!isxdigit((unsigned char)busId[i]))
      return 0;
  }
  return domainDigits;
}
static flagcxResult_t getPciPath(const char *busId, char **path) {
  if (path == NULL)
    return flagcxInvalidArgument;
  *path = NULL;
  int domainDigits = pciBusIdDomainDigits(busId);
  if (domainDigits == 0)
    return flagcxInvalidArgument;
  char lower[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE];
  size_t length = strlen(busId);
  for (size_t i = 0; i < length; i++)
    lower[i] = (char)tolower((unsigned char)busId[i]);
  lower[length] = '\0';
  char busPath[128];
  int written =
      snprintf(busPath, sizeof(busPath), "/sys/class/pci_bus/%.*s/../../%s",
               domainDigits + 3, lower, lower);
  if (written < 0 || (size_t)written >= sizeof(busPath))
    return flagcxInvalidArgument;
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t xmlGetValue(FILE *file, char *value, char *last) {
  char c;
  FLAGCXCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      FLAGCXCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return flagcxSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return flagcxInternalError;
#endif
  }
  int o = 0;
  do {
    FLAGCXCHECK(xmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o - 1] = '\0';
  FLAGCXCHECK(xmlGetChar(file, last));
  return flagcxSuccess;
}

flagcxResult_t xmlGetToken(FILE *file, char *name, char *value, char *last) {
  char c;
  char *ptr = name;
  int o = 0;
  do {
    FLAGCXCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return flagcxInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN - 1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return flagcxInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o - 1] = '\0';
  *last = c;
  return flagcxSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c)                                                     \
  do {                                                                         \
    s[0] = s[1];                                                               \
    s[1] = s[2];                                                               \
    s[2] = c;                                                                  \
  } while (0)
flagcxResult_t xmlSkipComment(FILE *file, char *start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i = 0; i < strlen(start); i++)
    SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return flagcxInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return flagcxSuccess;
}

flagcxResult_t xmlGetNode(FILE *file, struct flagcxXmlNode *node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0)
      return flagcxSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return flagcxInternalError;
  }
  // Read XML element name
  FLAGCXCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    FLAGCXCHECK(xmlSkipComment(file, node->name + 3, c));
    return xmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    FLAGCXCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c,
           node->name);
      return flagcxInternalError;
    }
    return flagcxSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    FLAGCXCHECK(
        xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(FLAGCX_GRAPH, "XML Parse : Ignoring extra attributes (max %d)",
           MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an
      // extra one.
    } else
      a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    FLAGCXCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetStrFromSys(const char *path, const char *fileName,
                                       char *strValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE *file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(strValue + offset, 1, MAX_STR_LEN - offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(FLAGCX_GRAPH, "Topology detection : could not read %s, ignoring",
         filePath);
  } else {
    strValue[offset - 1] = '\0';
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoSetAttrFromSys(struct flagcxXmlNode *pciNode,
                                        const char *path, const char *fileName,
                                        const char *attrName) {
  char strValue[MAX_STR_LEN];
  FLAGCXCHECK(flagcxTopoGetStrFromSys(path, fileName, strValue));
  if (strValue[0] != '\0') {
    FLAGCXCHECK(xmlSetAttr(pciNode, attrName, strValue));
  }
  INFO(FLAGCX_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName,
       strValue);
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetPciNode(struct flagcxXml *xml, const char *busId,
                                    struct flagcxXmlNode **pciNode) {
  FLAGCXCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    FLAGCXCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
    FLAGCXCHECK(xmlSetAttr(*pciNode, "busid", busId));
  }

  return flagcxSuccess;
}

// Check whether a path component is a four- or eight-digit-domain BDF.
int checkBDFFormat(char *bdf) { return pciBusIdDomainDigits(bdf) != 0; }

// TODO: it would be better if we have a device handle and can call APIs to get
// Apu information using that device handle
flagcxResult_t flagcxTopoGetXmlFromApu(struct flagcxXmlNode *pciNode,
                                       struct flagcxXml *xml,
                                       struct flagcxXmlNode **apuNodeRet) {
  struct flagcxXmlNode *apuNode = NULL;
  FLAGCXCHECK(xmlGetSub(pciNode, "apu", &apuNode));
  if (apuNode == NULL) {
    FLAGCXCHECK(xmlAddNode(xml, pciNode, "apu", &apuNode));
  }
  // TODO: maybe add vendor information to the xml node in the future
  *apuNodeRet = apuNode;
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetXmlFromCpu(struct flagcxXmlNode *cpuNode,
                                       struct flagcxXml *xml) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char *numaId;
    FLAGCXCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return flagcxInternalError;
    }
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node0000";
    sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaId);
    FLAGCXCHECK(
        flagcxTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    FLAGCXCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    FLAGCXCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    FLAGCXCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    FLAGCXCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId : 4;
        unsigned modelId : 4;
        unsigned familyId : 4;
        unsigned processorType : 2;
        unsigned resv0 : 2;
        unsigned extModelId : 4;
        unsigned extFamilyId : 8;
        unsigned resv1 : 4;
      };
      uint32_t val;
    } cpuid1;
    unsigned unused;
    __cpuid(1, cpuid1.val, unused, unused, unused);
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    FLAGCXCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    FLAGCXCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetXmlFromSys(struct flagcxXmlNode *pciNode,
                                       struct flagcxXml *xml) {
  const char *busId;
  FLAGCXCHECK(xmlGetAttr(pciNode, "busid", &busId));
  char *path = NULL;
  getPciPath(busId, &path);

  if (path) {
    FLAGCXCHECK(flagcxTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
  }
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "device", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "device", "device");
  }
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "subsystem_vendor",
                               "subsystem_vendor");
  }
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "subsystem_device",
                               "subsystem_device");
  }
  flagcxTopoSetAttrFromSys(pciNode, path, "max_link_speed", "link_speed");
  flagcxTopoSetAttrFromSys(pciNode, path, "max_link_width", "link_width");

  struct flagcxXmlNode *parent = pciNode->parent;
  if (parent == NULL) {
    // try to find the parent along the pci path
    if (path) {
      // Save that for later in case next step is a CPU
      char numaIdStr[MAX_STR_LEN];
      FLAGCXCHECK(flagcxTopoGetStrFromSys(path, "numa_node", numaIdStr));

      // Go up one level in the PCI tree. Rewind two "/" and follow the upper
      // PCI switch, or stop if we reach a CPU root complex.
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path) - 1; parentOffset > 0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start > 0 && path[start] != '/')
            start--;
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path + start + 1) == 0) {
            // This a CPU root complex. Create a CPU tag and stop there.
            struct flagcxXmlNode *topNode;
            FLAGCXCHECK(xmlFindTag(xml, "system", &topNode));
            FLAGCXCHECK(
                xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
            if (parent == NULL) {
              FLAGCXCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
              FLAGCXCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
              FLAGCXCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
            }
          } else if (slashCount == 2) {
            // Continue on the upper PCI switch
            for (int i = strlen(path) - 1; i > 0; i--) {
              if (path[i] == '/') {
                FLAGCXCHECK(
                    xmlFindTagKv(xml, "pci", &parent, "busid", path + i + 1));
                if (parent == NULL) {
                  FLAGCXCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                  FLAGCXCHECK(xmlSetAttr(parent, "busid", path + i + 1));
                }
                break;
              }
            }
          }
        }
        if (parent)
          break;
      }
    } else {
      // No information on /sys, attach GPU to unknown CPU
      FLAGCXCHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
      if (parent == NULL) {
        struct flagcxXmlNode *topNode;
        FLAGCXCHECK(xmlFindTag(xml, "system", &topNode));
        FLAGCXCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
        FLAGCXCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
        FLAGCXCHECK(xmlSetAttr(parent, "numaid", "-1"));
        FLAGCXCHECK(flagcxTopoGetXmlFromCpu(parent, xml));
      }
    }
    pciNode->parent = parent;
    // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
    int subIndex = parent->nSubs;
    const char *newBusId;
    FLAGCXCHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
    for (int s = 0; s < parent->nSubs; s++) {
      const char *busId;
      FLAGCXCHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
      if (busId != NULL && strcmp(newBusId, busId) < 0) {
        subIndex = s;
        break;
      }
    }
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
      return flagcxInternalError;
    }
    for (int s = parent->nSubs; s > subIndex; s--)
      parent->subs[s] = parent->subs[s - 1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  if (strcmp(parent->name, "pci") == 0) {
    FLAGCXCHECK(flagcxTopoGetXmlFromSys(parent, xml));
  } else if (strcmp(parent->name, "cpu") == 0) {
    FLAGCXCHECK(flagcxTopoGetXmlFromCpu(parent, xml));
  }
  free(path);
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoFillApu(struct flagcxXml *xml, const char *busId,
                                 struct flagcxXmlNode **gpuNode) {
  struct flagcxXmlNode *pciNode;
  INFO(FLAGCX_INIT, "creating xml pci node for busId [%s]", busId);
  FLAGCXCHECK(flagcxTopoGetPciNode(xml, busId, &pciNode));
  FLAGCXCHECK(flagcxTopoGetXmlFromSys(pciNode, xml));
  INFO(FLAGCX_INIT, "creating xml apu node for busId [%s]", busId);
  FLAGCXCHECK(flagcxTopoGetXmlFromApu(pciNode, xml, gpuNode));
  return flagcxSuccess;
}

// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
flagcxResult_t flagcxTopoGetSubsystem(const char *sysPath, char *subSys) {
  char subSysPath[PATH_MAX];
  sprintf(subSysPath, "%s/subsystem", sysPath);
  char *path = realpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--)
      ;
    strcpy(subSys, path + offset + 1);
    free(path);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoFillNet(struct flagcxXml *xml, const char *pciPath,
                                 const char *netName,
                                 struct flagcxXmlNode **netNode) {
  FLAGCXCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));
  if (*netNode != NULL)
    return flagcxSuccess;

  const char *pciSysPath = pciPath;
  if (pciSysPath) {
    char subSystem[PATH_MAX];
    INFO(FLAGCX_INIT, "gettting subsystem for pciPath [%s]", pciSysPath);
    FLAGCXCHECK(flagcxTopoGetSubsystem(pciSysPath, subSystem));
    if (strcmp(subSystem, "pci") != 0) {
      INFO(FLAGCX_GRAPH,
           "Topology detection: network path %s is not a PCI device (%s). "
           "Attaching to first CPU",
           pciSysPath, subSystem);
      pciSysPath = NULL;
    }
  }

  struct flagcxXmlNode *parent = NULL;
  if (pciSysPath) {
    INFO(FLAGCX_INIT, "getting parent pci node for nic");
    char busId[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE];
    const char *basename = strrchr(pciSysPath, '/');
    if (basename == NULL)
      return flagcxInvalidArgument;
    int written = snprintf(busId, sizeof(busId), "%s", basename + 1);
    if (written < 0 || (size_t)written >= sizeof(busId))
      return flagcxInvalidArgument;
    INFO(FLAGCX_INIT, "busId for parent pci node is [%s]", busId);
    FLAGCXCHECK(flagcxTopoGetPciNode(xml, busId, &parent));
    FLAGCXCHECK(flagcxTopoGetXmlFromSys(parent, xml));
  } else {
    // Virtual NIC, no PCI device, attach to first CPU
    FLAGCXCHECK(xmlFindTag(xml, "cpu", &parent));
  }

  struct flagcxXmlNode *nicNode = NULL;
  FLAGCXCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    FLAGCXCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // We know that this net does not exist yet (we searched for it at the
  // beginning of this function), so we can add it.
  FLAGCXCHECK(xmlAddNode(xml, nicNode, "net", netNode));
  FLAGCXCHECK(xmlSetAttr(*netNode, "name", netName));
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoDumpXmlRec(int indent, FILE *file,
                                    struct flagcxXmlNode *node) {
  for (int i = 0; i < indent; i++)
    fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a = 0; a < node->nAttrs; a++) {
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s = 0; s < node->nSubs; s++) {
      FLAGCXCHECK(flagcxTopoDumpXmlRec(indent + 2, file, node->subs[s]));
    }
    for (int i = 0; i < indent; i++)
      fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return flagcxSuccess;
}

typedef flagcxResult_t (*xmlHandlerFunc_t)(FILE *, struct flagcxXml *,
                                           struct flagcxXmlNode *);

struct xmlHandler {
  const char *name;
  xmlHandlerFunc_t func;
};

flagcxResult_t xmlLoadSub(FILE *file, struct flagcxXml *xml,
                          struct flagcxXmlNode *head,
                          struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE)
    return flagcxSuccess;
  while (1) {
    if (xml->maxIndex == xml->maxNodes) {
      WARN("Error : XML parser is limited to %d nodes", xml->maxNodes);
      return flagcxInternalError;
    }
    struct flagcxXmlNode *node = xml->nodes + xml->maxIndex;
    memset(node, 0, sizeof(struct flagcxXmlNode));
    FLAGCXCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return flagcxInternalError;
      } else {
        // All done
        return flagcxSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return flagcxInternalError;
      }
      return flagcxSuccess;
    }
    int found = 0;
    for (int h = 0; h < nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) {
          if (head->nSubs == MAX_SUBS) {
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return flagcxInternalError;
          }
          head->subs[head->nSubs++] = node;
        }
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        FLAGCXCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers)
        INFO(FLAGCX_GRAPH, "Ignoring element %s", node->name);
      FLAGCXCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

/****************************************/
/* Parser rules for our specific format */
/****************************************/
flagcxResult_t flagcxTopoXmlLoadGpu(FILE *file, struct flagcxXml *xml,
                                    struct flagcxXmlNode *head) {
  FLAGCXCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoXmlLoadSystem(FILE *file, struct flagcxXml *xml,
                                       struct flagcxXmlNode *head) {
  int version;
  FLAGCXCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != FLAGCX_TOPO_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version,
         FLAGCX_TOPO_XML_VERSION);
    return flagcxInvalidUsage;
  }
  const char *name;
  FLAGCXCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL)
    INFO(FLAGCX_GRAPH, "Loading topology %s", name);
  else
    INFO(FLAGCX_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = {{"gpu", flagcxTopoXmlLoadGpu}};
  FLAGCXCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetXmlFromFile(const char *xmlTopoFile,
                                        struct flagcxXml *xml, int warn) {
  FILE *file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    if (warn) {
      WARN("Could not open XML topology file %s : %s", xmlTopoFile,
           strerror(errno));
    }
    return flagcxSuccess;
  }
  INFO(FLAGCX_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = {{"system", flagcxTopoXmlLoadSystem}};
  xml->maxIndex = 0;
  FLAGCXCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoDumpXmlToFile(const char *xmlTopoFile,
                                       struct flagcxXml *xml) {
  // TODO: clear file content if file is not empty
  FILE *file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    WARN("Unable to open %s, not dumping topology.", xmlTopoFile);
    return flagcxSuccess;
  }
  FLAGCXCHECK(flagcxTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return flagcxSuccess;
}

flagcxResult_t xmlGetApuByIndex(struct flagcxXml *xml, int apu,
                                struct flagcxXmlNode **apuNode) {
  // iterate through all nodes in xml and find the apuNode with logical index ==
  // apu
  for (int i = 0; i < xml->maxIndex; i++) {
    struct flagcxXmlNode *n = xml->nodes + i;
    if (strcmp(n->name, "apu") == 0) {
      int value = 0;
      FLAGCXCHECK(xmlGetAttrInt(n, "dev", &value));
      if (value == apu) {
        *apuNode = n;
        return flagcxSuccess;
      }
    }
  }

  return flagcxSuccess;
}

flagcxResult_t xmlFindClosestNetUnderCpu(struct flagcxXml *xml,
                                         struct flagcxXmlNode *apuNode,
                                         struct flagcxXmlNode **retNet) {
  INFO(FLAGCX_INIT, "searching for local net node under one cpu node");
  std::queue<struct flagcxXmlNode *> nodeQueue;
  std::map<struct flagcxXmlNode *, bool> visited;
  nodeQueue.push(apuNode);
  visited[apuNode] = true;
  while (!nodeQueue.empty()) {
    struct flagcxXmlNode *node = nodeQueue.front();
    nodeQueue.pop();
    // INFO(FLAGCX_INIT, "node name = %s", node->name);
    if (strcmp(node->name, "system") == 0) {
      // do not go through root node, we are searching under one cpu node
      continue;
    }
    if (strcmp(node->name, "net") == 0) {
      // found a net node
      *retNet = node;
      break;
    }
    // push parent if parent is not visited
    if (node->parent && !visited[node->parent]) {
      nodeQueue.push(node->parent);
      visited[node->parent] = true;
    }
    // push children if children are not visited
    for (int i = 0; i < node->nSubs; i++) {
      if (!visited[node->subs[i]]) {
        nodeQueue.push(node->subs[i]);
        visited[node->subs[i]] = true;
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t xmlFindClosestNetUnderServer(struct flagcxXml *xml,
                                            struct flagcxXmlNode *apuNode,
                                            struct flagcxXmlNode **retNet) {
  INFO(FLAGCX_INIT, "searching for local net node under one server");
  std::queue<struct flagcxXmlNode *> nodeQueue;
  std::map<struct flagcxXmlNode *, bool> visited;
  nodeQueue.push(apuNode);
  visited[apuNode] = true;
  while (!nodeQueue.empty()) {
    struct flagcxXmlNode *node = nodeQueue.front();
    nodeQueue.pop();
    if (strcmp(node->name, "net") == 0) {
      // found a net node
      *retNet = node;
      break;
    }
    // push parent if parent is not visited
    if (node->parent && !visited[node->parent]) {
      nodeQueue.push(node->parent);
      visited[node->parent] = true;
    }
    // push children if children are not visited
    for (int i = 0; i < node->nSubs; i++) {
      if (!visited[node->subs[i]]) {
        nodeQueue.push(node->subs[i]);
        visited[node->subs[i]] = true;
      }
    }
  }
  return flagcxSuccess;
}
