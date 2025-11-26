#ifndef FLAGCX_REGPOOL_H
#define FLAGCX_REGPOOL_H

#include "check.h"
#include "device.h"
#include "flagcx.h"
#include "net.h"
#include "register.h"
#include <map>
#include <unistd.h>

class flagcxRegPool {
public:
  flagcxRegPool();
  ~flagcxRegPool();

  inline void getPagedAddr(void *data, size_t length, uintptr_t *beginAddr,
                           uintptr_t *endAddr);
  flagcxResult_t removeRegItemNetHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t registerBuffer(void *comm, void *data, size_t length);
  flagcxResult_t deregisterBuffer(void *comm, void *handle);
  std::map<uintptr_t, std::map<uintptr_t, flagcxRegItem *>> &getGlobalMap();
  flagcxRegItem *getItem(const void *comm, void *data);
  void dump();

private:
  std::map<uintptr_t, std::map<uintptr_t, flagcxRegItem *>>
      regMap; // <commPtr, <dataPtr, regItemPtr>>
  std::map<uintptr_t, std::list<flagcxRegItem>>
      regPool; // <commPtr, regItemList>
  uintptr_t pageSize;
};

extern flagcxRegPool globalRegPool;

#endif // FLAGCX_REGPOOL_H