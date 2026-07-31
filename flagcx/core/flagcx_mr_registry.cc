/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Unified MR Registry implementation.
 * Sorted flat array + binary search + pthread_rwlock.
 ************************************************************************/

#include "flagcx_mr_registry.h"

#include <stdlib.h>
#include <string.h>

#include "debug.h" // WARN, INFO

#define MR_REGISTRY_INITIAL_CAPACITY 16

/* ───── Global instance ───── */

struct flagcxMrRegistry *flagcxGlobalMrRegistry = NULL;

/* ───── Internal helpers ───── */

static inline int ownerBitToIdx(uint32_t ownerBit) {
  switch (ownerBit) {
    case FLAGCX_MR_OWNER_P2P:
      return FLAGCX_MR_OWNER_IDX_P2P;
    case FLAGCX_MR_OWNER_COLL:
      return FLAGCX_MR_OWNER_IDX_COLL;
    case FLAGCX_MR_OWNER_RMA:
      return FLAGCX_MR_OWNER_IDX_RMA;
    default:
      return -1;
  }
}

/*
 * Binary search: find the largest index i where entries[i].baseAddr <= addr.
 * Returns -1 if all entries have baseAddr > addr.
 */
static int bsearchContaining(const struct flagcxMrEntry *entries, int count,
                             uintptr_t addr) {
  int lo = 0, hi = count - 1, result = -1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (entries[mid].baseAddr <= addr) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result;
}

/*
 * Binary search: find exact match by baseAddr.
 * Returns index or -1 if not found.
 */
static int bsearchExact(const struct flagcxMrEntry *entries, int count,
                        uintptr_t addr) {
  int lo = 0, hi = count - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (entries[mid].baseAddr == addr)
      return mid;
    else if (entries[mid].baseAddr < addr)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

/*
 * Find insertion point: returns the index where a new entry with baseAddr=addr
 * should be inserted to maintain sorted order.
 */
static int findInsertionPoint(const struct flagcxMrEntry *entries, int count,
                              uintptr_t addr) {
  int lo = 0, hi = count;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (entries[mid].baseAddr < addr)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

static flagcxResult_t ensureCapacity(struct flagcxMrRegistry *reg) {
  if (reg->count < reg->capacity)
    return flagcxSuccess;

  int newCap =
      reg->capacity == 0 ? MR_REGISTRY_INITIAL_CAPACITY : reg->capacity * 2;
  struct flagcxMrEntry *newEntries = (struct flagcxMrEntry *)realloc(
      reg->entries, (size_t)newCap * sizeof(struct flagcxMrEntry));
  if (newEntries == NULL) {
    WARN("flagcxMrRegistry: realloc failed for capacity %d", newCap);
    return flagcxSystemError;
  }
  reg->entries = newEntries;
  reg->capacity = newCap;
  return flagcxSuccess;
}

/*
 * Copy extension data into caller-provided flagcxMrExtension slots.
 * Each non-NULL slot gets a by-value copy tagged with FLAGCX_MR_OWNER_*.
 * NULL slots in the array are skipped. NULL array means "no extensions needed."
 */
static inline void copyExtensions(const struct flagcxMrEntry *entry,
                                  struct flagcxMrExtension *outExts[]) {
  if (outExts == NULL)
    return;
  if (outExts[FLAGCX_MR_OWNER_IDX_P2P]) {
    if (entry->p2p) {
      outExts[FLAGCX_MR_OWNER_IDX_P2P]->type = FLAGCX_MR_OWNER_P2P;
      outExts[FLAGCX_MR_OWNER_IDX_P2P]->p2p = *entry->p2p;
    } else {
      outExts[FLAGCX_MR_OWNER_IDX_P2P]->type = FLAGCX_MR_OWNER_NONE;
    }
  }
  if (outExts[FLAGCX_MR_OWNER_IDX_COLL]) {
    if (entry->coll) {
      outExts[FLAGCX_MR_OWNER_IDX_COLL]->type = FLAGCX_MR_OWNER_COLL;
      outExts[FLAGCX_MR_OWNER_IDX_COLL]->coll = *entry->coll;
    } else {
      outExts[FLAGCX_MR_OWNER_IDX_COLL]->type = FLAGCX_MR_OWNER_NONE;
    }
  }
  if (outExts[FLAGCX_MR_OWNER_IDX_RMA]) {
    if (entry->rma) {
      outExts[FLAGCX_MR_OWNER_IDX_RMA]->type = FLAGCX_MR_OWNER_RMA;
      outExts[FLAGCX_MR_OWNER_IDX_RMA]->rma = *entry->rma;
    } else {
      outExts[FLAGCX_MR_OWNER_IDX_RMA]->type = FLAGCX_MR_OWNER_NONE;
    }
  }
}

static inline void sanitizeOutEntry(struct flagcxMrEntry *entry) {
  if (entry) {
    entry->p2p = NULL;
    entry->coll = NULL;
    entry->rma = NULL;
  }
}

static void freeEntryExtensions(struct flagcxMrEntry *entry) {
  if (entry->p2p) {
    free(entry->p2p);
    entry->p2p = NULL;
  }
  if (entry->coll) {
    free(entry->coll);
    entry->coll = NULL;
  }
  if (entry->rma) {
    free(entry->rma);
    entry->rma = NULL;
  }
}

/* ───── mrId index helpers ───── */

#define ID_INDEX_INITIAL_CAPACITY 16

/*
 * Append a new {mrId, baseAddr} pair to the end of idIndex.
 * Since mrIds are monotonically increasing, appending maintains sorted order.
 * Must be called under write lock.
 */
static flagcxResult_t idIndexAppend(struct flagcxMrRegistry *reg, uint64_t mrId,
                                    uintptr_t baseAddr) {
  if (reg->idCount > 0 && mrId <= reg->idIndex[reg->idCount - 1].mrId) {
    WARN("flagcxMrRegistry: non-monotonic mrId %lu (last %lu)",
         (unsigned long)mrId,
         (unsigned long)reg->idIndex[reg->idCount - 1].mrId);
    return flagcxInternalError;
  }
  if (reg->idCount >= reg->idCapacity) {
    int newCap =
        reg->idCapacity == 0 ? ID_INDEX_INITIAL_CAPACITY : reg->idCapacity * 2;
    struct flagcxMrIdEntry *newIdx = (struct flagcxMrIdEntry *)realloc(
        reg->idIndex, (size_t)newCap * sizeof(struct flagcxMrIdEntry));
    if (newIdx == NULL) {
      WARN("flagcxMrRegistry: idIndex realloc failed for capacity %d", newCap);
      return flagcxSystemError;
    }
    reg->idIndex = newIdx;
    reg->idCapacity = newCap;
  }
  reg->idIndex[reg->idCount].mrId = mrId;
  reg->idIndex[reg->idCount].baseAddr = baseAddr;
  reg->idCount++;
  return flagcxSuccess;
}

/*
 * Binary search idIndex for mrId, remove entry with memmove.
 * Must be called under write lock.
 */
static void idIndexRemove(struct flagcxMrRegistry *reg, uint64_t mrId) {
  int lo = 0, hi = reg->idCount - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (reg->idIndex[mid].mrId == mrId) {
      if (mid < reg->idCount - 1) {
        memmove(&reg->idIndex[mid], &reg->idIndex[mid + 1],
                (size_t)(reg->idCount - 1 - mid) *
                    sizeof(struct flagcxMrIdEntry));
      }
      reg->idCount--;
      return;
    } else if (reg->idIndex[mid].mrId < mrId) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
}

/*
 * Binary search idIndex for mrId, return associated baseAddr.
 * Returns 0 if not found (valid baseAddr is never 0 for real registrations).
 */
static uintptr_t idIndexFindBaseAddr(const struct flagcxMrRegistry *reg,
                                     uint64_t mrId) {
  int lo = 0, hi = reg->idCount - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (reg->idIndex[mid].mrId == mrId)
      return reg->idIndex[mid].baseAddr;
    else if (reg->idIndex[mid].mrId < mrId)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return 0;
}

/* ───── Lifecycle ───── */

flagcxResult_t flagcxMrRegistryCreate(struct flagcxMrRegistry **reg) {
  struct flagcxMrRegistry *r =
      (struct flagcxMrRegistry *)calloc(1, sizeof(struct flagcxMrRegistry));
  if (r == NULL)
    return flagcxSystemError;

  r->entries = NULL;
  r->count = 0;
  r->capacity = 0;
  r->nextId = 1;
  r->idIndex = NULL;
  r->idCount = 0;
  r->idCapacity = 0;

  if (pthread_rwlock_init(&r->rwlock, NULL) != 0) {
    free(r);
    return flagcxSystemError;
  }

  *reg = r;
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryDestroy(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxSuccess;

  /* Free all extension structs */
  for (int i = 0; i < reg->count; i++) {
    freeEntryExtensions(&reg->entries[i]);
  }

  free(reg->entries);
  free(reg->idIndex);
  pthread_rwlock_destroy(&reg->rwlock);
  free(reg);
  return flagcxSuccess;
}

/* ───── Registration ───── */

flagcxResult_t flagcxMrRegistryRegister(struct flagcxMrRegistry *reg,
                                        uintptr_t addr, size_t size, int ibDevN,
                                        int ptrType, uint32_t ownerBit,
                                        void *mhandle, void *ext,
                                        uint64_t *outId) {
  if (reg == NULL || size == 0 || addr == 0)
    return flagcxInternalError;

  /* Reject regions that wrap the address space */
  if (addr + size < addr)
    return flagcxInternalError;

  int ownerIdx = ownerBitToIdx(ownerBit);
  if (ownerIdx < 0)
    return flagcxInternalError;

  pthread_rwlock_wrlock(&reg->rwlock);

  /* Check for exact baseAddr match */
  int exactIdx = bsearchExact(reg->entries, reg->count, addr);
  if (exactIdx >= 0) {
    struct flagcxMrEntry *existing = &reg->entries[exactIdx];

    /* Size mismatch from another owner is an error */
    if (existing->size != size) {
      WARN(
          "flagcxMrRegistry: addr 0x%lx size mismatch: existing %zu vs new %zu",
          (unsigned long)addr, existing->size, size);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }

    /* ibDevN / ptrType mismatch from another owner is an error */
    if (existing->ibDevN != ibDevN || existing->ptrType != ptrType) {
      WARN("flagcxMrRegistry: addr 0x%lx metadata mismatch: "
           "ibDevN %d vs %d, ptrType %d vs %d",
           (unsigned long)addr, existing->ibDevN, ibDevN, existing->ptrType,
           ptrType);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }

    /* Already owned by this subsystem */
    if (existing->ownerMask & ownerBit) {
      /* Update mhandle if it changed */
      if (existing->mhandles[ownerIdx] != mhandle && mhandle != NULL) {
        INFO(FLAGCX_REG,
             "MrRegistry: updating mhandle for addr 0x%lx owner 0x%x",
             (unsigned long)addr, ownerBit);
        existing->mhandles[ownerIdx] = mhandle;
      }
      /* Replace extension if caller provided a new one */
      if (ext != NULL) {
        switch (ownerBit) {
          case FLAGCX_MR_OWNER_P2P: {
            /* Preserve the assigned mrId across ext replacement.
             * Reject if caller provides a conflicting non-zero mrId. */
            uint64_t prevMrId = existing->p2p ? existing->p2p->mrId : 0;
            struct flagcxMrP2pExt *newP2p = (struct flagcxMrP2pExt *)ext;
            if (newP2p->mrId != 0 && prevMrId != 0 &&
                newP2p->mrId != prevMrId) {
              WARN("flagcxMrRegistry: mrId conflict on ext replacement: "
                   "existing %lu vs new %lu",
                   (unsigned long)prevMrId, (unsigned long)newP2p->mrId);
              pthread_rwlock_unlock(&reg->rwlock);
              return flagcxInternalError;
            }
            free(existing->p2p);
            existing->p2p = newP2p;
            if (existing->p2p->mrId == 0)
              existing->p2p->mrId = prevMrId;
            break;
          }
          case FLAGCX_MR_OWNER_COLL:
            free(existing->coll);
            existing->coll = (struct flagcxMrCollExt *)ext;
            break;
          case FLAGCX_MR_OWNER_RMA:
            free(existing->rma);
            existing->rma = (struct flagcxMrRmaExt *)ext;
            break;
        }
      }
      if (outId != NULL) {
        *outId = (ownerBit == FLAGCX_MR_OWNER_P2P && existing->p2p)
                     ? existing->p2p->mrId
                     : 0;
      }
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxSuccess;
    }

    /* Add new owner — ext is required for new ownership */
    if (ext == NULL) {
      WARN(
          "flagcxMrRegistry: NULL ext when adding new owner 0x%x to addr 0x%lx",
          ownerBit, (unsigned long)addr);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }
    existing->ownerMask |= ownerBit;
    existing->mhandles[ownerIdx] = mhandle;

    switch (ownerBit) {
      case FLAGCX_MR_OWNER_P2P:
        existing->p2p = (struct flagcxMrP2pExt *)ext;
        if (existing->p2p) {
          if (existing->p2p->mrId == 0)
            existing->p2p->mrId = reg->nextId++;
          if (existing->p2p->mrId >= reg->nextId)
            reg->nextId = existing->p2p->mrId + 1;
          flagcxResult_t appendRes =
              idIndexAppend(reg, existing->p2p->mrId, addr);
          if (appendRes != flagcxSuccess) {
            /* Roll back: remove P2P ownership, caller retains ext */
            existing->p2p = NULL;
            existing->ownerMask &= ~ownerBit;
            existing->mhandles[ownerIdx] = NULL;
            pthread_rwlock_unlock(&reg->rwlock);
            return appendRes;
          }
          if (outId)
            *outId = existing->p2p->mrId;
        }
        break;
      case FLAGCX_MR_OWNER_COLL:
        existing->coll = (struct flagcxMrCollExt *)ext;
        if (outId)
          *outId = reg->nextId++;
        break;
      case FLAGCX_MR_OWNER_RMA:
        existing->rma = (struct flagcxMrRmaExt *)ext;
        if (outId)
          *outId = reg->nextId++;
        break;
    }

    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxSuccess;
  }

  /* No exact match — check for overlap at insertion point */
  int pos = findInsertionPoint(reg->entries, reg->count, addr);

  /* Check left neighbor overlap (subtraction-based to avoid overflow) */
  if (pos > 0) {
    struct flagcxMrEntry *left = &reg->entries[pos - 1];
    if (addr - left->baseAddr < left->size) {
      WARN("flagcxMrRegistry: overlap with left neighbor [0x%lx, +%zu) vs new "
           "[0x%lx, +%zu)",
           (unsigned long)left->baseAddr, left->size, (unsigned long)addr,
           size);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }
  }

  /* Check right neighbor overlap (subtraction-based to avoid overflow) */
  if (pos < reg->count) {
    struct flagcxMrEntry *right = &reg->entries[pos];
    if (right->baseAddr - addr < size) {
      WARN("flagcxMrRegistry: overlap with right neighbor [0x%lx, +%zu) vs new "
           "[0x%lx, +%zu)",
           (unsigned long)right->baseAddr, right->size, (unsigned long)addr,
           size);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }
  }

  /* New entry requires an extension struct */
  if (ext == NULL) {
    WARN(
        "flagcxMrRegistry: NULL ext for new registration addr 0x%lx owner 0x%x",
        (unsigned long)addr, ownerBit);
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  /* Grow array if needed */
  flagcxResult_t res = ensureCapacity(reg);
  if (res != flagcxSuccess) {
    pthread_rwlock_unlock(&reg->rwlock);
    return res;
  }

  /* Shift entries right to make room */
  if (pos < reg->count) {
    memmove(&reg->entries[pos + 1], &reg->entries[pos],
            (size_t)(reg->count - pos) * sizeof(struct flagcxMrEntry));
  }

  /* Initialize new entry */
  struct flagcxMrEntry *entry = &reg->entries[pos];
  memset(entry, 0, sizeof(struct flagcxMrEntry));
  entry->baseAddr = addr;
  entry->size = size;
  entry->ibDevN = ibDevN;
  entry->ptrType = ptrType;
  entry->ownerMask = ownerBit;
  entry->mhandles[ownerIdx] = mhandle;

  switch (ownerBit) {
    case FLAGCX_MR_OWNER_P2P:
      entry->p2p = (struct flagcxMrP2pExt *)ext;
      if (entry->p2p) {
        /* Assign mrId from registry's monotonic counter if not pre-set */
        if (entry->p2p->mrId == 0)
          entry->p2p->mrId = reg->nextId++;
        if (entry->p2p->mrId >= reg->nextId)
          reg->nextId = entry->p2p->mrId + 1;
        flagcxResult_t appendRes = idIndexAppend(reg, entry->p2p->mrId, addr);
        if (appendRes != flagcxSuccess) {
          /* Roll back: remove the entry we just inserted */
          entry->p2p = NULL; /* caller retains ext ownership */
          if (pos < reg->count) {
            memmove(&reg->entries[pos], &reg->entries[pos + 1],
                    (size_t)(reg->count - pos) * sizeof(struct flagcxMrEntry));
          }
          pthread_rwlock_unlock(&reg->rwlock);
          return appendRes;
        }
        if (outId)
          *outId = entry->p2p->mrId;
      }
      break;
    case FLAGCX_MR_OWNER_COLL:
      entry->coll = (struct flagcxMrCollExt *)ext;
      if (outId)
        *outId = reg->nextId++;
      break;
    case FLAGCX_MR_OWNER_RMA:
      entry->rma = (struct flagcxMrRmaExt *)ext;
      if (outId)
        *outId = reg->nextId++;
      break;
  }

  reg->count++;
  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryDeregister(struct flagcxMrRegistry *reg,
                                          uintptr_t addr, uint32_t ownerBit,
                                          struct flagcxMrEntry *outEntry,
                                          void **outExt) {
  if (reg == NULL)
    return flagcxInternalError;

  int ownerIdx = ownerBitToIdx(ownerBit);
  if (ownerIdx < 0)
    return flagcxInternalError;

  pthread_rwlock_wrlock(&reg->rwlock);

  int idx = bsearchExact(reg->entries, reg->count, addr);
  if (idx < 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  struct flagcxMrEntry *entry = &reg->entries[idx];

  if (!(entry->ownerMask & ownerBit)) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  /* Copy out before modification */
  if (outEntry) {
    *outEntry = *entry;
    sanitizeOutEntry(outEntry);
  }

  /* Extract subsystem extension */
  void *ext = NULL;
  switch (ownerBit) {
    case FLAGCX_MR_OWNER_P2P:
      ext = entry->p2p;
      if (entry->p2p && entry->p2p->mrId != 0)
        idIndexRemove(reg, entry->p2p->mrId);
      entry->p2p = NULL;
      break;
    case FLAGCX_MR_OWNER_COLL:
      ext = entry->coll;
      entry->coll = NULL;
      break;
    case FLAGCX_MR_OWNER_RMA:
      ext = entry->rma;
      entry->rma = NULL;
      break;
  }
  if (outExt)
    *outExt = ext;
  else
    free(ext);

  entry->ownerMask &= ~ownerBit;
  entry->mhandles[ownerIdx] = NULL;

  /* If no owners remain, remove entry from array */
  if (entry->ownerMask == 0) {
    freeEntryExtensions(entry);
    if (idx < reg->count - 1) {
      memmove(&reg->entries[idx], &reg->entries[idx + 1],
              (size_t)(reg->count - 1 - idx) * sizeof(struct flagcxMrEntry));
    }
    reg->count--;
  }

  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

/* ───── Lookup ───── */

flagcxResult_t flagcxMrRegistryLookup(struct flagcxMrRegistry *reg,
                                      uintptr_t addr,
                                      struct flagcxMrEntry *outEntry,
                                      struct flagcxMrExtension *outExts[]) {
  if (reg == NULL || outEntry == NULL)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  if (reg->count == 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  int idx = bsearchContaining(reg->entries, reg->count, addr);
  if (idx < 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  struct flagcxMrEntry *entry = &reg->entries[idx];
  if (addr >= entry->baseAddr && (addr - entry->baseAddr) < entry->size) {
    *outEntry = *entry;
    copyExtensions(entry, outExts);
    sanitizeOutEntry(outEntry);
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxSuccess;
  }

  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxInternalError;
}

flagcxResult_t flagcxMrRegistryFindExact(struct flagcxMrRegistry *reg,
                                         uintptr_t addr,
                                         struct flagcxMrEntry *outEntry,
                                         struct flagcxMrExtension *outExts[]) {
  if (reg == NULL || outEntry == NULL)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  int idx = bsearchExact(reg->entries, reg->count, addr);
  if (idx < 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  *outEntry = reg->entries[idx];
  copyExtensions(&reg->entries[idx], outExts);
  sanitizeOutEntry(outEntry);
  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryLookupById(struct flagcxMrRegistry *reg,
                                          uint64_t mrId,
                                          struct flagcxMrEntry *outEntry,
                                          struct flagcxMrExtension *outExts[]) {
  if (reg == NULL || outEntry == NULL)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  /* O(log n) lookup via mrId index → baseAddr → main entries[] */
  uintptr_t baseAddr = idIndexFindBaseAddr(reg, mrId);
  if (baseAddr == 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  int idx = bsearchExact(reg->entries, reg->count, baseAddr);
  if (idx < 0 || !reg->entries[idx].p2p) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  *outEntry = reg->entries[idx];
  copyExtensions(&reg->entries[idx], outExts);
  sanitizeOutEntry(outEntry);
  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t
flagcxMrRegistryFindByHandle(struct flagcxMrRegistry *reg, int ownerIdx,
                             void *mhandle, struct flagcxMrEntry *outEntry,
                             struct flagcxMrExtension *outExts[]) {
  if (reg == NULL || outEntry == NULL || mhandle == NULL)
    return flagcxInternalError;
  if (ownerIdx < 0 || ownerIdx >= FLAGCX_MR_OWNER_COUNT)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  for (int i = 0; i < reg->count; i++) {
    if (reg->entries[i].mhandles[ownerIdx] == mhandle) {
      *outEntry = reg->entries[i];
      copyExtensions(&reg->entries[i], outExts);
      sanitizeOutEntry(outEntry);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxSuccess;
    }
  }

  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxInternalError;
}

/* ───── Iteration support ───── */

flagcxResult_t flagcxMrRegistryRdLock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  if (pthread_rwlock_rdlock(&reg->rwlock) != 0)
    return flagcxInternalError;
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryRdUnlock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  if (pthread_rwlock_unlock(&reg->rwlock) != 0)
    return flagcxInternalError;
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryWrLock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  if (pthread_rwlock_wrlock(&reg->rwlock) != 0)
    return flagcxInternalError;
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryWrUnlock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  if (pthread_rwlock_unlock(&reg->rwlock) != 0)
    return flagcxInternalError;
  return flagcxSuccess;
}

int flagcxMrRegistryCount(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return 0;
  return reg->count;
}

struct flagcxMrEntry *flagcxMrRegistryEntries(struct flagcxMrRegistry *reg) {
  if (reg == NULL || reg->count == 0)
    return NULL;
  return reg->entries;
}

/* ───── Global instance management ───── */

static pthread_mutex_t gRegistryMutex = PTHREAD_MUTEX_INITIALIZER;
static int gRegistryRefCount = 0;

flagcxResult_t flagcxMrRegistryGlobalInit(void) {
  pthread_mutex_lock(&gRegistryMutex);
  if (gRegistryRefCount == 0) {
    flagcxResult_t res = flagcxMrRegistryCreate(&flagcxGlobalMrRegistry);
    if (res != flagcxSuccess) {
      pthread_mutex_unlock(&gRegistryMutex);
      return res;
    }
  }
  gRegistryRefCount++;
  pthread_mutex_unlock(&gRegistryMutex);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryGlobalRelease(void) {
  pthread_mutex_lock(&gRegistryMutex);
  if (gRegistryRefCount <= 0) {
    pthread_mutex_unlock(&gRegistryMutex);
    return flagcxInternalError;
  }
  gRegistryRefCount--;
  if (gRegistryRefCount == 0) {
    flagcxMrRegistryDestroy(flagcxGlobalMrRegistry);
    flagcxGlobalMrRegistry = NULL;
  }
  pthread_mutex_unlock(&gRegistryMutex);
  return flagcxSuccess;
}
