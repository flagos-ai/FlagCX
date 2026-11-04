#pragma once

#include "perf_common.h"

static inline unsigned char perfAlltoallByte(int source, int target,
                                             size_t offset, size_t size) {
  return static_cast<unsigned char>(1 + (source * 43 + target * 71 +
                                         (offset % 251) * 17 +
                                         (size / (1024 * 1024)) % 251) %
                                            251);
}

static inline void perfAlltoallFill(unsigned char *buffer, size_t length,
                                    int source, int target, size_t size) {
  enum { blockSize = 4096 };
  unsigned char pattern[blockSize];
  for (size_t i = 0; i < blockSize; ++i)
    pattern[i] = perfAlltoallByte(source, target, i, size);
  for (size_t offset = 0; offset < length; offset += blockSize) {
    const size_t bytes =
        length - offset < blockSize ? length - offset : blockSize;
    memcpy(buffer + offset, pattern, bytes);
    const size_t tagBytes = bytes < sizeof(uint64_t) ? bytes : sizeof(uint64_t);
    const uint64_t block = offset / blockSize;
    for (size_t i = 0; i < tagBytes; ++i)
      buffer[offset + i] = static_cast<unsigned char>(block >> (i * 8));
  }
}

static inline void perfAlltoallCheck(const unsigned char *buffer, size_t length,
                                     int rank, int source, size_t base,
                                     size_t size, const char *operation) {
  enum { blockSize = 4096 };
  unsigned char pattern[blockSize];
  for (size_t i = 0; i < blockSize; ++i)
    pattern[i] = perfAlltoallByte(source, rank, i, size);
  for (size_t offset = 0; offset < length; offset += blockSize) {
    const size_t bytes =
        length - offset < blockSize ? length - offset : blockSize;
    const size_t tagBytes = bytes < sizeof(uint64_t) ? bytes : sizeof(uint64_t);
    const uint64_t block = offset / blockSize;
    for (size_t i = 0; i < tagBytes; ++i) {
      const unsigned char expected =
          static_cast<unsigned char>(block >> (i * 8));
      if (buffer[offset + i] != expected) {
        fprintf(
            stderr,
            "%s rank=%d source=%d offset=%zu expected=0x%02x actual=0x%02x\n",
            operation, rank, source, base + offset + i, expected,
            buffer[offset + i]);
        perfAbort("received data mismatch", __FILE__, __LINE__);
      }
    }
    if (bytes > tagBytes && memcmp(buffer + offset + tagBytes,
                                   pattern + tagBytes, bytes - tagBytes) != 0) {
      for (size_t i = tagBytes; i < bytes; ++i) {
        if (buffer[offset + i] != pattern[i]) {
          fprintf(
              stderr,
              "%s rank=%d source=%d offset=%zu expected=0x%02x actual=0x%02x\n",
              operation, rank, source, base + offset + i, pattern[i],
              buffer[offset + i]);
          perfAbort("received data mismatch", __FILE__, __LINE__);
        }
      }
    }
  }
}

static inline void perfAlltoallCheckEmpty(const unsigned char *buffer,
                                          size_t length, int rank, size_t base,
                                          const char *operation) {
  enum { blockSize = 4096 };
  const unsigned char empty[blockSize] = {};
  for (size_t offset = 0; offset < length; offset += blockSize) {
    const size_t bytes =
        length - offset < blockSize ? length - offset : blockSize;
    if (memcmp(buffer + offset, empty, bytes) != 0) {
      for (size_t i = 0; i < bytes; ++i) {
        if (buffer[offset + i] != 0) {
          fprintf(stderr,
                  "%s rank=%d unexpected write at offset=%zu actual=0x%02x\n",
                  operation, rank, base + offset + i, buffer[offset + i]);
          perfAbort("received data outside active region", __FILE__, __LINE__);
        }
      }
    }
  }
}
